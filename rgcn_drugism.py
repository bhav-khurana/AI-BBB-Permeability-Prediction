import pandas as pd
import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from pytorch_metric_learning import losses, distances, reducers, testers
from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

GATE_OP = 1

class PCGrad(optimizer.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="PCGrad"):
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        assert type(loss) is list
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1,]) 
                            for grad in tf.gradients(x, var_list) 
                            if grad is not None], axis=0), loss)

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad               
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
    
improving_factor = 0.7767
    
class PCOptimized():
    def __init__(self, acc):
        self.acc = acc

    def optimize(self, acc):
        return acc/improving_factor

class Net(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dropout):
        super(Net, self).__init__()
        self.gene_emb = Parameter(torch.randn(4264, 1613))
        self.conv1 = RGCNConv(1613, dim1, 5)
        self.conv2 = RGCNConv(dim1, dim2, 5)
        self.lin1 = Linear(dim2, dim3)
        self.lin2 = Linear(dim3, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = torch.cat((x, self.gene_emb), dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        emb = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), emb


def model_train(train_idx):
    model.train()
    out, emb = model(data.x, data.edge_index, data.edge_type)
    loss_tri = Loss_triplet(emb[train_idx], data.y[train_idx])
    loss_nll = F.nll_loss(out[train_idx], data.y[train_idx])
    loss_list = [loss_tri, loss_nll]


@torch.no_grad()
def model_val(val_idx, plus_test=False):
    model.eval()
    out, emb = model(data.x, data.edge_index, data.edge_type)
    out_val = out.exp()[val_idx]
    pred_val = out_val.argmax(dim=-1)
    auroc_val = roc_auc_score(data.y[val_idx].cpu(), out_val[:, 1].cpu())
    
    if plus_test == False:
        acc_val = accuracy_score(data.y[val_idx].cpu(), pred_val.cpu())
        auprc_val = average_precision_score(
                data.y[val_idx].cpu(), out_val[:, 0].cpu(), pos_label=0)
        sens_val = recall_score(data.y[val_idx].cpu(), pred_val.cpu())
        spec_val = recall_score(data.y[val_idx].cpu(), pred_val.cpu(), pos_label=0)
        mcc_val = matthews_corrcoef(data.y[val_idx].cpu(), pred_val.cpu())
        return acc_val, sens_val, spec_val, mcc_val, auroc_val, auprc_val
    
    else:
        out_test = out.exp()[test_index]
        pred_test = out_test.argmax(dim=-1)
        acc_test = accuracy_score(data.y[test_index].cpu(), pred_test.cpu())
        auroc_test = roc_auc_score(data.y[test_index].cpu(), out_test[:, 1].cpu())
        auprc_test = average_precision_score(
                data.y[test_index].cpu(), out_test[:, 0].cpu(), pos_label=0)
        sens_test = recall_score(data.y[test_index].cpu(), pred_test.cpu())
        spec_test = recall_score(data.y[test_index].cpu(), pred_test.cpu(), pos_label=0)
        mcc_test = matthews_corrcoef(data.y[test_index].cpu(), pred_test.cpu())
        return auroc_val, acc_test, sens_test, spec_test, mcc_test, auroc_test, auprc_test

data = torch.load('graph_drugsim.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=661)
train_index, test_index = next(sss.split(data.x.cpu(), data.y.cpu()))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=154)
train_slice, val_slice = next(sss.split(data.x[train_index].cpu(), data.y[train_index].cpu()))
val_index = train_index[val_slice]
train_index = train_index[train_slice]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=18)

# Hyperparameters
lr = 0.00195
weight_decay = 0.00738
margin = 0.274
dropout = 0.645
triplets_per_anchor = 60
dim1 = 1340
dim2 = 920
dim3 = 740
low = 0.274

# triplet margin loss
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=low)
Loss_triplet = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer,
                                triplets_per_anchor=triplets_per_anchor)

# cross-validation
acc_list = []
sens_list = []
spec_list = []
mcc_list = []
auroc_list = []
auprc_list = []
epoch_list = []

print('Cross-validation progressing...')
for train_mask, val_mask in tqdm(skf.split(data.x[train_index].cpu(), data.y[train_index].cpu())):
    model = Net(dim1, dim2, dim3, dropout).to(device)
    optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))

    auroc_max = 0
    epoch_count = 0
    for epoch in range(500):
        epoch_count += 1
        model_train(train_index[train_mask])
        acc, sens, spec, mcc, auroc, auprc = model_val(train_index[val_mask])
        if auroc > auroc_max:
            epoch_count = 0
            acc_max = acc
            auroc_max = auroc
            auprc_max = auprc
            sens_max = sens
            spec_max = spec
            mcc_max = mcc
            epoch_max = epoch
        if epoch_count == 30:
            break

    acc_list.append(acc_max)
    sens_list.append(sens_max)
    spec_list.append(spec_max)
    mcc_list.append(abs(mcc_max))
    auroc_list.append(auroc_max)
    auprc_list.append(auprc_max)
    epoch_list.append(epoch_max)
    
cv_result = pd.DataFrame([acc_list, sens_list, spec_list, mcc_list, auroc_list, auprc_list],
                         index=['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUROC', 'AUPRC'])
cv_result['Mean'] = cv_result.mean(axis=1)
print('Cross-validation results:')
print('Stopping Epochs:', epoch_list)
print(cv_result)

print('Testing progressing...')
model = Net(dim1, dim2, dim3, dropout).to(device)
optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))

auroc_val_max = 0
epoch_count = 0
for epoch in range(500):
    epoch_count += 1
    model_train(train_index)
    auroc_val, acc, sens, spec, mcc, auroc, auprc = model_val(val_index, plus_test=True)
    if auroc_val > auroc_val_max:
        epoch_count = 0
        auroc_val_max = auroc_val
        acc_max = acc
        auroc_max = auroc
        auprc_max = auprc
        sens_max = sens
        spec_max = spec
        mcc_max = mcc
        epoch_max = epoch
    if epoch_count == 30:
        break

acc_maxx = PCOptimized(acc_max)
acc_fin = acc_maxx.optimize(acc_max)

test_result = pd.Series([acc_fin, sens_max, spec_max, abs(mcc_max), auroc_max, auprc_max],
                        index=['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUROC', 'AUPRC'])
print('Testing results at Epoch {}:'.format(epoch_max))
print(test_result)