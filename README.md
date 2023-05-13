# RGCN for predicting BBB Permeability of Drug Molecules

### Getting started

1. First, Install all the libraries and packages listed down under requiremets.txt.

2. Then download all the required large files from the link given in large_files.txt.

3. Calculate the drug-drug similarity by running `drug_similarity.py`. The results will be stored in `drug_similarity.csv`.

```
python drug_similarity.py
```

4. Calculate the drug-drug similarity by running `drug_protein_interaction.py`. The results will be stored in `drug_protein_interaction.csv`.

```
python drug_protein_interaction.py
```

5. Run `graph.py` to generate the drug features and to structure the data into graphs. Two graphs will be built and be saved separately in the `graph.pt` and `graph_drugism.pt` files.

   - They include the drug-drug similarity as the edges, and the Mordred descriptors as the node features.

```
python graph.py
```

6. Run `rgcn_drugism.py` to train and evaluate the RGCN model with `graph_drugism.pt` as the input.

```
python rgcn_drugism.py
```