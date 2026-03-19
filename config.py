cfg = dict(
    submodel='DrugGEN',
    inference_model='experiments/models/DrugGEN-akt1',
    inf_smiles='data/chembl_test.smi',
    train_smiles='data/chembl_train.smi',
    train_drug_smiles='data/akt_train.smi',
    max_atom=60,
)
