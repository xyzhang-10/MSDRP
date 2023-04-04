# Requirement

* python == 3.6
* pytorch == 1.10
* Numpy == 1.19.2
* scikit-learn == 0.24.2
* scipy == 1.5.4

# Files

## 1. data

### 1. cell  

cn_580cell_706gene.pkl :  the gene copy number matrix of cell lines. 

exp_580cell_706gene.pkl : the gene expression matrix of cell lines.

mu_580cell_706gene.pkl: the gene mutation matrix of cell lines.

### 2. drug

drug_DFP.pkl : the Daylight FingerPrints (DFP)  matrix of drugs.

drug_ECFP.pkl: the Extended-Connectivity FingerPrints (ECFP)  matrix of drugs.

drug_ERGFP.pkl: the Extended Reduced Graph FingerPrints (ERGFP) matrix of drugs.

drug_ESPFP.pkl: the Explainable Substructure Partition FingerPrints (ESPFP) matrix of drugs.

drug_PSFP.pkl : the PubChem Substructure FingerPrints (PSFP) matrix of drugs.

drug_RDKFP.pkl:  the RDKit 2D normalized FingerPrints (RDKFP) matrix of drugs.

SNF_smiles_drug_stitch.pkl : the chemical–chemical combined scores matrix obtained from filling process.

SNF_smiles_drug_ADR.pkl : the similarity matrix of drug-ADR associations obtained from filling process.

SNF_smiles_drug_disease.pkl: the similarity matrix of drug-disease associations obtained from filling process.

SNF_smiles_drug_miRNA.pkl: the similarity matrix of drug-microRNA associations obtained from filling process.

SNF_smiles_drug_target.pkl: the similarity matrix of drug-target interactions obtained from filling process.

### 3. IC50

ic_170drug_580cell.pkl: the IC50 values matrix between 170 drugs and 580 cells.

samples_82833.pkl: the 82833 IC50 values between 170 drugs and 580 cells. The first column is cell index. The second column is drug index. The second column is drug index. The last column is IC50 values.

## 2. snfpy

SNF similarity fusion algorithm package.

## 3. network.py

 This function contains the network framework of our entire model and is based on pytorch 1.10.

## 4. utils.py

This function contains the necessary processing subroutines.

# Train and test 

python main.py --mode mode --epochs number  --batch_size number  --rawpath path --weight_path path

mode: Set the mode to train or test, then you can train the model or test the model

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training.

rawpath: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

weight_path: Define the path to save the model.

All files of Data and Code should be stored in the same folder to run the model.



Example:

```
python main.py --mode train --epochs 200  --batch_size 128  --rawpath data/ --weight_path best
```



