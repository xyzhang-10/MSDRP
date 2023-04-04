import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from snfpy import snf
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
def train(model, loader, criterion, opt, device):
    model.train()


    for idx, data in enumerate(tqdm(loader, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。

        drug, cell, label,fusion= data

        output = model(drug, cell,device,fusion)
        loss = criterion(output, label.float().to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, loader, device):
    # rmse, _, _, _ = validate(model, val_loader, args.device)
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            drug, cell, label,fusion = data
            # drug[:, 2040:] = 0
            # cell[:, 2118:] = 0
            output = model(drug, cell,device,fusion)
            total_loss += F.mse_loss(output, label.float().to(device), reduction='sum')
            y_true.append(label)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(total_loss / len(loader.dataset))
    MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]


    return rmse, MAE, r2, r

class EarlyStopping():
    # EarlyStopping(mode='lower', patience=args.patience)
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


def Chebyshev_Distance(matrix):
    sim = np.zeros((len(matrix), len(matrix)))
    for A in range(len(matrix)):
        for B in range(len(matrix)):
            sim[A][B] = np.linalg.norm(matrix[A]-matrix[B],ord=np.inf)

    return sim


class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict,fused_network, IC):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        self.fused_network = fused_network
        # IC.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset
        self.drug_name = IC[:,1]
        self.Cell_line_name = IC[:,0]
        self.value = IC[:,2]
    def __len__(self):
        return len(self.value)
    def __getitem__(self, index):
        # self.cell[self.Cell_line_name[index]].adj_t = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])
        return (self.drug[int(self.drug_name[index])], self.cell[int(self.Cell_line_name[index])], self.value[index],self.fused_network[int(self.drug_name[index])])

def load_data(args):
    rawdata_dir = args.rawpath
    with open(args.rawpath+'IC50/samples_82833.pkl','rb') as f:
        final_sample = pickle.load(f)
    drug_features, cell_features,fused_network = read_raw_data(rawdata_dir)
    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    cell_features_matrix = cell_features[0]
    for i in range(1, len(cell_features)):
        cell_features_matrix = np.hstack((cell_features_matrix, cell_features[i]))


    train_set, val_test_set = train_test_split(final_sample, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)

    Dataset = MyDataset
    train_dataset = Dataset(drug_features_matrix, cell_features_matrix, fused_network,train_set)
    test_dataset = Dataset(drug_features_matrix, cell_features_matrix,fused_network, test_set)
    val_dataset = Dataset(drug_features_matrix, cell_features_matrix, fused_network,val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader,test_loader,val_loader


def read_raw_data(rawdata_dir):

    ###drug
    gii = open(rawdata_dir + 'IC50/' + 'ic_170drug_580cell.pkl', 'rb')
    ic = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_stitch.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()

    # gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_target.pkl', 'rb')
    # drug_target_sim = pickle.load(gii)
    # gii.close()

    gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_ADR.pkl', 'rb')
    drug_ADR_sim= pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_disease.pkl', 'rb')
    drug_dis_sim = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_target.pkl', 'rb')
    drug_gene_sim = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'SNF_smiles_drug_miRNA.pkl', 'rb')
    drug_miRNA_sim= pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_DFP.pkl', 'rb')
    daylight = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_ERGFP.pkl', 'rb')
    erg = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_ESPFP.pkl', 'rb')
    espf = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_ECFP.pkl', 'rb')
    morgan = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_PSFP.pkl', 'rb')
    pubchem = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'drug/' + 'drug_RDKFP.pkl', 'rb')
    rdk = pickle.load(gii)
    gii.close()

    daylight_sim = Chebyshev_Distance(daylight)
    erg_sim = Chebyshev_Distance(erg)
    espf_sim = Chebyshev_Distance(espf)
    morgan_sim =  Chebyshev_Distance(morgan)
    pubchem_sim = Chebyshev_Distance(pubchem)
    rdk_sim = Chebyshev_Distance(rdk)

    # drug_target_sim = Chebyshev_Distance(drug_target)
    # drug_ADR_sim =Chebyshev_Distance(drug_ADR)
    # drug_dis_sim = Chebyshev_Distance(drug_dis)
    # drug_gene_sim =Chebyshev_Distance(drug_gene)
    # drug_miRNA_sim =  Chebyshev_Distance(drug_miRNA)

    drug_ic_sim = Chebyshev_Distance(ic) # 170

    ####cell
    gii = open(rawdata_dir + 'cell/' + 'cn_580cell_706gene.pkl', 'rb')
    cn = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'cell/' + 'exp_580cell_706gene.pkl', 'rb')
    exp = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + 'cell/' + 'mu_580cell_706gene.pkl', 'rb')
    mu = pickle.load(gii)
    gii.close()

    drug_features, cell_features= [], []

    drug_features.append(drug_Tfeature_five)
    drug_features.append(drug_ADR_sim)
    drug_features.append(daylight_sim)
    drug_features.append(erg_sim)
    drug_features.append(espf_sim)
    drug_features.append(morgan_sim)
    drug_features.append(pubchem_sim)
    drug_features.append(rdk_sim)
    drug_features.append(drug_dis_sim)
    drug_features.append(drug_gene_sim)
    drug_features.append(drug_miRNA_sim)
    drug_features.append(drug_ic_sim)
    cell_ic_sim =Chebyshev_Distance(ic.T)

    cell_features.append(exp) # 706
    cell_features.append(mu)# 706
    cell_features.append(cn)# 706
    cell_features.append(cell_ic_sim)# 580

    fused_network = snf.snf(drug_features, K=20)

    return drug_features, cell_features, fused_network
