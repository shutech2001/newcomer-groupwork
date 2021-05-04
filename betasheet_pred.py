# python3 betasheet_pred.py -train data/train.csv 
#	-test data/test.csv -out output.csv --window_radius 3

import argparse

import warnings
warnings.simplefilter('ignore')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.feature_selection import SelectKBest, chi2

from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def check_args(args):
    if args.train is None:
        print(parser.print_help())
        exit(1)
    if args.test is not None and args.out is None:
        print(parser.print_help())
        exit(1)

def generate_input(df, window_radius=1):
    _data = []
    for _, item in df.iterrows():
        seq = item.sequence
        length = len(seq)
        
        seq = ("_" * window_radius) + seq + ("_" * window_radius) #add spacer
        for resn in range(length):
            _in = list(seq[resn:resn+window_radius*2+1])
            _data.append(_in)
    return _data

def generate_label(df):
    label = []
    for _, item in df.iterrows():
        ss = item.label
        for resn, _label in enumerate(ss):
            label.append(int(_label))
    return np.array(label)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="example program")
    parser.add_argument("-train", help="path to training data (required)")
    parser.add_argument("-test", help="path to test data (optional)")
    parser.add_argument("-out", help="path to predicted information for test data (required only if --test is set)")
    parser.add_argument("--window_radius", type=int, default=1)
    args = parser.parse_args()

    check_args(args)

    ###### 1. data preparation ######
    
    # read train.csv files
    train_val_df = pd.read_csv(args.train)
    # split into train dataset and validation dataset (not train-test splitting)
    train_df, val_df = train_test_split(train_val_df, random_state=0)
    # delete for memory
    del train_val_df

    # setting window radius
    window_radius = args.window_radius

    # -----train data set-----
    train_data_ = generate_input(train_df, window_radius)
    y_train = generate_label(train_df)
    del train_df
    # set encoder
    transformer = OneHotEncoder().fit(train_data_)
    X_train_  = transformer.transform(train_data_)
    del train_data_
    # set feature selection for dimension reduction
    skb = SelectKBest(chi2, k=1591)
    skb.fit(X_train_, y_train)
    X_train = skb.transform(X_train_)
    del X_train_

    # -----validation data set-----
    val_data_   = generate_input(val_df, window_radius)
    y_val   = generate_label(val_df)
    del val_df
    X_val_    = transformer.transform(val_data_)
    del val_data_
    X_val = skb.transform(X_val_)
    del X_val_

    # -----test data set-----
    test_df      = pd.read_csv(args.test) if (args.test is not None) else None
    test_data_  = generate_input(test_df, window_radius) if (test_df is not None) else None
    # test_label = None
    X_test_   = transformer.transform(test_data_) if (test_data_ is not None) else None
    del test_data_
    X_test = skb.transform(X_test_)
    del X_test_

    ###### 2. model construction (w/ training dataset) ######    
    total_num = X_train.shape[0]
    D_in = X_train.shape[1]
    H1 = 512
    H2 = 64
    D_out = 2

    X_coo = sparse.coo_matrix(X_train)
    del X_train
    X_train = torch.sparse_coo_tensor([X_coo.row, X_coo.col], X_coo.data, (X_coo.shape[0], X_coo.shape[1])).float()
    del X_coo

    X_coo = sparse.coo_matrix(X_val)
    del X_val
    X_val = torch.sparse_coo_tensor([X_coo.row, X_coo.col], X_coo.data, (X_coo.shape[0], X_coo.shape[1])).float()
    del X_coo

    X_coo = sparse.coo_matrix(X_test)
    del X_test
    X_test = torch.sparse_coo_tensor([X_coo.row, X_coo.col], X_coo.data, (X_coo.shape[0], X_coo.shape[1])).float()
    del X_coo

    y_train_ = np.zeros((total_num, 2))
    for idx, item in enumerate(y_train):
        if item == 0:
            y_train_[idx, 0] = 1.
        else:
            y_train_[idx, 1] = 1.
    del y_train
    y_train = torch.tensor(y_train_, dtype=torch.float)
    del y_train_

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(D_in, H1)
            self.fc2 = nn.Linear(H1, H2)
            self.fc3 = nn.Linear(H2, D_out)

        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=0.8)
            x = F.gelu(self.fc2(x))
            x = F.dropout(x, p=0.2)
            x = F.gelu(self.fc3(x))
            x = F.softmax(x)
            return x

    num_epochs = 350

    model = Net()
    model.train()

    loss_fn = nn.BCELoss()

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_loss = []

    for epoch in tqdm(range(num_epochs)):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.numpy().tolist())

    ###### output loss #####
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list(range(len(epoch_loss))), epoch_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig('loss.png')

    ###### 3. model evaluation (w/ validation dataset) ######

    model.eval()
    y_pred = model(X_val)
    
    score = r2_score(y_val, y_pred[:, 1].detach().numpy())
    auc = roc_auc_score(y_val, y_pred[:, 1].detach().numpy())

    print('Q2 accuracy: %.4f'%(score))
    print('AUC: %.4f'%(auc))

    ###### 4. prediction for test dataset ######

    if (test_df is not None) and (X_test is not None):
        
        predicted = model(X_test)[:, 1].detach().numpy()

        sequence_id_list    = []
        residue_number_list = []
        for _, item in test_df.iterrows():
            sequence_id = item.sequence_id
            sequence    = item.sequence
            for i, aa in enumerate(sequence):
                sequence_id_list.append(sequence_id)
                residue_number_list.append(i+1) #0-origin to 1-origin

        predicted_df = pd.DataFrame.from_dict({
            "sequence_id": sequence_id_list,
            "residue_number": residue_number_list,
            "predicted_value": predicted,
            })
        predicted_df.to_csv(args.out, index=None)

            