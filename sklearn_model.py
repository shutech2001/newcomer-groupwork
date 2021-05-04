import argparse
import numpy as np
import pandas as pd
import tqdm
from sklearn import *
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

import lightgbm as lgb
import optuna

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
        
        seq = ('_' * window_radius) + seq + ('_' * window_radius) #add spacer
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
    
    model = lgb.LGBMClassifier().fit(X_train, y_train)

    ###### 3. model evaluation (w/ validation dataset) ######
    
    score = model.score(X_val, y_val)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print('Q2 accuracy: %.4f'%(score))
    print('AUC: %.4f'%(auc))

    ###### 4. prediction for test dataset ######

    if (test_df is not None) and (X_test is not None):
        
        predicted = model.predict_proba(X_test)[:, 1]

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
            
