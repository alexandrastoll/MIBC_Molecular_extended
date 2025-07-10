

def strat_k_fold(annotations_file, n_splits=5, random_state=42, shuffle=True): #sampling_method=r_os):
    """Applies stratified kfold based on/integrated from scikit-learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    
    if isinstance(annotations_file, pd.DataFrame):
        
        annotations_file
        
    else:
        annotations_file = pd.read_excel(annotations_file) 
    
    df_kfold = annotations_file.copy()
    X = np.array(annotations_file.iloc[:,0]).reshape(-1,1)
    y = np.array(annotations_file.iloc[:,1]).reshape(-1,1)
    
    skf = StratifiedKFold(n_splits=n_splits,
    shuffle=shuffle, random_state=random_state)

    i = 0
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        df_kfold.loc[train_index, f'split{i}'] = 'False'
        df_kfold.loc[test_index, f'split{i}'] = 'True'
        i += 1
        
    return df_kfold


def sampler_strat_kfold(df_strat_kfold, rs='ros', random_state=None, random_state_valid=42):
    """For validation sets, random-oversampling is always used to make random-oversampling and
    random-undersampling more comparable. "rs": specifies sampler for training."""
    
    import numpy as np
    import pandas as pd
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler 
    
    n_splits = (len(df_strat_kfold.columns)-2)

    dfs_compl = []

    
    rs = rs
    
    if rs == 'ros':
        rs = RandomOverSampler(random_state=random_state)
        
    elif rs == 'rus':
        rs = RandomUnderSampler(random_state=random_state)
        
    rs_valid = RandomOverSampler(random_state=random_state_valid)

    for i in range(n_splits):

        df = df_strat_kfold.iloc[:,[0,1,(i+2)]]

        df_train = df[df[df.columns[2]] == 'False'].reset_index(drop=True)
        df_valid = df[df[df.columns[2]] == 'True'].reset_index(drop=True)

        X = np.array(df_train[df_train.columns[0]]).reshape(-1,1)
        y = np.array(df_train[df_train.columns[1]]).reshape(-1,1)

        X_res_train, y_res_train = rs.fit_resample(X, y)

        df_train = df_train[:0]

        df_train[df_train.columns[0]] = [i.item() for i in X_res_train]
        df_train[df_train.columns[1]] = y_res_train.tolist()
        df_train[df_train.columns[2]] = False

        X = np.array(df_valid[df_valid.columns[0]]).reshape(-1,1)
        y = np.array(df_valid[df_valid.columns[1]]).reshape(-1,1)

        X_res_valid, y_res_valid = rs_valid.fit_resample(X, y)

        df_valid = df_valid[:0]

        df_valid[df_valid.columns[0]] = [i.item() for i in X_res_valid]
        df_valid[df_valid.columns[1]] = y_res_valid.tolist()
        df_valid[df_valid.columns[2]] = True

        #df_compl = df_train.append(df_valid).reset_index(drop=True)
        df_compl = pd.concat([df_train, df_valid]).reset_index(drop=True)
        dfs_compl.append(df_compl)
    
    return dfs_compl