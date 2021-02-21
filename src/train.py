import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

# executé depuis le repertoire parent par python -m src.train
"""
TODO Done : Trouver l equivalent windows de 
export TRAINING_DATA=input/train_folds.csv
export FOLD=0
-
python -m src.train.

Ces commandes sont stockées dans un fichier *.sh
"""
TRAINING_DATA = os.getenv("TRAINING_DATA")
TEST_DATA = os.getenv("TEST_DATA")
FOLD = int(os.getenv("FOLD"))
MODEL =os.getenv("MODEL")
#TRAINING_DATA="input/train_folds.csv"
#FOLD=0

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__== "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
   
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id","target","kfold"],axis=1)
    valid_df = valid_df.drop(["id","target","kfold"],axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist() + 
                df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train
    # clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2)
    clf= dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    '''
    TODO: voir pourquoi il dit "When data is skewed use ROC_AUC metrics ?"
    '''
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders,f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns,f"models/{MODEL}_columns.pkl")






