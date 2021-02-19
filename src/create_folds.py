import pandas as pd
from sklearn import model_selection
# A ete execute depuis le repertoire parent par python -m src.create_folds
if __name__=="__main__": 
    '''
    Dans un module, mettre cette condition permet de donner la stratégie si le script est appelé
    dans un module ou bien de manière autonome
    '''
    df=pd.read_csv("input/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True) # mélanger les données

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False,random_state=42)

    for fold,(train_idx,val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(val_idx))
        df.loc[val_idx,'kfold']=fold
    
    df.to_csv("input/train_folds.csv",index=False)

