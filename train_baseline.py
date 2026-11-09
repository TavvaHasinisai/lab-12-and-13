from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parent

def main():
    X=pd.read_csv(ROOT/'data/processed/X_train.csv'); y=pd.read_csv(ROOT/'data/processed/y_train.csv').squeeze().astype(int)
    nums=['release_year','duration_value']; cats=[c for c in X.columns if c not in nums]
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cats)])
    model=Pipeline([('preprocess',prep),('classifier',RandomForestClassifier(n_estimators=180,max_depth=18,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))])
    model.fit(X,y); joblib.dump(model,ROOT/'models/netflix_type_model.pkl')
    print('Baseline Netflix title-type model trained and saved.')
if __name__=='__main__': main()
