from pathlib import Path
import json,joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
def evaluate(m,X,y):
 p=m.predict(X); return {'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1_score':float(f1_score(y,p,zero_division=0))}
def main():
 Xtr=pd.read_csv(ROOT/'data/processed/X_train.csv'); Xte=pd.read_csv(ROOT/'data/processed/X_test.csv'); ytr=pd.read_csv(ROOT/'data/processed/y_train.csv').squeeze().astype(int); yte=pd.read_csv(ROOT/'data/processed/y_test.csv').squeeze().astype(int)
 base=joblib.load(ROOT/'models/netflix_type_model.pkl'); nums=['release_year','duration_value']; cats=[c for c in Xtr.columns if c not in nums]
 prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cats)])
 candidate=Pipeline([('preprocess',prep),('classifier',RandomForestClassifier(n_estimators=260,max_depth=22,min_samples_leaf=1,class_weight='balanced_subsample',random_state=24,n_jobs=-1))]); candidate.fit(Xtr,ytr)
 before=evaluate(base,Xte,yte); after=evaluate(candidate,Xte,yte); promoted=after['f1_score']>=before['f1_score']
 joblib.dump(candidate,ROOT/'models/netflix_type_candidate.pkl')
 if promoted: joblib.dump(candidate,ROOT/'models/netflix_type_model.pkl')
 report={'experiment':12,'project':'Netflix Content Type Classification','baseline_model_metrics':before,'candidate_model_metrics':after,'promotion_decision':'PROMOTED' if promoted else 'NOT PROMOTED','promotion_rule':'Candidate F1-score must be equal to or higher than baseline F1-score','production_model_updated':bool(promoted)}
 out=ROOT/'outputs/experiment12_retraining_report.json'; out.write_text(json.dumps(report,indent=2))
 print('EXPERIMENT 12 - MODEL RETRAINING'); print('Baseline:',before); print('Candidate:',after); print('Decision:',report['promotion_decision']); print('Report saved:',out)
if __name__=='__main__':main()
