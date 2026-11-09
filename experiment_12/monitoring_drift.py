from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
def main():
    model=joblib.load(ROOT/'models/netflix_type_model.pkl')
    X=pd.read_csv(ROOT/'data/processed/X_test.csv'); y=pd.read_csv(ROOT/'data/processed/y_test.csv').squeeze().astype(int)
    pred=model.predict(X)
    metrics={'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1_score':float(f1_score(y,pred,zero_division=0))}
    incoming=X.copy(); rng=np.random.default_rng(42)
    incoming['release_year']=(incoming['release_year']+rng.integers(-5,6,len(incoming))).clip(1925,2026)
    incoming['country_feature']=incoming['country_feature'].replace({'United States':'India'})
    incoming['rating_feature']=incoming['rating_feature'].replace({'TV-MA':'TV-14'})
    drift={}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            stat,p=ks_2samp(X[c].dropna(),incoming[c].dropna()); test='Kolmogorov-Smirnov'
        else:
            a=X[c].astype(str).value_counts(normalize=True); b=incoming[c].astype(str).value_counts(normalize=True); cats=sorted(set(a.index)|set(b.index))
            obs=np.array([[a.get(k,0)*len(X),b.get(k,0)*len(incoming)] for k in cats]); stat,p,_,_=chi2_contingency(obs+1e-8); test='Chi-square'
        drift[c]={'test':test,'statistic':float(stat),'p_value':float(p),'status':'DRIFT' if p<.05 else 'STABLE'}
    newpred=model.predict(incoming)
    report={'experiment':12,'project':'Netflix Content Type Classification','target_column':'type','target_encoding':{'Movie':0,'TV Show':1},'baseline_performance':metrics,'prediction_distribution':{'baseline_tv_show_rate':float(np.mean(pred)),'incoming_tv_show_rate':float(np.mean(newpred)),'change':float(np.mean(newpred)-np.mean(pred))},'feature_drift':drift,'drift_detected':any(v['status']=='DRIFT' for v in drift.values()),'note':'Incoming batch is synthetically shifted for academic drift-monitoring demonstration; it is not a live Netflix feed.'}
    out=ROOT/'outputs/experiment12_monitoring_report.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print('EXPERIMENT 12 - MODEL MONITORING AND DRIFT DETECTION'); print('Baseline metrics:',metrics); print('Drifted features:',[k for k,v in drift.items() if v['status']=='DRIFT']); print('Report saved:',out)
if __name__=='__main__':main()
