from pathlib import Path
import subprocess,sys,json,joblib,pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
def main():
 print('='*64+'\nEXPERIMENT 13 - END-TO-END NETFLIX MLOPS PIPELINE\n'+'='*64)
 stages=[]
 prep=subprocess.run([sys.executable,str(ROOT/'prepare_data.py')],capture_output=True,text=True); print(prep.stdout); prep.check_returncode(); stages.append({'stage':'Dataset loading and preprocessing','status':'SUCCESS'})
 train=subprocess.run([sys.executable,str(ROOT/'train_baseline.py')],capture_output=True,text=True); print(train.stdout); train.check_returncode(); stages.append({'stage':'Baseline model training','status':'SUCCESS'})
 exp12=subprocess.run([sys.executable,str(ROOT/'experiment_12/run_experiment12.py')],capture_output=True,text=True); print(exp12.stdout); exp12.check_returncode(); stages.append({'stage':'Monitoring, drift detection and retraining','status':'SUCCESS'})
 model=joblib.load(ROOT/'models/netflix_type_model.pkl'); X=pd.read_csv(ROOT/'data/processed/X_test.csv'); y=pd.read_csv(ROOT/'data/processed/y_test.csv').squeeze().astype(int); p=model.predict(X)
 metrics={'accuracy':float(accuracy_score(y,p)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'f1_score':float(f1_score(y,p,zero_division=0))}; stages.append({'stage':'Final model evaluation','status':'SUCCESS'})
 report={'experiment':13,'project':'Netflix Content Type Classification','target_column':'type','target_encoding':{'Movie':0,'TV Show':1},'workflow':['Load Netflix CSV dataset','Clean and engineer metadata features','Create stratified train/test split','Train baseline Random Forest pipeline','Monitor performance and feature drift','Train and compare candidate model','Promote candidate only if F1-score does not decrease','Evaluate final production model'],'stages':stages,'final_model_metrics':metrics,'overall_status':'SUCCESS'}
 out=ROOT/'outputs/experiment13_end_to_end_report.json'; out.write_text(json.dumps(report,indent=2)); print('Final model metrics:',metrics); print('End-to-end report:',out); print('EXPERIMENT 13 COMPLETED SUCCESSFULLY')
if __name__=='__main__':main()
