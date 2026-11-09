from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent

def main():
    df=pd.read_csv(ROOT/'data/raw/netflix.csv')
    # Keep useful, interpretable metadata and derive compact features.
    df=df.dropna(subset=['type']).copy()
    df['country_feature']=df['country'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
    df['rating_feature']=df['rating'].fillna('Unknown').astype(str).str.strip()
    df['genre_feature']=df['listed_in'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
    df['director_feature']=df['director'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
    duration=df['duration'].fillna('').astype(str).str.extract(r'(\d+)\s*(min|Season|Seasons)', expand=True)
    df['duration_value']=pd.to_numeric(duration[0],errors='coerce').fillna(0)
    df['duration_unit']=duration[1].fillna('Unknown').replace({'Season':'Seasons'})
    X=df[['release_year','duration_value','country_feature','rating_feature','genre_feature','director_feature','duration_unit']].copy()
    X['release_year']=pd.to_numeric(X['release_year'],errors='coerce').fillna(X['release_year'].median())
    y=df['type'].map({'Movie':0,'TV Show':1}).astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    for name,obj in [('X_train',Xtr),('X_test',Xte),('y_train',ytr),('y_test',yte)]: obj.to_csv(ROOT/f'data/processed/{name}.csv',index=False)
    meta={'dataset':'Netflix titles','rows':int(len(df)),'features':list(X.columns),'target_column':'type','target_encoding':{'Movie':0,'TV Show':1},'train_rows':len(Xtr),'test_rows':len(Xte)}
    (ROOT/'data/processed/dataset_metadata.json').write_text(json.dumps(meta,indent=2))
    print('NETFLIX DATA PREPARATION')
    print(f'Rows: {len(df)} | Features: {len(X.columns)} | Train: {len(Xtr)} | Test: {len(Xte)}')
    print('Target distribution:',df['type'].value_counts().to_dict())
    print('Saved processed train/test files.')
if __name__=='__main__': main()
