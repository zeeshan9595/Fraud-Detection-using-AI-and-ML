"""Leakage-aware IEEE-CIS/PaySim fraud experiment runner."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def load_data(csv_path, dataset, identity_csv=None):
    if dataset not in {"ieee","paysim"}: raise ValueError("dataset must be ieee or paysim")
    frame=pd.read_csv(csv_path,low_memory=False)
    if dataset=="ieee" and identity_csv:
        identity=pd.read_csv(identity_csv,low_memory=False)
        if "TransactionID" not in frame or "TransactionID" not in identity: raise ValueError("Both IEEE files need TransactionID")
        if identity.TransactionID.duplicated().any(): raise ValueError("Identity has duplicate TransactionID values")
        n=len(frame)
        frame=frame.merge(identity,on="TransactionID",how="left",validate="many_to_one",suffixes=("","_identity"))
        if len(frame)!=n: raise RuntimeError("Identity merge changed transaction count")
    return frame

def chronological_split(frame,target,time_col,train_fraction=.60,val_fraction=.20):
    if target not in frame or time_col not in frame: raise ValueError("Required target/time column missing")
    if not (0<train_fraction<1 and 0<val_fraction<1 and train_fraction+val_fraction<1): raise ValueError("Invalid split fractions")
    data=frame.sort_values(time_col,kind="mergesort").reset_index(drop=True)
    a=int(len(data)*train_fraction); b=int(len(data)*(train_fraction+val_fraction))
    if a<1 or b<=a or b>=len(data): raise ValueError("Not enough rows for three-way split")
    parts=(data.iloc[:a].copy(),data.iloc[a:b].copy(),data.iloc[b:].copy())
    for name,p in zip(("train","validation","test"),parts):
        if p[target].nunique(dropna=False)<2: raise ValueError(f"{name} partition lacks both classes; review temporal boundaries")
    return parts

def features(frame,dataset,target,time_col,include_post_event=False):
    excluded={target,time_col,"TransactionID"}
    if dataset=="paysim":
        excluded.update({"nameOrig","nameDest","isFlaggedFraud"})
        if not include_post_event: excluded.update({"newbalanceOrig","newbalanceDest"})
    return frame.drop(columns=[c for c in excluded if c in frame.columns])

def make_model(X,y,seed=42,n_estimators=400,max_depth=6,learning_rate=.05):
    numeric=X.select_dtypes(include=[np.number,"bool"]).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric: transformers.append(("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True))]),numeric))
    if categorical: transformers.append(("cat",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="__MISSING__")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=20,sparse_output=True,dtype=np.float32))]),categorical))
    if not transformers: raise ValueError("No feature columns")
    prep=ColumnTransformer(transformers,sparse_threshold=1.0,verbose_feature_names_out=False)
    pos=int((y==1).sum()); neg=int((y==0).sum())
    if not pos or not neg: raise ValueError("Training labels must contain both classes")
    model=XGBClassifier(objective="binary:logistic",eval_metric="aucpr",n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=.8,colsample_bytree=.8,min_child_weight=1,reg_lambda=1.0,tree_method="hist",random_state=seed,n_jobs=-1,scale_pos_weight=neg/pos)
    pipe=Pipeline([("preprocess",prep),("model",model)])
    pipe.fit(X,y)
    return pipe

def choose_f1(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    if not len(thresholds): return {"threshold":.5,"validation_constraint_met":False}
    scores=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    i=int(np.nanargmax(scores))
    return {"threshold":float(thresholds[i]),"validation_precision":float(precision[i]),"validation_recall":float(recall[i]),"validation_f1":float(scores[i]),"validation_constraint_met":True}

def choose_precision(y,p,min_precision):
    if not 0<min_precision<=1: raise ValueError("min-precision must be in (0,1]")
    precision,recall,thresholds=precision_recall_curve(y,p)
    valid=np.flatnonzero(precision[:-1]>=min_precision)
    if not len(valid):
        i=int(np.argmax(precision[:-1]))
        return {"threshold":float(thresholds[i]),"validation_precision":float(precision[i]),"validation_recall":float(recall[i]),"validation_constraint_met":False,"requested_min_precision":min_precision}
    max_recall=np.max(recall[valid]); candidates=valid[recall[valid]==max_recall]
    i=int(candidates[np.argmax(precision[candidates])])
    return {"threshold":float(thresholds[i]),"validation_precision":float(precision[i]),"validation_recall":float(recall[i]),"validation_constraint_met":True,"requested_min_precision":min_precision}

def threshold_metrics(y,p,threshold):
    pred=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(threshold),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"true_positives":int(tp),"false_positives":int(fp),"true_negatives":int(tn),"false_negatives":int(fn),"flagged_count":int(tp+fp)}

def cost_sensitivity(metrics,fp_cost,fn_cost):
    if fp_cost is None and fn_cost is None: return None
    if fp_cost is None or fn_cost is None: raise ValueError("Supply both cost assumptions or neither")
    if fp_cost<0 or fn_cost<0: raise ValueError("Costs must be non-negative")
    fp=metrics["false_positives"]*fp_cost; fn=metrics["false_negatives"]*fn_cost
    return {"fp_unit_cost_assumption":fp_cost,"fn_unit_cost_assumption":fn_cost,"fp_cost":fp,"fn_cost":fn,"total_illustrative_cost":fp+fn,"interpretation":"Scenario analysis only; not measured operational cost, ROI, or realised savings."}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",choices=["ieee","paysim"],required=True); ap.add_argument("--csv",required=True); ap.add_argument("--identity-csv")
    ap.add_argument("--output-dir",default="outputs/run"); ap.add_argument("--min-precision",type=float,default=.70)
    ap.add_argument("--train-fraction",type=float,default=.60); ap.add_argument("--validation-fraction",type=float,default=.20)
    ap.add_argument("--random-state",type=int,default=42); ap.add_argument("--n-estimators",type=int,default=400); ap.add_argument("--max-depth",type=int,default=6); ap.add_argument("--learning-rate",type=float,default=.05)
    ap.add_argument("--fp-cost",type=float); ap.add_argument("--fn-cost",type=float); ap.add_argument("--include-post-event",action="store_true",help="Diagnostic only; not suitable for pre-decision scoring")
    args=ap.parse_args()
    target,time_col=("isFraud","TransactionDT") if args.dataset=="ieee" else ("isFraud","step")
    data=load_data(args.csv,args.dataset,args.identity_csv)
    if data[target].isna().any() or not set(data[target].unique()).issubset({0,1}): raise ValueError("Target must be non-missing binary 0/1")
    train,val,test=chronological_split(data,target,time_col,args.train_fraction,args.validation_fraction)
    Xtr=features(train,args.dataset,target,time_col,args.include_post_event); Xv=features(val,args.dataset,target,time_col,args.include_post_event); Xt=features(test,args.dataset,target,time_col,args.include_post_event)
    ytr=train[target].astype(int); yv=val[target].astype(int); yt=test[target].astype(int)
    Xv=Xv.reindex(columns=Xtr.columns); Xt=Xt.reindex(columns=Xtr.columns)
    model=make_model(Xtr,ytr,args.random_state,args.n_estimators,args.max_depth,args.learning_rate)
    pv=model.predict_proba(Xv)[:,1]; pt=model.predict_proba(Xt)[:,1]
    f1choice=choose_f1(yv,pv); pchoice=choose_precision(yv,pv,args.min_precision)
    if yt.nunique()!=2: raise ValueError("Test partition needs both classes for ranking metrics")
    results={"ranking":{"average_precision":float(average_precision_score(yt,pt)),"roc_auc":float(roc_auc_score(yt,pt)),"positive_prevalence":float(yt.mean()),"n_observations":int(len(yt)),"n_positive":int(yt.sum())}}
    for name,choice in (("f1_selected_threshold",f1choice),("precision_constrained_threshold",pchoice)):
        results[name]=threshold_metrics(yt,pt,choice["threshold"])
        results[name]["illustrative_cost"]=cost_sensitivity(results[name],args.fp_cost,args.fn_cost)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    report={"dataset":args.dataset,"status":"generated_by_run","target":target,"time_column_for_ordering_only":time_col,
      "temporal_split":{"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"train_time_min":str(train[time_col].min()),"train_time_max":str(train[time_col].max()),"validation_time_min":str(val[time_col].min()),"validation_time_max":str(val[time_col].max()),"test_time_min":str(test[time_col].min()),"test_time_max":str(test[time_col].max())},
      "feature_count_before_encoding":len(Xtr.columns),"feature_names":list(Xtr.columns),"threshold_selection_validation_only":{"f1_selected":f1choice,"precision_constrained":pchoice,"requested_min_precision":args.min_precision},"test_results":results,
      "caveats":["Thresholds selected on validation only, not test.","Temporal ordering does not guarantee production generalisation.","Cost inputs are assumptions, not measured ROI.","PaySim is synthetic; IEEE-CIS is anonymised."]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame({"y_true":yt.to_numpy(),"probability":pt,"prediction_f1_threshold":(pt>=f1choice["threshold"]).astype(int),"prediction_precision_threshold":(pt>=pchoice["threshold"]).astype(int)}).to_csv(output/"test_predictions.csv",index=False)
    joblib.dump(model,output/"model.joblib")
    print(json.dumps({"metrics":str(output/"metrics.json"),"predictions":str(output/"test_predictions.csv"),"model":str(output/"model.joblib"),"test_results":results},indent=2))

if __name__=="__main__": main()
