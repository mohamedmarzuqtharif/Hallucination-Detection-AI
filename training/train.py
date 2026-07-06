"""Train and serialize a classical hallucination classifier from CSV features."""
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("csv"); parser.add_argument("--output",default="models/classifier.joblib")
    parser.add_argument("--algorithm",choices=["random_forest","xgboost"],default="random_forest")
    args=parser.parse_args(); data=pd.read_csv(args.csv)
    features=data.drop(columns=["label"])
    if args.algorithm == "xgboost":
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,eval_metric="logloss",random_state=42)
    else:
        model=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)
    model.fit(features,data["label"]); joblib.dump({"model":model,"features":list(features.columns)},args.output)
if __name__=="__main__": main()
