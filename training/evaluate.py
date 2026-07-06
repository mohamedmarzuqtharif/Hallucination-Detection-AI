"""Evaluate a serialized classical classifier."""
import argparse
import joblib
import pandas as pd
from metrics import classification_metrics
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("csv"); args=parser.parse_args()
    artifact=joblib.load(args.model); data=pd.read_csv(args.csv); x=data[artifact["features"]]
    predictions=artifact["model"].predict(x); scores=artifact["model"].predict_proba(x)[:,1]
    print(classification_metrics(data["label"],predictions,scores))
if __name__=="__main__": main()
