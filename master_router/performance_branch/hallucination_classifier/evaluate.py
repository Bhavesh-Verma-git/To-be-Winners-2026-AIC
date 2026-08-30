import os
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix

def evaluate_model(data_dir, model_dir, split='test'):
    model_path = os.path.join(model_dir, "xgb_hallucination_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        
    filepath = os.path.join(data_dir, f"{split}_features.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading {split} data...")
    df = pd.read_csv(filepath)
    
    features = [
        'overlap_score',
        'nli_entailment_score',
        'nli_contradiction_score',
        'nli_neutral_score',
        'sentence_level_max_contradiction',
        'entity_overlap_ratio',
        'new_entities_count',
        'response_length',
        'context_length',
        'length_ratio',
        'model_tier',
        'temperature'
    ]
    
    df = df.dropna(subset=features + ['label'])
    X = df[features]
    y_true = df['label'].astype(int)
    
    print("Loading model...")
    model = XGBClassifier()
    model.load_model(model_path)
    
    print("Predicting...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f" EVALUATION RESULTS ({split.upper()} SPLIT)")
    print("="*40)
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("="*40)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives (Clean classified as Clean): {cm[0][0]}")
    print(f"False Positives (Clean classified as Hallucinated): {cm[0][1]}")
    print(f"False Negatives (Hallucinated classified as Clean): {cm[1][0]}  <-- MOST DANGEROUS")
    print(f"True Positives (Hallucinated classified as Hallucinated): {cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Clean (0)", "Hallucinated (1)"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_dir, split=args.split)
