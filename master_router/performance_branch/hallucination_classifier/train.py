import os
import argparse
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

class XGBTrainer:
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.features = [
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
        
    def load_data(self, split='train'):
        filepath = os.path.join(self.data_dir, f"{split}_features.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        # Drop rows with NaN labels or features just in case
        df = df.dropna(subset=self.features + ['label'])
        
        X = df[self.features]
        y = df['label'].astype(int)
        return X, y, df

    def get_scale_pos_weight(self, y):
        # scale_pos_weight = count(negative examples) / count(positive examples)
        # 0 = clean (negative), 1 = hallucinated (positive)
        neg_count = sum(y == 0)
        pos_count = sum(y == 1)
        return neg_count / pos_count if pos_count > 0 else 1.0

    def run_optuna_tuning(self, X, y, n_trials=50):
        print(f"Running Optuna tuning with {n_trials} trials...")
        # Split train into train/val for tuning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        scale_pos_weight = self.get_scale_pos_weight(y_train)
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "scale_pos_weight": scale_pos_weight,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42
            }
            
            model = XGBClassifier(**params)
            
            # Using early stopping callback
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val)
            return f1_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        # Suppress optuna logging clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("\nBest Trial:")
        print("  F1 Value: ", study.best_value)
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
            
        return study.best_params

    def train_final_model(self, X, y, best_params):
        print("\nTraining final model on full dataset with best parameters...")
        
        params = best_params.copy()
        params['scale_pos_weight'] = self.get_scale_pos_weight(y)
        params['use_label_encoder'] = False
        params['eval_metric'] = "logloss"
        params['random_state'] = 42
        
        model = XGBClassifier(**params)
        model.fit(X, y, verbose=True)
        
        model_path = os.path.join(self.model_dir, "xgb_hallucination_model.json")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--trials", type=int, default=50, help="Number of optuna trials")
    parser.add_argument("--combine_all", action="store_true", help="Combine train and test sets for final training")
    args = parser.parse_args()
    
    trainer = XGBTrainer(args.data_dir, args.model_dir)
    
    print("Loading training data...")
    X_train, y_train, _ = trainer.load_data('train')
    print(f"Train data size: {len(X_train)} rows")
    
    # Run Tuning
    best_params = trainer.run_optuna_tuning(X_train, y_train, n_trials=args.trials)
    
    # Train Final
    if args.combine_all:
        print("Loading test data to combine for final training...")
        try:
            X_test, y_test, _ = trainer.load_data('test')
            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            trainer.train_final_model(X_full, y_full, best_params)
        except FileNotFoundError:
            print("Test data not found. Falling back to train data only.")
            trainer.train_final_model(X_train, y_train, best_params)
    else:
        trainer.train_final_model(X_train, y_train, best_params)
