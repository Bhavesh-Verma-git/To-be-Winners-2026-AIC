import os
from xgboost import XGBClassifier
import json
import logging
from .hallucination_classifier.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class XGBoostHallucinationAgent:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Default to the model dir inside hallucination_classifier
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "hallucination_classifier", "model")
            
        self.model_path = os.path.join(model_dir, "xgb_hallucination_model.json")
        self.model = None
        self.feature_engineer = None

    def _load_model_lazy(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"XGBoost model not found at {self.model_path}. "
                    "Please run the feature engineering and training pipeline first."
                )
            logger.info("Loading XGBoost Hallucination Model...")
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)
            
        if self.feature_engineer is None:
            logger.info("Loading Feature Engineering models (Embedder, NLI, NER)...")
            # Try loading on GPU if available, else CPU (default handled in init)
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.feature_engineer = FeatureEngineer(device=device)

    def score(self, context: str, response: str, model_name: str, temperature: float) -> dict:
        """
        Takes the raw generation context and response, and scores it for hallucinations.
        Returns a dict with hallucination_probability, label, and risk_level.
        """
        self._load_model_lazy()
        
        # Extract features
        features_dict = self.feature_engineer.compute_features_for_row(
            context=context,
            output=response,
            model_name=model_name,
            temperature=temperature
        )
        
        # The model expects features in the exact order it was trained on
        feature_order = [
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
        
        import pandas as pd
        # Create a single-row DataFrame to preserve feature names for XGBoost
        # if the model was trained with feature names enabled
        X = pd.DataFrame([[features_dict[f] for f in feature_order]], columns=feature_order)
        
        prob = self.model.predict_proba(X)[0][1] # Probability of Class 1 (Hallucination)
        label = int(self.model.predict(X)[0])
        
        risk_level = "LOW"
        if prob > 0.8:
            risk_level = "CRITICAL"
        elif prob > 0.5:
            risk_level = "HIGH"
        elif prob > 0.2:
            risk_level = "MODERATE"
            
        return {
            "is_hallucination": bool(label == 1),
            "hallucination_probability": float(prob),
            "risk_level": risk_level,
            "feature_breakdown": features_dict # Useful for explaining *why* it flagged
        }

# For master_router wrapper testing
if __name__ == "__main__":
    agent = XGBoostHallucinationAgent()
    try:
        # Dummy test
        res = agent.score(
            context="The capital of France is Paris.",
            response="Paris is the capital of France, and London is in the UK.",
            model_name="gpt-4",
            temperature=0.7
        )
        print(json.dumps(res, indent=2))
    except FileNotFoundError as e:
        print(e)
