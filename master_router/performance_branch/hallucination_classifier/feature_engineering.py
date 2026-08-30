import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Ensure punkt is downloaded for sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class FeatureEngineer:
    def __init__(self, device='cpu'):
        print("Loading models...")
        self.device = device
        # 1. Cosine Similarity model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 2. NLI model
        print("Loading NLI model (this might take a moment)...")
        # roberta-large-mnli predicts: [contradiction, neutral, entailment]
        self.nli_model = pipeline("text-classification", model="roberta-large-mnli", device=0 if self.device=='cuda' else -1, top_k=None)
        
        # 3. NER model (spaCy)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            print("Downloading spacy en_core_web_sm model...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        print("Models loaded successfully.")

    def _get_model_tier(self, model_name):
        model_name = str(model_name).lower()
        if 'gpt-4' in model_name:
            return 3
        elif 'gpt-3.5' in model_name or '70b' in model_name:
            return 2
        else: # 7b, 13b, mistral, etc
            return 1

    def _derive_label(self, raw_label_str):
        if pd.isna(raw_label_str) or raw_label_str == "" or raw_label_str == "[]" or raw_label_str == "None":
            return 0
        try:
            parsed = json.loads(raw_label_str)
            return 1 if len(parsed) > 0 else 0
        except Exception:
            return 0

    def _extract_entities(self, text):
        if not isinstance(text, str):
            return set()
        # To avoid spacy length limits on huge context, truncate to first 100k chars
        doc = self.nlp(text[:100000])
        return set(ent.text.lower() for ent in doc.ents)

    def compute_features_for_row(self, context, output, model_name, temperature):
        context = str(context)
        output = str(output)
        
        # Group D: Metadata
        tier = self._get_model_tier(model_name)
        temp = float(temperature) if pd.notnull(temperature) else 0.5
        
        # Group C: Length
        ctx_len = len(context.split())
        out_len = len(output.split())
        len_ratio = out_len / (ctx_len + 1e-5)
        
        # Group B: Entity Overlap
        ctx_ents = self._extract_entities(context)
        out_ents = self._extract_entities(output)
        
        new_ents = out_ents - ctx_ents
        new_ents_count = len(new_ents)
        ent_overlap_ratio = len(out_ents.intersection(ctx_ents)) / (len(out_ents) + 1e-5)
        if len(out_ents) == 0:
            ent_overlap_ratio = 1.0 # If no entities, it's not a hallucination of entities
            
        # Group A: Semantic Overlap (Embeddings)
        ctx_emb = self.embedder.encode(context, convert_to_tensor=True, show_progress_bar=False)
        out_emb = self.embedder.encode(output, convert_to_tensor=True, show_progress_bar=False)
        overlap_score = util.cos_sim(ctx_emb, out_emb).item()
        
        # Group A: NLI (Whole text)
        # NLI typically takes "Premise" (context) and "Hypothesis" (output)
        # Truncate to avoid RoBERTa 512 limit blowing up on pipeline
        # 1 token roughly 4 chars. We take first 1500 chars of context and output.
        nli_input = f"{context[:1500]} </s></s> {output[:1500]}"
        try:
            nli_res = self.nli_model(nli_input, truncation=True, max_length=512)[0]
            # pipeline top_k=None returns list of dicts: [{'label': 'CONTRADICTION', 'score': 0.1}, ...]
            nli_scores = {res['label']: res['score'] for res in nli_res}
            entailment_score = nli_scores.get('ENTAILMENT', 0.0)
            contradiction_score = nli_scores.get('CONTRADICTION', 0.0)
            neutral_score = nli_scores.get('NEUTRAL', 0.0)
        except Exception as e:
            # Fallback
            entailment_score = 0.5
            contradiction_score = 0.5
            neutral_score = 0.5

        # Group A: Sentence-level Max Contradiction
        sentences = sent_tokenize(output)
        max_contradiction = 0.0
        
        for sent in sentences:
            if not sent.strip(): continue
            sent_input = f"{context[:1500]} </s></s> {sent}"
            try:
                s_res = self.nli_model(sent_input, truncation=True, max_length=512)[0]
                s_scores = {res['label']: res['score'] for res in s_res}
                max_contradiction = max(max_contradiction, s_scores.get('CONTRADICTION', 0.0))
            except Exception:
                continue
                
        return {
            'overlap_score': overlap_score,
            'nli_entailment_score': entailment_score,
            'nli_contradiction_score': contradiction_score,
            'nli_neutral_score': neutral_score,
            'sentence_level_max_contradiction': max_contradiction,
            'entity_overlap_ratio': ent_overlap_ratio,
            'new_entities_count': new_ents_count,
            'response_length': out_len,
            'context_length': ctx_len,
            'length_ratio': len_ratio,
            'model_tier': tier,
            'temperature': temp
        }

    def process_dataset(self, split='train', max_rows=None, output_path=None):
        print(f"Loading {split} split from wandb/RAGTruth-processed...")
        ds = load_dataset('wandb/RAGTruth-processed', split=split)
        
        # Convert to pandas
        df = ds.to_pandas()
        
        # Filter for quality == "good"
        initial_len = len(df)
        df = df[df['quality'] == 'good'].copy()
        print(f"Filtered quality: {initial_len} -> {len(df)} rows")
        
        if max_rows and max_rows < len(df):
            df = df.sample(n=max_rows, random_state=42)
            print(f"Sampled down to {max_rows} rows for quick testing.")
            
        print("Extracting labels and features...")
        features_list = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            # Label
            label = self._derive_label(row['hallucination_labels'])
            labels.append(label)
            
            # Features
            feats = self.compute_features_for_row(
                context=row['context'],
                output=row['output'],
                model_name=row['model'],
                temperature=row['temperature']
            )
            features_list.append(feats)
            
        # Combine into final dataframe
        feature_df = pd.DataFrame(features_list)
        feature_df['label'] = labels
        
        if output_path:
            # Ensure dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            feature_df.to_csv(output_path, index=False)
            print(f"Saved processed features to {output_path}")
            
        return feature_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows for quick testing")
    parser.add_argument("--out_dir", type=str, default="processed_data")
    args = parser.parse_args()
    
    # Optional: use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    engineer = FeatureEngineer(device=device)
    
    if args.split == "all":
        splits = ["train", "test"]
    else:
        splits = [args.split]
        
    for sp in splits:
        out_file = os.path.join(args.out_dir, f"{sp}_features.csv")
        engineer.process_dataset(split=sp, max_rows=args.max_rows, output_path=out_file)
        print(f"Finished processing {sp} split.")
