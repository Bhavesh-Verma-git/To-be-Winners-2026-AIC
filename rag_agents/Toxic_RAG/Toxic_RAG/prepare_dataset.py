#!/usr/bin/env python3
"""
prepare_dataset.py
Extracts and cleans toxic RAG dataset from annotated_train.csv and annotated_test.csv.
Saves the cleaned final dataset to Dataset/final_tox_Rag.csv (and dataset/final_tox_Rag.csv).
"""

import os
import ast
import re
import pandas as pd
from pathlib import Path

def clean_raw_text(val: str) -> str:
    """
    Cleans raw text strings from the dataset:
    - Removes b'...' and b"..." byte literal encodings
    - Unescapes escaped quotes and special characters
    - Strips whitespace and normalizes text
    """
    if not isinstance(val, str):
        return ""
    val = val.strip()
    
    # Check if text is encoded as a byte literal string representation
    if (val.startswith("b'") and val.endswith("'")) or (val.startswith('b"') and val.endswith('"')):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, bytes):
                return parsed.decode("utf-8", errors="ignore").strip()
            elif isinstance(parsed, str):
                return parsed.strip()
        except Exception:
            pass
        # Fallback manual strip
        val = val[2:-1]
        val = val.replace(r"\'", "'").replace(r'\"', '"').replace(r"\\", "\\")
    elif val.startswith("b'") or val.startswith('b"'):
        val = val[2:]
        val = val.replace(r"\'", "'").replace(r'\"', '"').replace(r"\\", "\\")
        if val.endswith("'") or val.endswith('"'):
            val = val[:-1]
            
    # Clean excessive whitespace
    val = re.sub(r"\s+", " ", val).strip()
    return val

def prepare_dataset(
    train_path: str = "Dataset/annotated_train.csv",
    test_path: str = "Dataset/annotated_test.csv",
    output_dir: str = "Dataset",
    output_filename: str = "final_tox_Rag.csv"
):
    print("=" * 60)
    print("Starting Toxic RAG Dataset Preparation")
    print("=" * 60)

    # Read datasets
    print(f"Reading train dataset: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"Train rows: {len(df_train)}")

    print(f"Reading test dataset: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"Test rows: {len(df_test)}")

    # Combine datasets
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Combined total rows: {len(df_combined)}")

    # Column mapping specified by requirement:
    # Text, target group, factual, in-group effect, framing, lewd, predicted group, stereotyping
    column_mapping = {
        "text": "Text",
        "target_group": "target group",
        "factual?": "factual",
        "ingroup_effect": "in-group effect",
        "framing": "framing",
        "lewd": "lewd",
        "predicted_group": "predicted group",
        "stereotyping": "stereotyping"
    }

    # Ensure all required source columns are present
    for src_col in column_mapping.keys():
        if src_col not in df_combined.columns:
            raise KeyError(f"Source column '{src_col}' missing from datasets!")

    # Select and rename columns
    df_final = df_combined[list(column_mapping.keys())].copy()
    df_final.rename(columns=column_mapping, inplace=True)

    # Clean the Text column
    print("Cleaning text column (removing b'' artifacts, unescaping quotes, normalizing whitespace)...")
    df_final["Text"] = df_final["Text"].apply(clean_raw_text)

    # Clean metadata columns: fill NaNs with empty string or sensible default
    for col in df_final.columns:
        if col != "Text":
            df_final[col] = df_final[col].fillna("").astype(str).str.strip()

    # Filter out any completely empty text rows if any
    initial_count = len(df_final)
    df_final = df_final[df_final["Text"] != ""].reset_index(drop=True)
    if len(df_final) < initial_count:
        print(f"Filtered out {initial_count - len(df_final)} empty text rows.")

    # Save output
    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Successfully saved cleaned dataset to: {output_path}")
    print(f"Final shape: {df_final.shape}")
    print(f"Columns: {df_final.columns.tolist()}")

    # Also make sure lowercase 'dataset' path exists if needed
    alt_output_path = Path("dataset") / output_filename
    if str(alt_output_path) != str(output_path) and not alt_output_path.exists():
        try:
            alt_output_path.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(alt_output_path, index=False)
            print(f"Also saved to lowercase dataset folder: {alt_output_path}")
        except Exception:
            pass

    print("=" * 60)
    print("Sample rows:")
    print(df_final.head(3).to_dict(orient="records"))
    print("=" * 60)
    return df_final

if __name__ == "__main__":
    prepare_dataset()
