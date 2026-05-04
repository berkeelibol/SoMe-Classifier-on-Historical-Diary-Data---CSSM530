"""Apply trained SoMe classifier to diary entries."""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def apply_model(model_path: str, input_path: str, output_path: str, model_type: str = "setfit"):
    """Apply a trained model to classify diary entries."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    if model_type == "setfit":
        from setfit import SetFitModel
        model = SetFitModel.from_pretrained(model_path)
        predictions = model.predict(df['content'].tolist())
    elif model_type == "bert":
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model_path, truncation=True)
        results = pipe(df['content'].tolist(), batch_size=32)
        predictions = [r['label'] for r in results]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    df['predicted_label'] = predictions
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")

    # Print distribution
    print("\nLabel distribution:")
    print(df['predicted_label'].value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply SoMe classifier')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--type', default='setfit', choices=['setfit', 'bert'])
    args = parser.parse_args()
    apply_model(args.model, args.input, args.output, args.type)
