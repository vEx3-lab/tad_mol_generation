import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from feedback.utils import canonicalize_smiles

try:
    from rl import sascorer
except Exception:
    sascorer = None


def _pick_smiles_column(df, smiles_column=None):
    if smiles_column:
        return smiles_column
    for candidate in ("canonical_smiles", "smiles", "generated_smiles", "SMILES"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def _load_training_set(path, smiles_column=None):
    if not path:
        return None
    df = pd.read_csv(path)
    col = _pick_smiles_column(df, smiles_column)
    return {
        smi
        for smi in (canonicalize_smiles(x) for x in df[col].dropna().tolist())
        if smi is not None
    }


def evaluate_smiles_csv(
    input_csv,
    output_prefix=None,
    smiles_column=None,
    training_csv=None,
    training_smiles_column=None,
):
    input_csv = Path(input_csv)
    output_prefix = Path(output_prefix) if output_prefix else input_csv.with_suffix("")

    df = pd.read_csv(input_csv)
    col = _pick_smiles_column(df, smiles_column)
    raw_smiles = df[col].dropna().astype(str).tolist()

    training_set = _load_training_set(training_csv, training_smiles_column)

    records = []
    for raw in raw_smiles:
        canonical = canonicalize_smiles(raw)
        item = {
            "input_smiles": raw,
            "canonical_smiles": canonical,
            "valid": canonical is not None,
            "novel": None,
            "qed": None,
            "sa": None,
            "logp": None,
        }

        if canonical is not None:
            mol = Chem.MolFromSmiles(canonical)
            item["novel"] = None if training_set is None else canonical not in training_set
            item["qed"] = QED.qed(mol)
            item["logp"] = Descriptors.MolLogP(mol)
            if sascorer is not None:
                try:
                    item["sa"] = sascorer.calculateScore(mol)
                except Exception:
                    item["sa"] = None

        records.append(item)

    result_df = pd.DataFrame(records)
    valid_df = result_df[result_df["valid"]].copy()
    unique_valid = valid_df["canonical_smiles"].drop_duplicates()

    total = len(result_df)
    valid_count = len(valid_df)
    unique_count = len(unique_valid)
    novelty = None
    if training_set is not None and valid_count:
        novelty = float(valid_df["novel"].fillna(False).mean())

    summary = {
        "input_csv": str(input_csv),
        "total": total,
        "valid": valid_count,
        "validity": valid_count / total if total else 0,
        "unique_valid": unique_count,
        "uniqueness": unique_count / valid_count if valid_count else 0,
        "duplicate_ratio": 1 - (unique_count / valid_count) if valid_count else 0,
        "novelty": novelty,
        "avg_qed": float(valid_df["qed"].dropna().mean()) if valid_count else 0,
        "avg_sa": float(valid_df["sa"].dropna().mean()) if "sa" in valid_df else 0,
        "avg_logp": float(valid_df["logp"].dropna().mean()) if valid_count else 0,
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    molecules_path = output_prefix.with_name(output_prefix.name + "_molecules.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")

    result_df.to_csv(molecules_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return summary, molecules_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecular SMILES.")
    parser.add_argument("input_csv")
    parser.add_argument("--output-prefix")
    parser.add_argument("--smiles-column")
    parser.add_argument("--training-csv")
    parser.add_argument("--training-smiles-column")
    args = parser.parse_args()

    summary, molecules_path, summary_path = evaluate_smiles_csv(
        input_csv=args.input_csv,
        output_prefix=args.output_prefix,
        smiles_column=args.smiles_column,
        training_csv=args.training_csv,
        training_smiles_column=args.training_smiles_column,
    )

    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"molecules_csv: {molecules_path}")
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
