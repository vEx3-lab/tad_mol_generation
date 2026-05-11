import os

import pandas as pd

from feedback.utils import canonicalize_smiles, from_smi_2_pdbqt


def _load_vina_cache(cache_file):
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return {}

    cache = {}
    for _, row in df.iterrows():
        smi = row.get("canonical_smiles")
        if not isinstance(smi, str) or not smi:
            continue
        score = row.get("score")
        cache[smi] = {
            "score": None if pd.isna(score) else float(score),
            "status": row.get("status", "cached"),
        }
    return cache


def _append_vina_cache(cache_file, rows):
    if not cache_file or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(cache_file):
        old = pd.read_csv(cache_file)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["canonical_smiles"], keep="last")
    df.to_csv(cache_file, index=False)


def batch_scores_from_vina(
    smiles_list,
    receptor_file="./8SC7.pdbqt",
    pdbqt_dir="./temp/pdbqt",
    output_dir="./vina_results/",
    cache_file=None,
    use_cache=True,
    keep_ligand_pdbqt=False,
    save_poses=False,
):
    """
    Batch-score SMILES with AutoDock Vina.

    Returns:
        dict: {original_smiles: {"score": float or None, "status": str,
        "canonical_smiles": str or None}}
    """
    os.makedirs(output_dir, exist_ok=True)
    if keep_ligand_pdbqt:
        os.makedirs(pdbqt_dir, exist_ok=True)

    if cache_file is None:
        cache_file = os.path.join(output_dir, "vina_cache.csv")
    cache = _load_vina_cache(cache_file) if use_cache else {}

    results = {}
    to_score = []
    for smi in smiles_list:
        canonical = canonicalize_smiles(smi)
        if canonical is None:
            results[smi] = {
                "score": None,
                "status": "invalid_smiles",
                "canonical_smiles": None,
            }
            continue

        if use_cache and canonical in cache:
            cached = cache[canonical]
            results[smi] = {
                "score": cached["score"],
                "status": "cached",
                "canonical_smiles": canonical,
            }
            continue

        to_score.append((smi, canonical))

    if not to_score:
        return results

    from vina import Vina

    v = Vina(sf_name="vina")
    v.set_receptor(receptor_file)
    v.compute_vina_maps(
        center=[24.6971, 7.5657, 58.6465],
        box_size=[79.498, 76.783, 61.505],
    )

    cache_rows = []
    scored_this_run = {}
    for original_smi, canonical in to_score:
        if canonical in scored_this_run:
            item = scored_this_run[canonical]
            results[original_smi] = dict(item)
            continue

        pdbqt_path = from_smi_2_pdbqt(
            canonical,
            pdbqt_dir,
            keep_pdbqt=keep_ligand_pdbqt,
        )

        if not isinstance(pdbqt_path, str) or not os.path.exists(pdbqt_path):
            status = pdbqt_path if isinstance(pdbqt_path, str) else "pdbqt_failed"
            item = {
                "score": None,
                "status": status,
                "canonical_smiles": canonical,
            }
            results[original_smi] = item
            scored_this_run[canonical] = item
            cache_rows.append({
                "canonical_smiles": canonical,
                "score": None,
                "status": status,
            })
            continue

        try:
            v.set_ligand_from_file(pdbqt_path)
            v.dock(exhaustiveness=2, n_poses=1)
            score = float(v.score()[0])

            if save_poses:
                ligand_name = os.path.basename(pdbqt_path)
                output_pdbqt = os.path.join(
                    output_dir, ligand_name.replace(".pdbqt", "_out.pdbqt")
                )
                v.write_poses(output_pdbqt, n_poses=1, overwrite=True)

            item = {
                "score": score,
                "status": "success",
                "canonical_smiles": canonical,
            }
        except Exception as e:
            item = {
                "score": None,
                "status": f"docking_failed: {e}",
                "canonical_smiles": canonical,
            }
        finally:
            if not keep_ligand_pdbqt and os.path.exists(pdbqt_path):
                os.remove(pdbqt_path)

        results[original_smi] = item
        scored_this_run[canonical] = item
        cache_rows.append({
            "canonical_smiles": canonical,
            "score": item["score"],
            "status": item["status"],
        })

    if use_cache:
        _append_vina_cache(cache_file, cache_rows)

    return results
