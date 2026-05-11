import math

import torch

from  sample.sample import generate_selfies
import selfies as sf
# -----------------  rollout（π_old 采样） -----------------
def sample_selfies_batch_from_generate_selfies(
    model,
    vocab,
    batch_size=16,
    max_len=80,
    temperature=1.0,
    top_k=None,
    device="cuda",
    include_eos=False,
):
    '''
    根据old agent 进行采样
    :param model_name:
    :param vocab:
    :param model:
    :param batch_size:
    :param max_len:
    :param temperature:
    :param top_k:
    :param device:
    :return:
    '''
    model.eval()
    batch_token_ids, batch_smiles = [], []

    with torch.no_grad():
        for _ in range(batch_size):
            result = generate_selfies(
                model=model,
                vocab=vocab,
                device=device,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k
            )
            if include_eos and result.get("tokens"):
                tokens = [
                    tok for tok in result["tokens"]
                    if tok not in ("<SOS>", "<PAD>", "<UNK>")
                ]
            else:
                tokens = list(sf.split_selfies(result["selfies"]))

            token_ids = []
            for tok in tokens:
                try:
                    token_ids.append(vocab[tok])
                except Exception:
                    continue

            if len(token_ids) >= 2:
                batch_token_ids.append(token_ids)
                batch_smiles.append(result["smiles"])

    return batch_token_ids, batch_smiles



def make_reward_fn_from_vina(vina_results, invalid_penalty=0):
    def reward_fn(smiles):
        res = vina_results.get(smiles, None)
        if res is None:
            return invalid_penalty
        score = res.get("score", None)
        if score is None:
            return invalid_penalty
        return vina_reward(score)
    return reward_fn


import math
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import sascorer  # 确保 sascorer.py 在路径中
from feedback.vina_scores import batch_scores_from_vina
from feedback.utils import canonicalize_smiles



def _legacy_make_composite_reward(batch_smiles, vina_results, weights=None, invalid_penalty=-4):
    """
    Legacy-compatible reward function.

    All sub-objectives are converted into rewards where higher is better.
    """
    if weights is None:
        weights = {'vina': 0.7, 'qed': 0.1, 'sa': 0.1, 'logp': 0.1}

    results = []

    for smi in batch_smiles:
        item = {'smiles': smi}

        vina_res = vina_results.get(smi, None)
        if vina_res is None or 'score' not in vina_res or vina_res['score'] is None:
            item['vina'] = invalid_penalty
        else:
            item['vina'] = vina_reward(vina_res['score'])

        mol = Chem.MolFromSmiles(smi)
        if mol:
            item['qed'] = QED.qed(mol)
            try:
                item['sa'] = sa_score_to_reward(sascorer.calculateScore(mol))
            except Exception:
                item['sa'] = invalid_penalty
            item['logp'] = logp_score_to_reward(Descriptors.MolLogP(mol))
        else:
            item['qed'] = invalid_penalty
            item['sa'] = invalid_penalty
            item['logp'] = invalid_penalty

        item['reward'] = (
            weights['vina'] * item['vina'] +
            weights['qed'] * item['qed'] +
            weights['sa'] * item['sa'] +
            weights['logp'] * item['logp']
        )
        results.append(item)

    return results

def make_composite_reward(batch_smiles, vina_results, weights=None, invalid_penalty=-4):
    """
    Unified reward schema.

    Keeps legacy keys: vina, sa, logp, reward.
    Adds traceable keys: vina_raw, vina_reward, sa_raw, sa_reward,
    logp_raw, logp_reward, total_reward, canonical_smiles, status.
    """
    if weights is None:
        weights = {'vina': 0.7, 'qed': 0.1, 'sa': 0.1, 'logp': 0.1}

    results = []
    for smi in batch_smiles:
        canonical = canonicalize_smiles(smi)
        item = {
            'smiles': smi,
            'canonical_smiles': canonical,
            'vina_raw': None,
            'vina_reward': invalid_penalty,
            'qed': invalid_penalty,
            'sa_raw': None,
            'sa_reward': invalid_penalty,
            'logp_raw': None,
            'logp_reward': invalid_penalty,
            'status': 'ok',
        }

        vina_res = vina_results.get(smi, None)
        if vina_res is None and canonical is not None:
            vina_res = vina_results.get(canonical, None)

        if vina_res is None or vina_res.get('score') is None:
            item['status'] = 'missing_vina'
        else:
            vina_score = vina_res['score']
            item['vina_raw'] = vina_score
            item['vina_reward'] = vina_reward(vina_score)
            item['status'] = vina_res.get('status', 'ok')

        mol = Chem.MolFromSmiles(canonical) if canonical else None
        if mol:
            item['qed'] = QED.qed(mol)
            try:
                sa_raw = sascorer.calculateScore(mol)
                item['sa_raw'] = sa_raw
                item['sa_reward'] = sa_score_to_reward(sa_raw)
            except Exception:
                item['status'] = 'sa_failed'

            item['logp_raw'] = Descriptors.MolLogP(mol)
            item['logp_reward'] = logp_score_to_reward(item['logp_raw'])
        else:
            item['status'] = 'invalid_smiles'

        item['vina'] = item['vina_reward']
        item['sa'] = item['sa_reward']
        item['logp'] = item['logp_reward']
        item['reward'] = (
            weights['vina'] * item['vina_reward'] +
            weights['qed'] * item['qed'] +
            weights['sa'] * item['sa_reward'] +
            weights['logp'] * item['logp_reward']
        )
        item['total_reward'] = item['reward']
        results.append(item)

    return results


# batch_smiles = ['CCO', 'C1OC(=O)C=CC=C1[C@H1]2C=3C(C4=CC=CC=C4O)=N[NH1]C=3[C@H1](C5=CC=C(Cl)C=C5)N2C=C']  # 示例分子
# vina_results = batch_scores_from_vina(
#     batch_smiles,
#     receptor_file="../feedback/8sc7.pdbqt",
#     pdbqt_dir="../feedback/temp/pdbqt",
#     output_dir="../feedback/vina_results/"
# )
#
# reward_results = make_composite_reward(batch_smiles, vina_results)
#
# for r in reward_results:
#     print(r)


import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import sascorer


def vina_reward(vina_score, mid=-7.2, scale=1.5):
    """
    Docking score -> reward
    vina_score usually in [-12, -5]
    """
    return 1 / (1 + math.exp((vina_score - mid) / scale))


def sa_score_to_reward(sa_score):
    """
    SA raw score is usually in [1, 10], and lower is better.
    Convert it into a reward in [0, 1], where higher is better.
    """
    return float(np.clip(1.0 - (sa_score - 1.0) / 9.0, 0.0, 1.0))


def logp_score_to_reward(logp_score, target=2.5, sigma=1.0):
    """
    Favor a reasonable LogP interval instead of monotonically larger values.
    Peak reward is around the target value.
    """
    reward = math.exp(-((logp_score - target) ** 2) / (2.0 * sigma ** 2))
    return float(np.clip(reward, 0.0, 1.0))


def make_composite_reward_gspo(batch_smiles, vina_results, weights=None, invalid_penalty=0):

    if weights is None:
        weights = {
            "vina": 0.7,
            "qed": 0.2,
            "sa": 0.1
        }

    results = []

    for smi in batch_smiles:

        item = {"smiles": smi}

        mol = Chem.MolFromSmiles(smi)

        # ===== vina =====
        vina_res = vina_results.get(smi)

        if vina_res is None or vina_res.get("score") is None:
            item["vina"] = invalid_penalty
        else:
            vina_score = vina_res["score"]
            item["vina"] = vina_reward(vina_score)

        # ===== QED =====
        if mol:
            item["qed"] = QED.qed(mol)
        else:
            item["qed"] = invalid_penalty

        # ===== SA =====
        if mol:
            try:
                sa = sascorer.calculateScore(mol)
                item["sa"] = sa_score_to_reward(sa)
            except Exception:
                item["sa"] = invalid_penalty
        else:
            item["sa"] = invalid_penalty

        # ===== composite reward =====
        reward = (
            weights["vina"] * item["vina"] +
            weights["qed"] * item["qed"] +
            weights["sa"] * item["sa"]
        )

        item["reward_raw"] = reward
        results.append(item)

    # ===== reward normalization =====
    rewards = np.array([x["reward_raw"] for x in results])

    mean = rewards.mean()
    std = rewards.std() + 1e-6

    for item in results:
        item["reward"] = (item["reward_raw"] - mean) / std

    return results

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import sascorer


def zscore(x):
    """Z-score normalization"""
    x = np.array(x)
    return (x - x.mean()) / (x.std() + 1e-6)


def make_composite_reward_gdpo(batch_smiles, vina_results, weights=None):

    if weights is None:
        weights = {
            "vina": 1,
            "qed": 1,
            "sa": 1
        }

    results = []

    vina_list = []
    qed_list = []
    sa_list = []

    # ========= first pass: collect metrics =========
    for smi in batch_smiles:

        mol = Chem.MolFromSmiles(smi)

        # vina
        vina_res = vina_results.get(smi)
        vina_score = vina_res["score"] if vina_res and vina_res.get("score") else 0

        # qed
        qed_score = QED.qed(mol) if mol else 0

        # sa
        if mol:
            try:
                sa_score = sascorer.calculateScore(mol)
            except:
                sa_score = 10
        else:
            sa_score = 10

        vina_list.append(vina_score)
        qed_list.append(qed_score)
        sa_list.append(sa_score)

        results.append({
            "smiles": smi,
            "vina_raw": vina_score,
            "qed_raw": qed_score,
            "sa_raw": sa_score
        })

    # ========= metric normalization =========

    vina_norm = zscore(vina_list)

    # vina越负越好 -> 乘负号
    vina_norm = -vina_norm

    qed_norm = zscore(qed_list)

    # sa越小越好 -> 乘负号
    sa_norm = -zscore(sa_list)

    # ========= composite reward =========

    composite_rewards = []

    for i, item in enumerate(results):

        item["vina"] = vina_norm[i]
        item["qed"] = qed_norm[i]
        item["sa"] = sa_norm[i]

        reward = (
            weights["vina"] * item["vina"]
            + weights["qed"] * item["qed"]
            + weights["sa"] * item["sa"]
        )

        item["reward_raw"] = reward
        composite_rewards.append(reward)

    # ========= final normalization =========

    final_rewards = zscore(composite_rewards)

    for i, item in enumerate(results):
        item["reward"] = final_rewards[i]

    return results
