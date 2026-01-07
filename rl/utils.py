import math

import torch
from  generate_selfies import generate_selfies
import selfies as sf

# -----------------  rollout（π_old 采样） -----------------
def sample_selfies_batch_from_generate_selfies(
    model_name,
    vocab,
    model,
    batch_size=16,
    max_len=80,
    temperature=1.0,
    top_k=None,
    device="cuda",
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
                model_name=model_name,
                vocab=vocab,
                model=model,
                device=device,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k
            )
            tokens = sf.split_selfies(result["selfies"])
            token_ids = [vocab[t] for t in tokens]

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
        return 0.6 * math.exp(-score/3)
    return reward_fn
