# GSPO 5_6 No-Aux Update Record

Date: 2026-05-11

## Baseline Selection

Selected baseline:

`rl/multi_obj_gspo_4_21_5_6_20260508.py`

New generated file:

`rl/multi_obj_gspo_4_21_5_6_20260508_no_aux.py`

Reasons for selecting this baseline:

- It is explicitly labeled as the `GSPO 5_6 20260508 variant`.
- It contains the same core training surface as the May 8 logs: GSPO policy update, KL control, composite reward, SELFIES sampling, Vina docking, and diagnostic logging.
- It contains the auxiliary likelihood loss path through `elite_aux_likelihood_loss`, `aux_loss`, `aux_coef`, `aux_batch_size`, `aux_start`, and `aux_reward_temperature`.
- Compared with the older `rl/multi_obj_gspo_5_6.py`, it is structurally more complete and includes the 20260508 stability changes: unique rollout refill, true-token-KL beta control, and more exploratory sampling defaults.
- Compared with the current `rl/multi_obj_gspo_4_21_5_6.py`, it is the clearer May 8 baseline candidate and is most likely tied to `gspo_5_6_training_log_20260508_152919.csv`.

The baseline file was not modified. The no-aux version was created as a separate copy.

## Core Difference

The new no-aux file keeps the GSPO objective:

`total_loss = policy_loss + kl_beta * kl_loss`

and removes all elite auxiliary likelihood loss calculation from the training loop. The elite buffer remains only as a record of high-reward samples and is still exported to CSV.

## Python Modification Log

| Change | Location | Difference | Reason |
| --- | --- | --- | --- |
| Created no-aux file | `rl/multi_obj_gspo_4_21_5_6_20260508_no_aux.py` | Copied from the selected baseline instead of editing the baseline in place. | Preserve the original May 8 baseline exactly. |
| Updated module docstring | File header | Renamed the variant to `GSPO 5_6 20260508 no-aux variant` and documented that aux loss is removed from training calculations. | Make script intent obvious from the file itself. |
| Removed aux loss helper | Former `elite_aux_likelihood_loss(...)` block | Deleted the helper that computed elite likelihood auxiliary loss. | Prevent accidental use of aux loss in this no-aux variant. |
| Removed aux parameters | `train_gspo_5_6(...)` signature and main call | Removed `aux_batch_size`, `aux_coef`, `aux_start`, and `aux_reward_temperature`. | These settings no longer affect training and should not appear as live knobs. |
| Removed aux columns | Training CSV `columns` list and row dict | Removed `aux_loss`, `aux_batch_size`, and `aux_coef`. | Avoid logging values that are intentionally unused. |
| Removed aux sampling path | Inner update loop | Removed elite sampling for auxiliary likelihood and the call to `elite_aux_likelihood_loss(...)`. | Ensure the optimizer update is driven only by GSPO. |
| Simplified skip condition | Inner update loop | Changed the skip guard from `no GSPO samples and no elite aux samples` to `no GSPO samples`. | There is no auxiliary fallback path in this variant. |
| Simplified total loss | Inner update loop | Changed `total_loss = gspo_loss + aux_coef * aux_loss` to `total_loss = gspo_loss`. | Ensure aux loss cannot participate in gradients. |
| Updated console logging | Training print block | Removed printed `aux=...` value. | Keep runtime output aligned with actual optimization. |
| Added no-aux output names | Log and model save paths | New logs and checkpoints use `gspo_5_6_no_aux_*` / `*_5_6_no_aux_*` names. | Avoid confusing no-aux runs with baseline runs. |
| Stabilized CSV write order | Row append call | Changed to `pd.DataFrame([row], columns=columns).to_csv(...)`. | Keep CSV values aligned with headers after column changes. |

## Removed Aux-Related Identifiers

The no-aux file no longer contains these aux training identifiers:

- `aux_loss`
- `aux_coef`
- `aux_start`
- `aux_batch_size`
- `aux_reward_temperature`
- `elite_aux_likelihood_loss`

## Behavior Kept From Baseline

- GSPO mean-ratio policy objective.
- KL penalty inside `compute_gspo_loss_batch`.
- Adaptive KL beta using `true_token_kl_mean`.
- Unique rollout batch refill with `unique_attempt_multiplier`.
- Composite reward from Vina, QED, SA, and LogP.
- Elite buffer recording and CSV export.
- Reward/top5 EMA tracking, plateau early stopping, KL early stopping, and best-model saving.

## Verification

Completed:

- `python -m py_compile rl/multi_obj_gspo_4_21_5_6_20260508_no_aux.py` passed.
- Search confirmed no `aux_loss`, `aux_coef`, `aux_start`, `aux_batch_size`, `aux_reward_temperature`, `elite_aux_likelihood_loss`, or `Elite Aux` identifiers remain in the new Python file.
- AST inspection confirmed `train_gspo_5_6(...)` has no `aux*` parameters.
- AST inspection confirmed the training CSV columns contain no aux fields.
- Baseline-to-no-aux diff summary: 19 insertions and 82 deletions.
