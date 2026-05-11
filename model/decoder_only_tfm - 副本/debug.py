import torch

state_dict = torch.load("best_model_fold1.pt", map_location="cpu")
for k, v in state_dict.items():
    print(f"{k}: {tuple(v.shape)}")
