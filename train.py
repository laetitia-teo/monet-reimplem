import os.path as op
import torch

from pathlib import Path
from data import batched_dataset, to_tensor
from model import MONet

LR = 1e-4
NRUN = 0
MODEL_SAVE_PATH = f'saves/run{NRUN}'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def train(N, model, opt, dataset, data_fn, device):
    llist = []
    for epoch in range(N):
        for i, data in enumerate(dataset):
            opt.zero_grad()
            imgs = data_fn(data).to(device)
            L, ml, il, pl = model(imgs)
            print(f'Epoch: {epoch}, batch: {i} L: {L}')
            llist.append(L.detach().item())
            L.backward()
            opt.step()
        save_model(model, op.join(MODEL_SAVE_PATH, f'epoch{epoch}.pt'))

if __name__ == '__main__':
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model = MONet(7).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train(1, model, opt, batched_dataset, to_tensor, device)