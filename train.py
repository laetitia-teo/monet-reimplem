import os.path as op
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from data import batched_dataset, to_tensor
from model import MONet

LR = 1e-4
NRUN = 1
MODEL_SAVE_PATH = f'saves/run{NRUN}'
BETA = 0.5
GAMMA = 0.5
K = 5

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
            imgs = data_fn(data).to(device)
            nll, kl_loss, mask_loss, img_recs, masks = model(imgs)
            print(f'epoch {epoch}, batch {i}')
            print(f'nll {nll}')
            print(f'kl_loss {kl_loss}')
            print(f'mask_loss {mask_loss}')
            L = nll + BETA * kl_loss + GAMMA * mask_loss
            llist.append(L.item())
            opt.zero_grad()
            L.backward()
            opt.step()
        save_model(model, op.join(MODEL_SAVE_PATH, f'epoch{epoch}.pt'))
    return llist

data = next(iter(batched_dataset))
imgs = to_tensor(data)

def test_model(N, ep):
    m = load_model(MONet(K), MODEL_SAVE_PATH + f'/epoch{ep}.pt')
    reco, img_recs, masks = m.reconstruction(imgs)
    for i in range(N):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(imgs[i].permute(1, 2, 0))
        axs[1].imshow(reco[i].permute(1, 2, 0))
        fig2, axs2 = plt.subplots(2, K)
        for j in range(K):
            axs2[0, j].matshow(masks[i, j, 0])
            axs2[1, j].imshow(img_recs[i, j].permute(1, 2, 0))
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model = MONet(K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    llist = train(10, model, opt, batched_dataset, to_tensor, device)
    plt.plot(llist)