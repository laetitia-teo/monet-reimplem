"""
Reimplementation of the MONet unsupervised object detection model
"""

import torch

class BlockDown_D(torch.nn.Module):
    # with downsampling
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()
        self.downsample = torch.nn.MaxPool2d(2, 2)

    def forward(self, fmap):
        skip = self.relu(self.norm(self.conv(fmap)))
        fmap = self.downsample(skip)
        return skip, fmap

class BlockDown_ND(torch.nn.Module):
    # no downsampling
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, fmap):
        skip = self.relu(self.norm(self.conv(fmap)))
        return skip

class BlockUp_U(torch.nn.Module):
    # with upsampling
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, fmap):
        return self.upsample(self.relu(self.norm(self.conv(fmap))))

class BlockUp_NU(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        # no upsampling
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, fmap):
        return self.relu(self.norm(self.conv(fmap)))

class AttentionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ch_list = [
            (4, 8),
            (8, 16),
            (16, 32),
            (32, 32),
            (32, 32)]
        # downsampling blocks
        self.blockd0 = BlockDown_D(*ch_list[0])
        self.blockd1 = BlockDown_D(*ch_list[1])
        self.blockd2 = BlockDown_D(*ch_list[2])
        self.blockd3 = BlockDown_D(*ch_list[3])
        self.blockd4 = BlockDown_ND(*ch_list[4])
        # upsampling blocks (reverse order)
        self.blocku4 = BlockUp_U(*ch_list[4][::-1])
        self.blocku3 = BlockUp_U(*ch_list[3][::-1])
        self.blocku2 = BlockUp_U(*ch_list[2][::-1])
        self.blocku1 = BlockUp_U(*ch_list[1][::-1])
        self.blocku0 = BlockUp_NU(*ch_list[0][::-1])
        # middle mlp
        self.flatten = torch.nn.Flatten()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )
        # last layer
        self.logitstolog = torch.nn.LogSigmoid()
        self.outconv = torch.nn.Conv2d(4, 1, 1)

    def forward(self, img, scope):
        img = torch.cat([img, scope], 1)
        skip0, fmap = self.blockd0(img)
        skip1, fmap = self.blockd1(fmap)
        skip2, fmap = self.blockd2(fmap)
        skip3, fmap = self.blockd3(fmap)
        skip4 = self.blockd4(fmap)
        vec = self.flatten(skip4)
        fmap = torch.reshape(vec, skip4.shape)
        fmap = self.blocku4(fmap + skip4)
        fmap = self.blocku3(fmap + skip3)
        fmap = self.blocku2(fmap + skip2)
        fmap = self.blocku1(fmap + skip1)
        fmap = self.blocku0(fmap + skip0)
        fmap = self.outconv(fmap)
        alpha = self.logitstolog(fmap)
        one_minus_alpha = self.logitstolog(-fmap)
        mask = scope + alpha
        scope = scope + one_minus_alpha
        return mask, scope

# component vae

class Encoder(torch.nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        ch_list = [
            (4, 32),
            (32, 32),
            (32, 32),
            (32, zdim)]
        encoder_list = []
        encoder_list.append(torch.nn.Conv2d(*ch_list[0], 3, 2, padding=1))
        encoder_list.append(torch.nn.ReLU())
        encoder_list.append(torch.nn.Conv2d(*ch_list[1], 3, 2, padding=1))
        encoder_list.append(torch.nn.ReLU())
        encoder_list.append(torch.nn.Conv2d(*ch_list[2], 3, 2, padding=1))
        encoder_list.append(torch.nn.ReLU())
        encoder_list.append(torch.nn.Conv2d(*ch_list[3], 3, 2, padding=1))
        encoder_list.append(torch.nn.ReLU()) # ?
        encoder_list.append(torch.nn.Flatten())
        encoder_list.append(torch.nn.Linear(zdim * 16, 256))
        encoder_list.append(torch.nn.ReLU())
        encoder_list.append(torch.nn.Linear(256, 32))
        self.net = torch.nn.Sequential(*encoder_list)

    def forward(self, img):
        return self.net(img)

class SpatialBroadcastDecoder(torch.nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        ch_list = [
            (zdim + 2, 32),
            (32, 32),
            (32, 32),
            (32, 32)]
        decoder_list = []
        decoder_list.append(torch.nn.Conv2d(*ch_list[0], 3, 1))
        decoder_list.append(torch.nn.ReLU())
        decoder_list.append(torch.nn.Conv2d(*ch_list[1], 3, 1))
        decoder_list.append(torch.nn.ReLU())
        decoder_list.append(torch.nn.Conv2d(*ch_list[2], 3, 1))
        decoder_list.append(torch.nn.ReLU())
        decoder_list.append(torch.nn.Conv2d(*ch_list[3], 3, 1))
        decoder_list.append(torch.nn.ReLU())
        decoder_list.append(torch.nn.Conv2d(32, 4, 1, 1))
        self.net = torch.nn.Sequential(*decoder_list)
        # meshgrid
        self.S = 64 + 8
        A = (torch.arange(self.S).float() / self.S) * 2 - 1
        x, y = torch.meshgrid(A, A)
        self.grid = torch.stack([x, y], 0).unsqueeze(0)

    def to(self, device):
        self.grid = self.grid.to(device)
        return super().to(device)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        shape = (z.shape[0], z.shape[1], self.S, self.S)
        zcast = z * torch.ones(shape).to(z.device)
        grid = torch.ones((z.shape[0], 1, 1, 1)).to(z.device) * self.grid
        fmap = torch.cat([zcast, grid], 1)
        return self.net(fmap)

class ComponentVAE(torch.nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.zdim = zdim
        self.encoder = Encoder(self.zdim)
        self.decoder = SpatialBroadcastDecoder(self.zdim)
        self.softplus = torch.nn.Softplus()
        
    def forward(self, img):
        params = self.encoder(img)
        mus = params[:, :self.zdim]
        sigmas = self.softplus(params[:, self.zdim:])
        N = mus.shape[0]
        eps = torch.normal(
            torch.zeros((N, self.zdim), device=img.device),
            torch.ones((N, self.zdim), device=img.device))
        z = mus + sigmas * eps
        pred_img = self.decoder(z)
        return pred_img, mus, sigmas # params are for internal part of loss

class MONet(torch.nn.Module):
    def __init__(self, K, beta=1., gamma=1., zdim=16):
        super().__init__()
        # hparams
        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.zdim = zdim
        # nets
        self.cvae = ComponentVAE(self.zdim)
        self.an = AttentionNet()

    def to(self, device):
        self.cvae.decoder.to(device) # for the grid
        return super().to(device)

    def forward(self, img):
        bsize = img.shape[0]
        N = img.shape[0] * img.shape[2] * img.shape[3]
        M = bsize * self.zdim
        mask_list = []
        img_list = []
        param_list = []
        sk = torch.zeros((bsize, 1, 64, 64), device=img.device)
        L = 0. # loss is computed inside forward loop
        for k in range(self.K):
            print(f'k = {k}')
            mk, sk = self.an(img, sk) # keep mk in log units ?
            imgk, muk, sigmak = self.cvae(torch.cat([img, mk], 1))
            mask_list.append(mk.detach())
            img_list.append(imgk.detach())
            param_list.append((muk.detach(), sigmak.detach()))
            # add loss components
            L1 = torch.sum(torch.exp(mk) * (img - imgk[:, :3, ...])**2 / N)
            # print(torch.exp(mk))
            print(f'L1 = {L1}')
            L2 = torch.sum((mk - imgk[:, 3:, ...])**2 / N)
            print(f'L2 = {L2}')
            L3 = - torch.sum(torch.log(sigmak) - sigmak**2 - muk**2) / M
            print(f'L3 = {L3}')
            L += L1 + self.gamma * L2 + self.beta * L3
        return L, mask_list, img_list, param_list

# testing stuff
if __name__ == '__main__':
    imgs = torch.rand((64, 3, 64, 64))
    scope = torch.rand((64, 1, 64, 64))
    an = AttentionNet()
    m, s = an(imgs, scope)
    imgs = torch.cat([imgs, scope], 1)
    cvae = ComponentVAE()
    p, m, s = cvae(imgs)
    imgs = torch.rand((64, 3, 64, 64))
    monet = MONet(7)
    L, mask_list, img_list, param_list = monet(imgs)
    print(L)
    print(len(mask_list))