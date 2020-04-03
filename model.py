"""
Reimplementation of the MONet unsupervised object detection model
"""

import torch
import torch.nn.functional as F

# TODO
# add central component in unet
# higher number of channels in unet
# use F.interpolate in unet


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
        self.logsigmoid = torch.nn.LogSigmoid()
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
        alpha = self.logsigmoid(fmap)
        one_minus_alpha = self.logsigmoid(-fmap)
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
        for ch_in, ch_out in ch_list:
            encoder_list.append(
                torch.nn.Conv2d(ch_in, ch_out, 3, 2, padding=1))
            encoder_list.append(torch.nn.ReLU())
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
        for ch_in, ch_out in ch_list:
            decoder_list.append(torch.nn.Conv2d(ch_in, ch_out, 3, 1))
            decoder_list.append(torch.nn.ReLU())
        decoder_list.append(torch.nn.Conv2d(32, 4, 1, 1))
        decoder_list.append(torch.nn.Sigmoid())
        self.net = torch.nn.Sequential(*decoder_list)
        # meshgrid
        self.S = 64 + 8
        A = torch.linspace(-1, 1, self.S)
        x, y = torch.meshgrid(A, A)
        x_grid = x.view((1, 1) + x.shape)
        y_grid = y.view((1, 1) + y.shape)
        self.register_buffer('grid', torch.cat([x_grid, y_grid], 1))

    def to(self, device):
        self.grid = self.grid.to(device)
        return super().to(device)

    def forward(self, z):
        z = z.view(z.shape + (1, 1))
        z = z.expand(-1, -1, self.S, self.S)
        fmap = torch.cat([z, self.grid.expand(z.shape[0], -1, -1, -1)], 1)
        return self.net(fmap)

class ComponentVAE(torch.nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.zdim = zdim
        self.encoder = Encoder(self.zdim)
        self.decoder = SpatialBroadcastDecoder(self.zdim)
        self.softplus = torch.nn.Softplus()

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # maybe change to softplus
        eps = torch.rand_like(std)
        return mu + eps * std
        
    def forward(self, img):
        params = self.encoder(img)
        mu, logvar = torch.chunk(params, 2, dim=1)
        z = self.sample(mu, logvar)
        pred_img = self.decoder(z)
        return pred_img, mu, logvar

def mykl(input, probs):
    # maybe implement this if we still have problems with kl
    pass

class MONet(torch.nn.Module):
    def __init__(self, K, scale=0.1, beta=1., gamma=1., zdim=16):
        super().__init__()
        # hparams
        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.zdim = zdim
        self.S = 64
        self.im_ch = 3
        s0 = torch.zeros((1, 1, self.S, self.S))
        self.register_buffer('s0', s0) # init scope
        self.register_buffer('scale', torch.empty(1, 1, 1, 1).fill_(scale))
        # nets
        self.cvae = ComponentVAE(self.zdim)
        self.an = AttentionNet()

    def forward(self, img):
        bsize = img.shape[0]
        log_scope = self.s0.expand(bsize, -1, -1, -1)
        scale = self.scale.expand_as(img)

        logprobs = []
        img_recs = []
        log_masks = []
        log_mask_recs = []
        
        kl_sum = 0.
        
        for k in range(self.K):
            if k < self.K - 1:
                log_mask, log_scope = self.an(img, log_scope)
            else:
                log_mask = log_scope
            out, mu, logvar = self.cvae(torch.cat([img, log_mask], 1))
        
            # image reconstruction
            rimg, rmask = torch.split(out, self.im_ch, dim=1)
            rec_dist = torch.distributions.Normal(rimg, scale)
            logprobs.append(log_mask + rec_dist.log_prob(img)) # log-likelihood
        
            # kl div with latent prior
            kl = - 0.5 * (1 + logvar - mu**2 - torch.exp(logvar))
            kl_sum += torch.sum(kl, -1)
        
            # mask = torch.clamp_min(torch.exp(log_mask), min=1e-9)
            img_recs.append(rimg)
            log_masks.append(log_mask)
            log_mask_recs.append(rmask)

        nll = - torch.logsumexp(torch.stack(logprobs, 1), 1).sum() / bsize
        kl_loss = kl_sum.mean()

        img_recs = torch.stack(img_recs, 1)
        log_mask_recs = torch.stack(log_mask_recs, 1)
        log_masks = torch.stack(log_masks, 1)
        # masks should already sum to 1
        # masks = torch.softmax(log_masks, 1)
        log_rmask = F.log_softmax(log_mask_recs, 1)
        mask_loss = F.kl_div(log_mask_recs, masks, reduction='batchmean')

        return nll, kl_loss, mask_loss, img_recs, masks

    def reconstruction(self, img):
        with torch.no_grad():
            _, _, _, img_recs, masks = self.forward(img)
            masked = torch.sum(img_recs * masks, 1)
        return masked, img_recs, masks

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
    nll, kl_loss, mask_loss, img_recs, masks = monet(imgs)
    print(f'nll {nll}')
    print(f'kl_loss {kl_loss}')
    print(f'mask_loss {mask_loss}')