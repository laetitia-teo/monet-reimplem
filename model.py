"""
Reimplementation of the MONet unsupervised object detection model
"""

import torch
import torch.nn.functional as F

# TODO
# add central component in unet
# higher number of channels in unet
# use F.interpolate in unet


class UBlock(torch.nn.Module):
    # with downsampling
    def __init__(self, in_ch, out_ch, scale_factor=1.):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()
        self.scale_factor = scale_factor

    def forward(self, fmap):
        if self.scale_factor < 1.:
            # block for the downsampling path
            skip = self.relu(self.norm(self.conv(fmap)))
            fmap = F.interpolate(
                skip,
                scale_factor=self.scale_factor,
                mode='bilinear')
            return skip, fmap
        elif self.scale_factor > 1.:
            # block for the upsampling path
            fmap = F.interpolate(
                fmap,
                scale_factor=self.scale_factor,
                mode='bilinear')
            return self.relu(self.norm(self.conv(fmap))), None
        else:
            # no up- or downsampling
            fmap = self.relu(self.norm(self.conv(fmap)))
            return fmap, fmap

class AttentionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ch_list = [
            (4, 32),
            (32, 32),
            (32, 64),
            (64, 64),
            (64, 128)]
        # downsampling blocks
        self.down = torch.nn.ModuleList()
        self.up = torch.nn.ModuleList()
        n = len(ch_list)
        for i in range(len(ch_list)):
            if i < n - 1:
                self.down.append(UBlock(*ch_list[i], scale_factor=.5))
                self.up.append(
                    UBlock(*ch_list[::-1][i][::-1], scale_factor=2.))
            else:
                self.down.append(UBlock(*ch_list[i]))
                self.up.append(UBlock(*ch_list[::-1][i][::-1]))
        self.mid = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, 1),
        )
        # last layer
        self.logsigmoid = torch.nn.LogSigmoid()
        self.outconv = torch.nn.Conv2d(4, 1, 1)

    def forward(self, img, scope):
        fmap = torch.cat([img, scope], 1)
        skips = []
        for i, downblock in enumerate(self.down):
            skip, fmap = downblock(fmap)
            skips.append(skip)
        fmap = self.mid(fmap)
        for i, upblock in enumerate(self.up):
            fmap, _ = upblock(fmap + skips[::-1][i])
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
        masks = torch.clamp_min(log_masks.exp(), min=1e-9)
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