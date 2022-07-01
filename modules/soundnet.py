import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1dTF(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size//2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, bias=False, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        self.block_f = nn.Sequential(
                       nn.Conv1d(dim, dim, 1, 1, bias=False),
                       nn.BatchNorm1d(dim),
                       nn.LeakyReLU(0.2, True)
        )
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)
    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=64, n_layers=6, nhead=6, n_classes=None, dim_feedforward=512):
        super(TAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, activation="gelu", dim_feedforward=dim_feedforward, dropout=drop_rate)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + self.num_tokens, embed_dim))
        self.fc = nn.Linear(embed_dim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,  nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        pred = self.fc(o[0])
        return pred


class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size//2, self.filt_size//2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels*2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels*2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels*2, stride=d, filt_size=k)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class SoundNetRaw(nn.Module):
    def __init__(self, nf=32, clip_length=None, embed_dim=128, n_layers=4, nhead=8, factors=[4, 4, 4, 4], n_classes=None, dim_feedforward=512):
        super().__init__()
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, nf, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
        ]
        self.start = nn.Sequential(*model)
        model = []
        for i, f in enumerate(factors):
            model += [Down(channels=nf, d=f, k=f*2+1)]
            nf *= 2
            if i % 2 == 0:
                model += [ResBlock1dTF(dim=nf, dilation=1, kernel_size=15)]
        self.down = nn.Sequential(*model)

        factors = [2, 2]
        model = []
        for _, f in enumerate(factors):
            for i in range(1):
                for j in range(3):
                    model += [ResBlock1dTF(dim=nf, dilation=3 ** j, kernel_size=15)]
            model += [Down(channels=nf, d=f, k=f*2+1)]
            nf *= 2
        self.down2 = nn.Sequential(*model)
        self.project = nn.Conv1d(nf, embed_dim, 1)
        self.clip_length = clip_length
        self.tf = TAggregate(embed_dim=embed_dim, clip_length=clip_length, n_layers=n_layers, nhead=nhead, n_classes=n_classes, dim_feedforward=dim_feedforward)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.start(x)
        x = self.down(x)
        x = self.down2(x)
        x = self.project(x)
        pred = self.tf(x)
        return pred


if __name__ == '__main__':
    pass