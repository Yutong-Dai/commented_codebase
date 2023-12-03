'''
adapted from https://github.com/lansinuote/Diffusion_From_Scratch/blob/main/2.vae.ipynb
'''
import torch


class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        # x -> [1, 128, 10, 10]

        res = x
        if self.res:
            # [1, 128, 10, 10] -> [1, 256, 10, 10]
            res = self.res(x)

        # [1, 128, 10, 10] -> [1, 256, 10, 10]
        return res + self.s(x)


class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)

        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        # x -> [1, 512, 64, 64]
        res = x

        # norm,维度不变
        # [1, 512, 64, 64]
        x = self.norm(x)

        # [1, 512, 64, 64] -> [1, 512, 4096] -> [1, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)

        # 线性运算,维度不变
        # [1, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # [1, 4096, 512] -> [1, 512, 4096]
        k = k.transpose(1, 2)

        # [1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        # 0.044194173824159216 = 1 / 512**0.5
        # atten = q.bmm(k) * 0.044194173824159216

        # 照理来说应该是等价的,但是却有很小的误差
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q,
                              k,
                              beta=0,
                              alpha=0.044194173824159216)

        atten = torch.softmax(atten, dim=2)

        # [1, 4096, 4096] * [1, 4096, 512] -> [1, 4096, 512]
        atten = atten.bmm(v)

        # 线性运算,维度不变
        # [1, 4096, 512]
        atten = self.out(atten)

        # [1, 4096, 512] -> [1, 512, 4096] -> [1, 512, 64, 64]
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)

        # 残差连接,维度不变
        # [1, 512, 64, 64]
        atten = atten + res

        return atten


class Pad(torch.nn.Module):
    """add one column and one row of zeros to the input
    [[1,1],
    [1,1]]
    ->
    [[1,1,0],
    [1,1,0],
    [0,0,0]]
    """

    def forward(self, x):
        return torch.nn.functional.pad(x, (0, 1, 0, 1),
                                       mode='constant',
                                       value=0)


class VAE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            # in with input size bsx3x512x512
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            # down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            # mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            # 正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            # 正态分布层
            torch.nn.Conv2d(4, 4, 1),

            # in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            # middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),

            # up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 3, 3, padding=1),
            ),
        )

    def sample(self, h):
        # h -> [1, 8, 64, 64]

        # [1, 4, 64, 64]
        mean = h[:, :4]
        # [1, 4, 64, 64]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        # [1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):
        # x -> [1, 3, 512, 512]

        # [1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)

        # make h follow the normal distribution
        # [1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)

        # [1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)

        return h
