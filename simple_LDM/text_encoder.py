'''
adapted from https://github.com/lansinuote/Diffusion_From_Scratch/blob/main/1.encoder.ipynb
'''

import torch


class Embed(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 49408 is the size of the vocabulary
        # 768 is the size of the embedding
        self.embed = torch.nn.Embedding(49408, 768)
        self.pos_embed = torch.nn.Embedding(77, 768)
        # 77 is the max length of the text
        # pos_ids is the position of each word in the text and not trainable
        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0))

    def forward(self, input_ids):
        # input_ids -> [b, 77]

        # [b, 77] -> [b, 77, 768]
        embed = self.embed(input_ids)

        # [1, 77] -> [1, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)

        # [b, 77, 768]
        return embed + pos_embed


class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768)
        self.k = torch.nn.Linear(768, 768)
        self.v = torch.nn.Linear(768, 768)
        self.out = torch.nn.Linear(768, 768)

    def forward(self, x):
        # x -> [b, 77, 768]

        b = x.shape[0]

        # 维度不变
        # [b, 77, 768]
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        # 拆分注意力头
        # [b, 77, 768] -> [b, 77, 12, 64] -> [b, 12, 77, 64] -> [b*12, 77, 64]
        q = q.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        k = k.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)
        v = v.reshape(b, 77, 12, 64).transpose(1, 2).reshape(b * 12, 77, 64)

        # 计算qk乘积
        # [b*12, 77, 64] * [b*12, 64, 77] -> [b*12, 77, 77]
        attn = torch.bmm(q, k.transpose(1, 2))

        # [b*12, 77, 77] -> [b, 12, 77, 77]
        attn = attn.reshape(b, 12, 77, 77)

        # 覆盖mask
        def get_mask(b):
            mask = torch.empty(b, 77, 77)

            # 上三角的部分置为负无穷
            mask.fill_(-float('inf'))

            # 对角线和以下的位置为0
            mask.triu_(1)

            return mask.unsqueeze(1)

        # [b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        # why use masked self attention?
        """
        https://github.com/openai/CLIP/issues/179
        Masked self-attention was used in the text encoder to preserve the ability to initialize with a pre-trained language model 
        or add language modeling as an auxiliary objective, though exploration of this is left as future work.
        """
        attn = attn + get_mask(attn.shape[0]).to(attn.device)

        # [b, 12, 77, 77] -> [b*12, 77, 77]
        attn = attn.reshape(b * 12, 77, 77)

        # 计算softmax,被mask的部分值为0
        attn = attn.softmax(dim=-1)

        # 计算和v的乘积
        # [b*12, 77, 77] * [b*12, 77, 64] -> [b*12, 77, 64]
        attn = torch.bmm(attn, v)

        # [b*12, 77, 64] -> [b, 12, 77, 64] -> [b, 77, 12, 64] -> [b, 77, 768]
        attn = attn.reshape(b, 12, 77, 64).transpose(1, 2).reshape(b, 77, 768)

        # 线性输出,维度不变
        # [b, 77, 768]
        return self.out(attn)


class ClipEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
        )

        self.s3 = torch.nn.Linear(3072, 768)

    def forward(self, x):
        # x -> [2, 77, 768]

        # 维度不变
        # [2, 77, 768]
        x = x + self.s1(x)

        # [2, 77, 768]
        res = x

        # [2, 77, 768] -> [2, 77, 3072]
        x = self.s2(x)

        # 维度不变
        # [2, 77, 3072]
        # quick gelu
        x = x * (x * 1.702).sigmoid()

        # [2, 77, 3072] -> [2, 77, 768]
        return res + self.s3(x)


class TextEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        list_of_modules = [Embed()] + [ClipEncoder()]*12 + \
            [torch.nn.LayerNorm(768)]
        self.encoder = torch.nn.Sequential(*list_of_modules)

    def forward(self, input_ids):
        return self.encoder(input_ids)


if __name__ == '__main__':
    # print(Embed())
    # print(Embed()(torch.ones(2, 77).long()))

    # print(Atten())
    # print(Atten()(torch.randn(2, 77, 768)).shape)

    # print(ClipEncoder())
    # print(ClipEncoder()(torch.randn(2, 77, 768)).shape)
    textencoder = TextEncoder()
    print(textencoder(torch.arange(77).unsqueeze(dim=0)))
