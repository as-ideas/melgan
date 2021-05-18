import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        d_model = 1024
        n_head = 8
        layers = 6
        d_fft = 1024
        self.pos_encoder = PositionalEncoding(1024, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=d_fft,
                                                   dropout=0.1,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                 num_layers=layers,
                                                 norm=encoder_norm)

        self.discriminator_1 = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])

        self.discriminator_2 = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])



    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator_1:
            x = module(x)
            features.append(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        features.append(x)
        for module in self.discriminator_2:
            x = module(x)
            features.append(x)

        return features[:-1], features[-1]


if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 1, 16000)
    print(x.shape)

    features, score = model(x)
    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)