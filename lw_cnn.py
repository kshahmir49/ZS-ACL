import torch.nn as nn

class lw_cnn(nn.Module):
    def __init__(self,n_chan,chan_embed=64):
        super(lw_cnn, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_chan, int(chan_embed/2), kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.SELU(inplace=True),
            nn.Conv2d(int(chan_embed/2), chan_embed, kernel_size=1),
            nn.SELU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(chan_embed, int(chan_embed/2), kernel_size=1),
            nn.SELU(inplace=True),
            nn.Conv2d(int(chan_embed/2), n_chan, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x