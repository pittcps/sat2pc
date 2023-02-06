import torch.nn as nn
from psgn.decoder.simple_decoder import Simple_Decoder
from psgn.decoder.psgn_2branch import PCGN_2Branch
from psgn.encoder.psgn_cond import PCGN_Cond

decoder_dict = {
    'simple': Simple_Decoder,
    'psgn_2branch': PCGN_2Branch
}

encoder_dist = {
    'conditioning' : PCGN_Cond
}


class PSGN(nn.Module):
    r''' The Point Set Generation Network.

    For the PSGN, the input image is first passed to a encoder network,
    e.g. the CNN proposed in the original publication. Next,
    this latent code is then used as the input for the decoder network, e.g.
    the 2-Branch model from the PSGN paper.

    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    '''

    def __init__(self, decoder, encoder, **kwargs):
        super().__init__()

        if decoder not in decoder_dict:
            raise ValueError('Invalid decoder "%s"' % str(decoder))
        if encoder not in encoder_dist:
            raise ValueError('Invalid encoder "%s"' % str(encoder))

        if "c_dim" in kwargs:
            c_dim = kwargs['c_dim']
        else:
            c_dim = 512

        if "dim" in kwargs:
            dim = kwargs['dim']
        else:
            dim = 3

        if "n_points" in kwargs:
            n_points = kwargs['n_points']
        else:
            n_points = 3000

        self.decoder = decoder_dict[decoder](dim, c_dim, n_points)
        self.encoder = encoder_dist[encoder](c_dim)

    def forward(self, r, x):
        c = self.encoder(r, x)
        points = self.decoder(c)
        return points
