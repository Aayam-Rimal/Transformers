import numpy as np
from FFN import FFN
from LayerNorm import LN
from Attention import MHA

class decoder_block:

    def __init__(self, d_model, d_ff, num_heads, eps=1e-5):

        self.ffn= FFN(d_model,d_ff)
        self.ln1= LN(d_model, eps=1e-5)
        self.ln2= LN(d_model, eps=1e-5)
        self.ln3= LN(d_model, eps=1e-5)
        self.attention= MHA(d_model,num_heads)

        self.eps= eps

    def forward(self,x,enc_out):

        B,S,D= x.shape

        mask= np.ones((S,S))
        mask= np.triu(mask, k=1)

        mask= mask * -1e9
        mask = mask[None, None, :, :]

        z= self.attention.forward(x,x,x,mask=mask)
        residual1= z + x
        ln1= self.ln1.norm(residual1)

        z2= self.attention.forward(q=ln1, k=enc_out, v=enc_out, mask=None)

        x= ln1 + z2
        x = self.ln2.norm(x)

        z3 = self.ffn.forward(x)
        x = x + z3
        x = self.ln3.norm(x)

        return x




    






