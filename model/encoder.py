import numpy as np
from FFN import FFN
from LayerNorm import LN
from Attention import MHA

class tsf_block:

    def __init__(self,d_model, num_heads, d_ff,eps=1e-5):

        self.ffn= FFN(d_model,d_ff)
        self.ln= LN(d_model, eps=1e-5)
        self.attention= MHA(d_model,num_heads)

        self.eps= eps

    
    def forward(self,x):

        z= self.attention.forward(x)

        residual1= z + x

        ln1= self.ln.norm(residual1)

        ffn1= self.ffn.forward(ln1)

        residual2= ln1 + ffn1

        ln2= self.ln.norm(residual2)

        return ln2

if __name__=="__main__":

    x= np.random.randn(1,4,4)

    block1= tsf_block(4,1,8)

    output= block1.forward(x)

    print(output)
        

