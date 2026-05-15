import numpy as np
from encoder import encoder_block
from decoder import decoder_block

class transformer:

    def __init__(self,d_model, num_heads, d_ff,eps=1e-5):
        
        self.encoder= encoder_block(d_model, num_heads,d_ff,eps)
        self.decoder= decoder_block(d_model,d_ff,num_heads,eps)
        self.eps= eps

    def forward(self,trgt,src):

        enc_out = self.encoder.forward(src)
        dec_out= self.decoder.forward(trgt,enc_out)

        return dec_out
    

if __name__ == "__main__":

    src= np.random.randn(1,5,8)
    trgt=np.random.randn(1,5,8)
    t1= transformer(8,1,16,eps=1e-5)

    output= t1.forward(trgt,src)

    print(output)




    

