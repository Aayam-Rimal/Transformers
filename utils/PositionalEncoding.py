import numpy as np

class sinosiudal:

    def __init__(self, seq_len, d_model):

        assert d_model % 2 == 0

        self.seq_len= seq_len
        self.d_model= d_model
        
        pos= np.arange(seq_len)[:, np.newaxis]
        i= np.arange(d_model)[np.newaxis, :]

        self.div_term = 1 / (10000 ** (2 * (i // 2) / d_model))
        self.pe = pos * self.div_term



    def forward(self,x):

        pe= self.pe.copy()

        pe[:, 0::2]= np.sin(pe[:,0::2])
        pe[:, 1::2]= np.cos(pe[:,0::2])

        return pe + x
    
    



