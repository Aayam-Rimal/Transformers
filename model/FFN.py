import numpy as np

class FFN:

    def __init__(self,d_model, d_ff):
        
        self.W1= np.random.randn(d_model,d_ff)
        self.b1= np.zeros((d_ff))

        self.W2= np.random.randn(d_ff,d_model)
        self.b2= np.zeros((d_model))

    def relu(self,x):
        return np.maximum(0,x)
    
    def forward(self,x):

        h= x @ self.W1 + self.b1
        h= self.relu(h)
        out= h @ self.W2 + self.b2
        return out
    

if __name__=="__main__":

    x= np.zeros((5,4))
    ffn= FFN(4,16)
    y=ffn.forward(x)

    print(x.shape)
    print(y.shape)

    