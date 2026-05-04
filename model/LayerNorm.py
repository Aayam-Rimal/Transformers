import numpy as np

class LN:

    def __init__(self, d_model, eps=1e-5):

        self.gamma=np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps= eps

    def norm(self,x):

        mean= np.mean(x, axis=-1, keepdims=True)
        var=  np.var(x, axis=-1, keepdims=True)

        x_norm= (x - mean)/(np.sqrt(var)+self.eps) 

        return self.gamma * x_norm + self.beta
    

if __name__ == "__main__":

    x= np.random.randn(6,16)
    ln= LN(16)

    y= ln.norm(x)

    print(y)
