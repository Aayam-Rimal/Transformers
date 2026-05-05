import numpy as np

class MHA:

    def __init__(self, d_model, num_heads):

        self.num_heads= num_heads
        self.d_model= d_model
        self.head_dim= d_model // num_heads

        self.Wq= np.random.randn(d_model,d_model)
        self.Wk= np.random.randn(d_model,d_model)
        self.Wv= np.random.randn(d_model,d_model)
        self.Wo = np.random.randn(d_model, d_model)

    
    def softmax(self,x):

        x= x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)



    def forward(self,x):

        B,S,D= x.shape
        H= self.num_heads
        Dh= self.head_dim

        Q= x @ self.Wq
        K= x @ self.Wk
        V= x @ self.Wv


        Q= Q.reshape(B,S,H,Dh)
        K= K.reshape(B,S,H,Dh)
        V= V.reshape(B,S,H,Dh)


        Q= Q.transpose(0,2,1,3)
        K= K.transpose(0,2,1,3)
        V= V.transpose(0,2,1,3)

        score= Q @ K.transpose(0,1,3,2)
        score= score/np.sqrt(Dh)

        attn= self.softmax(score)

        out= attn @ V

        out= out.transpose(0,2,1,3)
        out= out.reshape(B,S,D)

        out= out @ self.Wo

        return out 










        


