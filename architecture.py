import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import math
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps: float = 10**-6):
        super().__init__()
        self.d_model = d_model 
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std_dev = x.var(dim = -1, keepdim = True, unbiased = False)

        norm = self.alpha *(x - mean) / torch.sqrt(std_dev + self.eps) + self.bias
    
        return norm 
    

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, seq_len):
        super().__init__() #calling for nn.module
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_k
        self.seq_len = seq_len
        self.W_O = nn.Linear(self.d_model, self.d_model)
        self.q_pros = nn.Linear(d_model, d_model)
        self.k_pros = nn.Linear(d_model, d_model)
        self.v_pros = nn.Linear(d_model, d_model)
        self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

       #Treating projections as linear layers



    def forward(self,X):
        B,T,C = X.shape

        Q = self.q_pros(X).view(B,T,self.n_heads, self.d_k).transpose(1,2) #swapping T and n_heads to get n_heads first because last two dims are the actual matrix
        K= self.k_pros(X).view(B,T,self.n_heads, self.d_k).transpose(1,2)
        V= self.v_pros(X).view(B,T,self.n_heads, self.d_k).transpose(1,2)
        
        scores = Q@K.transpose(-2,-1) / self.d_k **0.5 #Getting the scores
            
        scores = scores.masked_fill(self.mask[:T, :T] ==0, float('-inf')) #masking future positions in the matrix to -inf

        scores = F.softmax(scores,dim=-1)
       
        out = (scores @ V).transpose(1, 2).contiguous().view(B, T, C) 
        return self.W_O(out)


        """
        contiguous forces PyTorch to copy data into memory; swap dims -> rearrange memory -> reshape
        """
       
       
       
       


class MLP(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model 
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
            

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads


    def forward(self, x):
        ln1 = LayerNorm()
        ln2 = LayerNorm()
        attn = SelfAttention()
        mlp = MLP()

        x = x + attn(ln1(x)) + mlp(ln2(x))
        return x
    

class GPT(nn.Module):
   def __init__(self, vocab_size, d_model):
       super().__init__()
       self.vocab_size = vocab_size
       self.d_model = d_model
       self.token_emb = nn.Embedding(vocab_size, d_model) 
       
    
class PositionalEncoding(nn.Module):
       def __init__(self, d_model, seq_len):
           super().__init__()
           self.d_model = d_model
           self.seq_len = seq_len
           pe = torch.zeros(seq_len, d_model)
           position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)

           div_term = torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0) / d_model))

           pe[:, 0::2] = torch.sin(position *div_term) #applies sine at every even position
           pe[:, 1::2] = torch.cos(position*div_term) #cosine at every odd
             
           pe = pe.unsqueeze(0) #Converts to a tensor by adding 1 as a new dimension so that the shapes of pe and x line up for addition 
           self.register_buffer('pe', pe)
        
       def forward(self, x):
            return x + self.pe[:, :x.size(1)] #pythonic way of applying positional encoding; applying for each token in input
       


