import random
from micrograd.engine import Value
import math
from typing import List

'''
possible future additions:
- training with multiple batches
- adding dropout
'''

'''
general todos:
- should check that all methods have init, call and parameters functions (and possibly repr for vis)
- should check that we always have access to correct parameters which get updated
 (that we save local copies in some operations) if we need parameters, so that we can get gradients later, etc.
'''

class Module:

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True, use_bias=False):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.use_bias=use_bias
        if self.use_bias:
            self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        if self.use_bias:
            act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        else:
            act = sum((wi*xi for wi,xi in zip(self.w, x)))
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Linear(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def get_weights(self):
        return [n.w for n in self.neurons]
    
    def get_biases(self):
        return [n.b for n in self.neurons]

    def __repr__(self):
        return f"Linear layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class Sequential(Module):

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Sequential of [{', '.join(str(layer) for layer in self.layers)}]"

class MLP(Module):

    def __init__(self, nin, nouts,use_bias=True):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1,use_bias=use_bias) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class Conv2D(Module):
    def __init__(self,out_channels=1,kernel_size=(3,3),input_shape=(1,8,8),padding_type="none",use_bias=True,stride=(1,1)):
        assert padding_type in ["none"] #TODO: add same padding
        c_in,h_in,w_in=input_shape
        self.padding=(0,0)
        self.dilation=(1,1)
        self.stride=stride
        self.kernel_size=kernel_size
        self.c_out=out_channels
        self.h_out=math.floor((h_in+2*self.padding[0]-self.dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
        self.w_out=math.floor((w_in+2*self.padding[1]-self.dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

        self.filters=[Linear(kernel_size[0]*kernel_size[1],use_bias=use_bias) for _ in range(self.c_out)]
        self.feature_maps=[
            Linear(self.h_out*self.w_out) for _ in range(self.c_out)
        ]
        
    def call(self,x):
        for curr_filter,idx in enumerate(self.filters):
            start_pos_x=0
            end_pos_x=len(x[0])-self.kernel_size
            start_pos_y=0
            end_pos_y=len(x[0])
            for x_start,x_cnt in enumerate(range(start_pos_x,end_pos_x,self.stride)):
                for y_start,y_cnt in enumerate(range(start_pos_y,end_pos_y,self.stride)):
                    if self.padding=="same":
                        #TODO: double check start/end pos
                        curr_positions=[
                            x_curr*self.input_size+y_curr 
                            for x_curr in range(x_start,x_start+self.kernel_size[0]) 
                            for y_curr in range(y_start,y_start+self.kernel_size[1])
                        ]
                    else:
                        #TODO: double check start/end pos
                        curr_positions=[
                            x_curr*self.input_size+y_curr 
                            for x_curr in range(x_start,x_start+self.kernel_size[0]) 
                            for y_curr in range(y_start,y_start+self.kernel_size[1])
                        ]
                    curr_inputs=[x[t] for t in curr_positions]
                    self.feature_maps[idx][x_cnt][y_cnt]+=curr_filter(curr_inputs)
            return self.feature_maps
        
    def parameters(self):
        return [p for filter in self.filters for p in filter.parameters()]


class MaxPool2D(Module):
    def __init__(self,out_channels=1,kernel_size=(3,3),input_shape=(1,8,8),padding_type="none",use_bias=True,stride=(1,1)):
        assert padding_type in ["none"] #TODO: add same padding
        c_in,h_in,w_in=input_shape
        self.padding=(0,0)
        self.dilation=(1,1)
        self.stride=stride
        self.kernel_size=kernel_size
        self.c_out=out_channels
        self.h_out=math.floor((h_in+2*self.padding[0]-self.dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
        self.w_out=math.floor((w_in+2*self.padding[1]-self.dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

        self.filters=[Linear(kernel_size[0]*kernel_size[1],use_bias=use_bias) for _ in range(self.c_out)]
        self.feature_maps=[
            Linear(self.h_out*self.w_out) for _ in range(self.c_out)
        ]
        
    def call(self,x):
        for curr_filter,idx in enumerate(self.filters):
            start_pos_x=0
            end_pos_x=len(x[0])-self.kernel_size
            start_pos_y=0
            end_pos_y=len(x[0])
            for x_start,x_cnt in enumerate(range(start_pos_x,end_pos_x,self.stride)):
                for y_start,y_cnt in enumerate(range(start_pos_y,end_pos_y,self.stride)):
                    if self.padding=="same":
                        #TODO: double check start/end pos
                        curr_positions=[
                            x_curr*self.input_size+y_curr 
                            for x_curr in range(x_start,x_start+self.kernel_size[0]) 
                            for y_curr in range(y_start,y_start+self.kernel_size[1])
                        ]
                    else:
                        #TODO: double check start/end pos
                        curr_positions=[
                            x_curr*self.input_size+y_curr 
                            for x_curr in range(x_start,x_start+self.kernel_size[0]) 
                            for y_curr in range(y_start,y_start+self.kernel_size[1])
                        ]
                    curr_inputs=[x[t] for t in curr_positions]
                    self.feature_maps[idx][x_cnt][y_cnt]+=curr_filter(curr_inputs)
            return self.feature_maps
    def parameters(self):
        pass

class Flatten(Module):
    def __init__(self):
        pass
    def call(self,x):
        if isinstance(x,list):
            out=[]
            for a in x:
                out.append(Flatten(a))
            return out
        else:
            return x
    def parameters(self):
        return []

class Softmax(Module):
    #TODO: use more numerically stable version where we divide everything by max value
    def __init__(self):
        pass

    def __call__(self, values):
        sum_exp = Value(0)
        out_values = []
        # max_val=-1e30
        # for val in values:
        #     max_val=max(val,max_val)
        for val in values:
            sum_exp+=val.exp()
        for val in values:
            out_values.append(val.exp()/sum_exp)
        return out_values
    
class Sigmoid(Module):
    def __init__(self):
        pass
    def __call__(self, values):
        outs = []
        for value in values:
            outs.append(1 / (1 + (-value).exp()))
        return outs

class CrossEntropyLoss(Module):
    #TODO: check that everything is fine with logits vs self.logits, etc.
    def __init__(self, reduction="mean", epsilon=1e-8):
        #only supporting unweighted cross-entropy loss
        self.reduction = reduction
        self.epsilon = epsilon
        assert(self.reduction in ["mean","sum"])
    def __call__(self, logits, values):
        loss = Value(0)
        num_classes = len(values)
        for logit, label in zip(logits,values):
            logit_clipped = logit.clip(self.epsilon,1-self.epsilon)
            loss-=logit_clipped.log()*label
        if self.reduction=="mean":
            return loss/num_classes
        return loss

class BinaryCrossEntropyLoss(Module):
    
    def __init__(self):
        pass

    def __call__(self, logit, label):
        if isinstance(logit, list):
            logit = logit[0]
        if isinstance(label, list):
            label = label[0]
       
        assert isinstance(logit, Value) and isinstance(label, Value)
        return -label * logit.log() -(1 - label) * (1 - logit).log()
       
# class TrainableMatrix(Module):
#     #TODO: add bias as option
#     def __init__(self, height, width):
#         self.width=width
#         self.height=height
#         self.values=[[Value(random.uniform(-1,1)) for _ in range(width)] for _ in range(height)]
#         self.outputs=[Value(0) for _ in range(height)]
#     #this is sort of the same thing as Linear above, except we have everything in one place, which makes it easier to do matmul, etc.
#     def __call__(self,x):
#         for i in range(self.height):
#             for j in range(self.width):
#                 self.outputs[i]+=self.values[j]*x[j]
#         return self.outputs
#     def parameters(self):
#         return [p for row in self.values for p in row]
    
#     def __repr__(self):
#         pass
    
# class MatMul(Module):
#     def __init__(self, mat1, mat2):
#         self.mat1=mat1
#         self.mat2=mat2
#         assert mat1.width==mat2.height
#         self.out=TrainableMatrix(mat1.height,mat2.width)
#     def __call__(self, mat1, mat2, transpose_b=False):
#         if not transpose_b:
#             for i in range(mat1.height):
#                 for j in range(mat2.width):
#                     for k in range(mat1.width):
#                         self.out[i][j]+=mat1[i][k]*mat2[k][j]
#         else:
#             for i in range(mat1.height):
#                 for j in range(mat2.height):
#                     for k in range(mat1.width):
#                         self.out[i][j]==mat1[i][k]*mat2[j][k]
#         return self.out
        
#     def parameters(self):
#         return self.mat1.parameters()+self.mat2.parameters()+self.out.parameters()

# class Scale(Module):
#     def __init__(self,mat,k):
#         self.k=k
#         self.mat=mat
#     def __call__(self,mat,k):
#         for i in range(len(mat.values)):
#             for j in range(len(mat.values[0])):
#                 mat[i][j]/=k
#         return mat

# class CausalAttentionMask(Module):
#     def __init__(self,mat):
#         assert mat.width==mat.height
#         self.mat=mat
#         self.n_embd=math.width
#     def __call__(self,mat):
#         for i in range(self.n_embd):
#             for j in range(i):
#                 mat.values[i][j]=Value(float('-inf'))
#         return mat
#     def parameters(self):
#         return self.mat.parameters()

# class CausalAttentionHead(Module):
#     #TODO: figure out how to fit attention into this framework
#     def __init__(self, n_embd, n_head,use_bias=False):
#         assert n_embd%n_head==0
#         self.n_embd=n_embd
#         self.n_head=n_head
#         self.head_size = n_embd // n_head
#         # self.wq=TrainableMatrix(n_embd,n_embd)
#         # self.wk=TrainableMatrix(n_embd,n_embd)
#         # self.wv=TrainableMatrix(n_embd,n_embd)
#         # self.q=TrainableMatrix(n_embd,n_embd)
#         # self.k=TrainableMatrix(n_embd,n_embd)
#         # self.v=TrainableMatrix(n_embd,n_embd)

#         self.wq = [Linear(n_embd, self.head_size, nonlin=False, use_bias=use_bias) for _ in range(n_head)]
#         self.wk = [Linear(n_embd, self.head_size, nonlin=False, use_bias=use_bias) for _ in range(n_head)]
#         self.wv = [Linear(n_embd, self.head_size, nonlin=False, use_bias=use_bias) for _ in range(n_head)]

#         self.proj = Linear(n_embd, n_embd, nonlin=False, use_bias=use_bias)

#     def __call__(self, x):
#         n, embd_size = x.size() # this works for python lists, but not for values
#         assert embd_size == self.n_embd
#         #we need to project our input x to matrices wiq, wik, wiv (https://arxiv.org/pdf/1706.03762.pdf, page 5)
#         # self.wq=self.wq(x)
#         # self.wk=self.wk(x)
#         # self.wv=self.wv(x)
#         # q_proj=MatMul(q,wq)
#         # k_proj=MatMul(k,wk)
#         # v_proj=MatMul(v,wv)
#         self.q = [self.wq[i](x) for i in range(self.n_head)] # (nh, n, hs)
#         self.k = [self.wk[i](x) for i in range(self.n_head)] # (nh, n, hs)
#         self.v = [self.wv[i](x) for i in range(self.n_head)] # (nh, n, hs)

#         z = [MatMul(Scale(MatMul(self.q[i], self.k[i], transpose_b=True), math.sqrt(self.head_size)), self.v[i]) for i in range(self.n_head)]
#         # concatenate z
#         # mask z
#         # softmax on z

#         return self.proj(z)

#     def parameters(self):
#         return self.wq.parameters()+self.wk.parameters()+self.wv.parameters()
#         + self.q.parameters()+self.k.parameters()+self.v.parameters()

#     def __repr__(self):
#         pass

# class Concat(Module):
#     #concatenate multiple matrices into one
#     def __init__():
#         pass
#     def __call__():
#         pass
    
# class MultiHeadAttention(Module):
#     #TODO
#     def __init__(self, n_embd,n_head,use_bias=False):
#         assert n_embd%n_head==0
#         self.n_embd=n_embd
#         self.n_head=n_head
#         self.single_head_dim=n_embd//n_head
#         self.attention_layers=[CausalAttentionHead(self.single_head_dim,use_bias=use_bias) for _ in range(n_head)]
#         self.wo=TrainableMatrix(n_embd,self.single_head_dim)
#     def __call__(self,x):
#         attn_outs=[self.attention_layers[i](x) for i in range(self.n_head)]
#         attn_concat=Concat(attn_outs)
#         out=MatMul(attn_concat,self.wo)
#     def parameters(self):
#         return [att.parameters() for att in self.attention_layers]+self.wo.parameters()
#     def __repr__(self):
#         pass


# class LayerNorm(Module):
#     #TODO
#     pass

# class Embedding(Module):
#     def __init__(self,vocab_size,n_embd):
#         self.vocab_size=vocab_size
#         self.n_embd=n_embd
#         #we embed each one-dimensional values to n_embd dimensions, and those are our trainable weights
#         self.embeddings_list=TrainableMatrix(n_embd,vocab_size)
#     def __call__(self,x):
#         assert(0<=x and x<vocab_size)
#         return self.embeddings_list.values[x]
#     def parameters(self):
#         return self.embeddings_list.parameters()
#     def __repr__(self):
#         return f"Embedding of [{"|".join(emb) for emb in embeddings_list}]"

# class GPT_MLP(Module):
#     def __init__(self, n_embd,use_bias=False):
#         self.c_fc = Linear(n_embd, 4 * n_embd, use_bias=use_bias,nonlin=True)
#         self.c_proj = Linear(4 * n_embd, n_embd, use_bias=use_bias,nonlin=False)

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.c_proj(x)
#         return x
#     def parameters(self):
#         return self.c_fc.parameters()+self.c_proj.parameters()
#     def __repr__(self):
#         pass
         
# class TransformerBlock(Module):
#     def __init__(self, n_embd, m_head, use_bias=False):
#         self.n_embd=n_embd
#         self.ln_1 = LayerNorm(n_embd, use_bias=use_bias)
#         self.attn = MultiHeadAttention(n_embd,n_head)
#         self.ln_2 = LayerNorm(n_embd, use_bias=use_bias)
#         self.mlp = GPT_MLP(n_embd, use_bias=use_bias)

#     def __call__(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
#     def parameters(self):
#         return self.ln_1.parameters()+self.attn.parameters()+self.ln_2.parameters()+self.mlp.parameters()

#     def __repr__(self):
#         pass

# class GPT(Module):
#     def __init__(self, n_layer, vocab_size, n_embd, n_head, use_bias=False):
#         self.token_embedding=Embedding(vocab_size,n_embd)
#         #not using positional embeddings! (https://twitter.com/a_kazemnejad/status/1664277559968927744)
#         self.transformer_blocks=[TransformerBlock(n_embd,n_head,use_bias) for _ in range(n_layer)]
#         self.layernorm_final=LayerNorm(n_embd)
#     def __call__(self,x):
#         pass
#     def parameters():
#         pass
