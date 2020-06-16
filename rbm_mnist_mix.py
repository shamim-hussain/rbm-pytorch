# -*- coding: utf-8 -*-
import numpy as np
import torch as tc
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test, Y_test)=mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

class RBM():
    def __init__(self, vu, hu, k=1):
        self.W = tc.zeros([vu, hu], requires_grad=False).cuda()
        self.bv = tc.zeros([vu], requires_grad=False).cuda()
        self.bh = tc.zeros([hu], requires_grad=False).cuda()
        self.k = k
        
        tc.nn.init.xavier_uniform_(self.W)
        
        self.opt = tc.optim.Adam([self.W,self.bv,self.bh], lr=1e-4)
        self.vp = None
        self.hp = None
    
    def calc_H(self, v, h):
        return ((v@self.W)*h).sum(-1)+(self.bv*v).sum(-1)+(self.bh*h).sum(-1)
        
    def h_given_v(self,v):
        ph_v = tc.sigmoid(v@self.W+self.bh)
        return ph_v
        
    def v_given_h(self,h):
        pv_h = tc.sigmoid(h@self.W.t()+self.bv)
        return pv_h
    
    def sample_k(self,k,v_init, h_init=None):
        v=v_init
        if h_init is None:
            h=self.h_given_v(v)
        else:
            h=h_init
        
        for s in range(k):
            h= tc.bernoulli(h)
            v=self.v_given_h(h)
            h=self.h_given_v(v)
        return (v,h)
    
    def train_on_batch(self, inp):
        vd=tc.from_numpy(inp).float().cuda()
        hd = self.h_given_v(vd)
        
        if self.vp is None:
            self.vp = tc.rand_like(vd).cuda()
            vm,hm = self.vp, self.hp = self.sample_k(self.k,self.vp)
        else:
            vm,hm = self.vp, self.hp = self.sample_k(self.k,self.vp, self.hp)
        
        self.W.grad = -(vd[:,:,None]@hd[:,None,:]-vm[:,:,None]@hm[:,None,:]).mean(dim=0)
        self.bh.grad = -(hd-hm).mean(dim=0)
        self.bv.grad = -(vd-vm).mean(dim=0)
        
        vm,hm = self.sample_k(self.k,vd,hd)
        self.W.grad += -(vd[:,:,None]@hd[:,None,:]-vm[:,:,None]@hm[:,None,:]).mean(dim=0)
        self.bh.grad += -(hd-hm).mean(dim=0)
        self.bv.grad += -(vd-vm).mean(dim=0)
        
        self.W.grad /= 2
        self.bh.grad /= 2
        self.bv.grad /= 2
        
        self.opt.step()
        return ((self.v_given_h(hd)-vd)**2).mean()#loss.item(), out.data.numpy()
    

num_epochs = 100
bsize=64
ind = np.arange(len(X_train))

rloss = 0.
racc = 0.
lmda = .99

model = RBM(28*28,320, k=10)


for e in range(num_epochs):
    np.random.shuffle(ind)
    for i in range(0,len(X_train)-len(X_train)%bsize, bsize):
        inds = ind[i:min(len(X_train), i+bsize)]
        #targ = Y_train[inds]
        l=model.train_on_batch(X_train[inds])
        rloss = lmda*rloss+(1-lmda)*l
        #racc = lmda*racc +(1-lmda)*accuracy_score(targ, o.argmax(-1))
        
        if not i%1000:
            print('Epoch: {} ; Step: {} || Loss= {} ; Acc= {}'
                  .format(e,i, rloss, racc))


#X=tc.tensor(np.zeros([32,28*28],dtype='float32')).cuda()
X=tc.rand([32,28*28]).cuda()
#X=tc.tensor(X_train[:32]).cuda()
V,_=model.sample_k(50000, X)
v=V.cpu().numpy().reshape([-1,28,28])
plt.figure()
for k in range(4):
    for l in range(8):
        plt.subplot(4,8,k*8+l+1)
        plt.imshow(v[k*8+l])
        
np.savez('rbm_320_mix.npz',W=model.W.cpu().numpy(),bh=model.bh.cpu().numpy(),
         bv=model.bv.cpu().numpy())