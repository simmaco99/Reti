
import torch
import torch.nn as fun
import numpy as np
import healpy as hp
import pynkowski as mf
import pickle
import time
import matplotlib.pyplot as plt
import os 
import random
import scipy.integrate as integrate
from scipy.special import legendre, binom, factorial2,erfc
from scipy.stats import norm as gs 
import pandas as pd
from pypdf import PdfMerger

## set path_root
if os.getlogin() =='dilillo':
    path_root = '/scratch/dilillo/new'
else:
    path_root = '/Users/simmaco/Desktop/Dottorato'


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def gaussian_act(x):
        return torch.exp(-x**2)

activation_fun= {
    'Identity'  :{  'np_activation'     : lambda u : u,
                    'torch_activation'  : lambda u : u,
                    'norm'              : 1,
                    'kernal'            : lambda u : u},
    
    'Logistic'  :{  'np_activation'     : lambda u : 1./(1+np.exp(-u)),
                    'torch_activation'  : fun.Sigmoid(),                     
                    'norm'              : None,
                    'kernal'            : None},

    'ELU'       :{
                    'np_activation'     : lambda u : (u>0)* u  + (np.exp(u) -1) * (u<=0),
                    'torch_activation'  : fun.ELU(),
                    'norm'              : 1 - np.sqrt(np.exp(1)) *erfc(1/np.sqrt(2)) + 1/2* np.exp(2) * erfc(np.sqrt(2)),
                    'kernal'            : None},
    
    'SiLU'  :{
                    'np_activation'     : lambda u : u/(1+np.exp(-u)),
                    'torch_activation'  : fun.SiLU(),
                    'norm'              : None,
                    'kernal'            : None,},
    
    'GELU'      :{
                    'np_activation'     : lambda u : u * gs.cdf(u),
                    'torch_activation'  : fun.GELU(),
                    'norm'              : 1/3 + np.sqrt(3)/(6*np.pi),
                    'kernal'            : None},      
    
    'ReLu'      :{  'np_activation'     : lambda u : np.max(0,u),
                    'torch_activation'  : fun.ReLU(),
                    'norm'              : 1/2,
                    'kernal'            : lambda u : ((u * (np.pi -np.arccos(u))) + np.sqrt(1-u*u))/np.pi},

    'Step'      :{  'np_activation'     : lambda u : 1/2 * (np.sign(u)+1),
                    'torch_activation'  : lambda u: torch.heaviside(u,torch.tensor([0.])), 
                    'norm'              : 1/2,
                    'kernal'            : lambda u : 1/np.pi * (np.pi -np.arccos(u))},

    'tanh'      :{  'np_activation'     : lambda u : np.tanh(u),
                    'torch_activation'  : fun.Tanh(),
                    'norm'              : None,
                    'kernal'            : None},
    
    'LReLU'     :{
                    'np_activation'     : lambda u : np.max(0,u) + 0.01 * np.min(0,u),
                    'torch_activation'  : fun.LeakyReLU(),
                    'norm'              : 0.50005,
                    'kernal'            : None},
    
    'Gaussian'  :{
                    'np_activation'     : lambda u : np.exp(-u^2),
                    'torch_activation'  : gaussian_act,
                    'norm'              : 1/np.sqrt(5),
                    'kernal'            : None}, 
    
    'minus'     :{  'np_activation'     : lambda u : (2. + u ) /np.sqrt(5),
                    'torch_activation'  : lambda u : u.add(1).multiply(1/np.sqrt(5)), # va cambiata
                    'norm'              : 1,
                    'kernal'            : lambda u : (4.+u)/5.}
}

class Activation:
    """
    The Activation class represents different activation functions used in neural networks.

    Attributes:
        name (str): The name of the activation function.
        activation (function): The activation function implementation.
        activation_np (function): The NumPy implementation of the activation function.
        norm (float): The normalization factor for the activation function.
        kernal (function): The kernel function associated with the activation function.

    Methods:
        __init__(name, p=None):
            Initializes an instance of the Activation class with the specified name and optional parameters.

        kernal_iterative(x, L):
            Computes the iterative kernel function for the activation function.

        lambda_iterative(L):
            Computes the iterative lambda function for the activation function.
    """
    def __init__(self,name,p=None):
        nname = None 
        if name not in activation_fun.keys():
                raise Exception('The parameter name must be in' + str(list(activation_fun.keys())))
        if name == 'ReLu_pow':
            if p == None:
                raise Exception('Insert the power of ReLU function')
            else:
                activation_fun['ReLu_pow']['norm'] =  factorial2(2*p-1)/2
                nname=  str(p)+'-ReLu'

        if nname ==None:
            nname =name
        
        Selected = activation_fun[name]
        
        if Selected['norm'] ==None:
            res,_=integrate.quad(lambda u : Selected['np_activation'](u)**2* np.exp(-u**2/2),-np.inf,np.inf)
            Selected['norm']= res/np.sqrt(2*np.pi)

        self.name =nname
        self.activation= Selected['torch_activation']
        self.activation_np = Selected['np_activation']
        self.norm = Selected['norm']
        self.kernal = Selected['kernal']

    def kernal_iterative(self,x,L):
        if L ==1: 
            return self.kernal(x)
        else:
            return self.kernal_iterative(self.kernal_iterative(x,L-1),1)

    def lambda_iterative(self,L):
        return lambda u: self.kernal_iterative(u,L)    


class NeuralNetwork:
    """
    The NeuralNetwork class facilitates the generation of neural network fields with customizable activation functions
    and parameters.

    Attributes:
        activation_list (str or list): The activation function used in the neural network. Can be either a string
            specifying the activation function name or a list containing the function name and its parameters.
        res (int): The resolution parameter for generating the neural network fields.
        L (list of int): A list of integers representing the number of hidden layers in the neural network.
        n (int): The number of neurons in hidden layers
        normalise (bool): A boolean indicating whether to normalize the generated fields.

    Methods:
        __init__(activation_list, res, L, n, u_min=-5, u_max=5, u_step=250, normalise=True):
            Initializes an instance of the NeuralNetwork class with specified parameters.
        
        generate(n_samples, random_list=None):
            Generates neural network fields based on the specified parameters.
        
        controllo_check():
            Performs a check on the generated fields.
        
        generate_hidden_torch(seed=0, ell_max=None, save=True, L_new=None):
            Generates hidden layers using PyTorch based on the specified parameters.
        
        mean_sperimental(int_ell=None, random_seed=50,generate=False):
            Estimates the mean experimental power spectrum.
        
        dire():
            Returns a list of directories containing generated data.
        
        merge_cl():
            Merges the computed power spectra.
        
        compute_range():
            Computes the range of values for generated fields.
        
        mollview(random):
            Plots a spherical map of generated fields.
        
        derivative(random):
            Plots the derivative of generated fields.
    """
    def __init__(self,activation_list,res,L,n):
        self.res = res
        self.coordinate= None
        self.L = L
    
        if type(activation_list) ==str:
            name  = activation_list
            p = None 
        elif len(activation_list)==2:
            name =activation_list[0]
            p = activation_list[1]
        else:
            raise Exception('activation_list must be a string or a list of lenght 2')
        
        self.activation = Activation(name,p)
        self.n = n
        self.cl=None
        self.mink=None
        self.coordinate = torch.from_numpy(np.transpose(np.array(hp.pix2vec(nside=2**self.res, ipix=np.arange(hp.nside2npix(2**self.res)))))).float()

        self.dir =os.path.join(path_root,'res'+str(self.res), self.activation.name+'_'+str(n))
        self.dir_plot = os.path.join(self.dir,'plot')
        
        check_dir(self.dir)

    def generate(self,n_samples,random_list=None):
        for nn in range(n_samples):
            if random_list==None:
                seed = random.randint(0,9999999)
            else:
                seed = random_list[nn]
            print('Iterazione numero ', nn,  'di', n_samples)
            self.generate_hidden_torch(seed)
    
    def controllo_check(self):
        self.dire()
        
    def generate_hidden_torch(self,seed=0,ell_max= None , save= True,L_new = None):
        ## L_new si usa solo per generare i campi che mancano 
        dir_out = os.path.join(self.dir, str(seed))
        if  not os.path.exists(os.path.join(dir_out)):
            os.makedirs(os.path.join(dir_out))
        
        mom = self.activation.norm
        L_list = self.L
        if not L_new == None:
            L_list=L_new
            
        for en, L in enumerate(L_list):
            start = time.time() 
            print(en,' di ', len(L_list))
            
            n = [self.n]*L
            random.seed(seed)
            W = torch.randn(3, n[0]) # first layer
            z = self.activation.activation(self.coordinate @ W)
            for i in range(L-1):
                W = torch.normal(0, np.sqrt(1 / (n[i]*mom)), size=(n[i],n[i+1]))
                z = self.activation.activation(z @ W)
            W = torch.normal(0, np.sqrt( 1/ (n[L-1]*mom)), size=(n[L-1],))
            field = np.array(z @ W)
            self.seed = seed
            cl = hp.anafast(field,lmax=ell_max)
            print('stop: ',time.time()-start)
            if save: 
                hp.fitsfunc.write_map(os.path.join(self.dir,str(seed),str(L)+'_'+str(self.n)+'.fits'),field,overwrite=True)
                hp.fitsfunc.write_cl(os.path.join(self.dir,str(seed),str(L)+'_cl_'+str(self.n)+'.fits'),cl,overwrite=True)
            else:
                return [field,cl]
    


    def mean_sperimental(self,int_ell=None,random_seed = 50,generate=False):
        if int_ell == None:
            int_ell = [0, 3*(2**self.res)-1]
        self.cl = PowerSpectrum(self.activation,self.L, int_ell)
        self.cl.sperimental(self)
        if generate:
            for i,L in enumerate(self.L) :
                np.random.seed(random_seed)
                field= hp.synfast(self.cl.cl_sper[i],nside=2**self.res)
                hp.write_map(os.path.join(self.dir, 'mean',str(L)+'_'+str(self.n)+'.fits'),field,overwrite=True)

    def dire(self):
        dir_list =  os.listdir(self.dir)
        for el in  ['old','plot','angular','range.csv','.DS_Store','mean']:
            if el in dir_list:
                dir_list.remove(el) 
        return dir_list

    def merge_cl(self):
        dir_list = self.dire()
        path_out = os.path.join(self.dir,'mean')
        check_dir(path_out)
       
        for L in self.L:
            C= []
            for dir in dir_list:
                if os.path.isdir(os.path.join(self.dir,dir)):
                    C.append(hp.read_cl(os.path.join(os.path.join(self.dir,dir,str(L)+'_cl_'+str(self.n)+'.fits'))))
            C_mean =np.mean(C,axis=0)
            hp.write_cl(os.path.join(path_out,str(L)+'_cl_'+str(self.n)+'.fits'),C_mean,overwrite=True)
        return C
    
    def compute_range(self):
        dir_list = self.dire()
        C=[]
        name = []
        for L in self.L:
            CC_min =[]
            CC_max =[]
            for seed in dir_list:
                field = hp.read_map(os.path.join(self.dir,seed,str(L)+'_'+str(self.n)+'.fits'))
                CC_min.append(np.min(field))
                CC_max.append(np.max(field))
            C.append(CC_min)
            C.append(CC_max)
            name.append('min ' + str(L))
            name.append('max ' + str(L))
        data = pd.DataFrame({ ll : C[n] for n,ll in enumerate(name)})
        data.to_csv(os.path.join(self.dir,'range.csv'))

    def mollview(self,random):
        dir_output = self.dir_plot
        check_dir(dir_output)        
        if os.path.isdir(os.path.join(self.dir,str(random))):
            merger = PdfMerger()
            for L in self.L:
                field1 = hp.read_map(os.path.join(os.path.join(self.dir,str(random),str(L)+'_'+str(self.n)+'.fits')))
                hp.mollview(field1,hold=True,title='L = ' + str(L) + '- seed ' + str(random))
                plt.savefig(os.path.join(self.dir,str(random),str(L)+'.pdf'))
                plt.close()
                merger.append(os.path.join(self.dir,str(random),str(L)+'.pdf'))
            
            merger.write(os.path.join(self.dir_plot,str(random)+'.pdf'))
            merger.close()
        else:
            print('The field with seed ', random, 'not exists')

    def derivative(self,random):
        if os.path.isdir(os.path.join(self.dir,str(random))):
            merger = PdfMerger()
            for L in self.L:
                cl = os.path.join(self.dir, str(random), str(L)+'_cl_'+str(self.n)+'.fits')
                ell =np.arange(0,len(cl))
                fat = ell * (ell + 1)
                cl = cl * fat
                field1= hp.synfast(cl,nside=2**self.res)
                hp.mollview(field1,hold=True,title='derivative L = ' + str(L) + '- seed ' + str(random))
                plt.savefig(os.path.join(self.dir,str(random),'derivative_'+str(L)+'.pdf'))
                plt.close()
                merger.append(os.path.join(self.dir,str(random),'derivative_'+str(L)+'.pdf'))
                
            merger.write(os.path.join(self.dir_plot,'derivative'+str(random)+'.pdf'))
            merger.close()

    def all_field(self, function):
        ## function = lambda random : self.fun(random) where fun is implemented
        dir_list = self.dire()
        for dir in dir_list:
            if os.path.isdir(os.path.join(self.dir,dir)):
                function(dir)
        
class PowerSpectrum:
    """
    A class for computing and analyzing the power spectrum of neural network-generated fields or theoretical.

    Attributes:
        activation: The activation function used in the neural network.
        L: List of integers representing the number of layers in the neural network.
        ell_min: Minimum value of ell for computing the power spectrum.
        ell_max: Maximum value of ell for computing the power spectrum.
        ell: Array of ell values between ell_min and ell_max.
        d: Dimensionality of the input space.
        dim_auto: Dimensionality of the angular autospace.

    Methods:
        __init__(activation, L, int_ell=[0, 3*(2**9)-1], d=2):
            Initializes an instance of the PowerSpectrum class with specified parameters.

        autospazio():
            Computes the dimensionality of the angular space based on the specified parameters.

        theoretical():
            Computes the theoretical power spectrum based on the activation function and other parameters.

        momenti(N):
            Computes moments of the power spectrum up to a specified order.

        sperimental(NN):
            Computes the experimental power spectrum based on generated data from the neural network.

        delete_even():
            Deletes even elements from the power spectrum.

        plot_sperimental(forma='.-', ini=None, fine=None, dir=None):
            Plots the experimental power spectrum for visualization.
    """
    
    def __init__(self,activation,L,int_ell=[0,3*(2**9)-1],d=2):
        self.cl_teo = None
        self.cl_sper = None
        self.L = L
        self.activation = activation
        self.distribution_teo = None
        self.distribution_sper = None
        self.momenti_teo=None
        self.momenti_sper=None
        self.ell_min= int_ell[0]
        self.ell_max=int_ell[1]
        self.ell = np.arange(self.ell_min,self.ell_max)
        self.d = d
        self.dim_auto = self.autospazio()
        self.even = False
        self.teo = False
        self.sper= False
    def autospazio(self):
        d = self.d
        ell = self.ell
        ell = np.array(ell)
        if d==2:
            dim_auto =  2*np.array(ell)+1
        else:
            if ell[0]==0:
                el = el[1:]
            else:
                el = ell
            dim_auto = (2*el[1:] + d -1)/el* binom(el + d -2, el-1)
            if ell[0] ==0:
                dim_auto = np.concatenate((1,dim_auto))
        return dim_auto
    
    def theoretical(self):
        for LL in self.L:
            C=[]
            for n in range(self.ell_min,self.ell_max+1,1):
                res,_= integrate.quad(lambda x: self.activation.kernal_iterative(x,LL) *legendre(n)(x),-1,1)
                C.append(res*2*np.pi)
            self.cl_teo= self.cl_teo + [C]
            self.distribution_teo = self.distribution_teo + [C*self.dim_auto/(4*np.pi)]

    def momenti(self,N):
        if type(N)==int:
            N = range(1,N+1)
        self.N = N
        mm=[]
        mm_sper = []
        
        for n in N:
            mm = mm + [ np.sum(np.power(self.ell,n)* self.distribution_teo,axis=1)]
            mm_sper = mm_sper + [ np.sum(np.power(self.ell,n)* self.distribution_sper,axis=1)]
        self.momenti_teo  = list(np.transpose(mm))
        self.momenti_sper = list(np.transpose(mm_sper))

    def sperimental(self,NN):
        C=[]
        for LL in self.L:
     
            file = os.path.join(NN.dir,'mean',str(LL)+ '_cl_'+str(NN.n)+'.fits')
            if os.path.exists(file):
                C = C+ [hp.read_cl(file)]
            else:
                raise Exception('Non esistono le medie')
        CC=[]
        for n,_ in enumerate(self.L):
            CC = CC + [C[n][self.ell_min:self.ell_max]]
        self.cl_sper = CC
        self.distribution_sper= self.cl_sper*self.dim_auto/(4*np.pi)   
      
    def delete_even(self):
        dis = []
        for n,_  in enumerate(self.L):
            if self.teo:
                dis0 = self.distribution_teo[n]
                dis0= dis0[np.arange(start=1,stop=len(dis0),step=2)]
                dis = dis + [ dis0]

            if self.sper:
                dis1 = self.distribution_sper[n]
                dis1= dis1[np.arange(start=1,stop=len(dis1),step=2)]
                dis_s = dis_s + [ dis1]
        if self.sper:
            self.distribution_sper = dis_s
            self.ell =self.ell[np.arange(start=1,stop=len(self.ell),step=2)]
            self.even=True   
        if self.teo:
            self.distribution_teo = dis 
            self.ell =self.ell[np.arange(start=1,stop=len(self.ell),step=2)]
            self.even=True   

    def plot_sperimental(self,forma = '.-',ini=None,fine=None, dir = None):
        plt.figure()
        if ini ==None:
            ini =0
        if fine ==None:
            fine = len(self.ell)
        
        dis = self.distribution_sper
        ell = self.ell        

        for n,LL in enumerate(self.L):
            plt.plot(ell[ini:fine+1],dis[n][ini:fine+1],forma,label='L = ' + str(LL))
            plt.legend()
            plt.xlabel('$\ell$',usetex=True)
            plt.ylabel('Neural Network Spectral Distribution')
            
        if not dir ==None:
           plt.savefig(dir)

    def plot_confronto(self,forma='.-', ini= None, fine = None, dir = None):
        if ini ==None:
            ini =0
        if fine ==None:
            fine = len(self.ell)
        
        ell = self.ell[ini:fine+1]
        for [n,LL] in enumerate(self.L):
            plt.figure()
            plt.plot(ell,self.distribution_sper[n][ini:fine+1],forma,label='Sperimental')
            plt.plot(ell,self.distribution_teo[n][ini:fine+1],forma,label='Theoretical')    
            plt.legend()       
            plt.xlabel('$\ell$',usetex=True)
            plt.ylabel('Neural Network Spectral Distribution')


