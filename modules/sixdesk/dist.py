import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from scipy import integrate
from scipy.special import erf, erfc

import pandas as pd
from pandas import HDFStore, DataFrame

from sixdesk.da import *


def gaussian(x,mux,sigx):
    '''Simple 1D Gaussian'''
    return (1./(sigx*np.sqrt(2*np.pi))*np.exp(-(0.5)*((x-mux)/sigx)**2))

def integrate_gauss(mux,sigx,boundaries,moment=0):
    def g(x):
        return gaussian(x,mux,sigx)*x**moment
    
    return integrate.nquad(g,boundaries)

def get_loss_from_da_series(pdseries, path=None,which='mean'):
    return pdseries.apply(get_loss_from_da, path=None, which=which)

def get_loss_from_da(da,path=None,which='mean'):
    '''Returns the minimum,maximum and mean expectable loss for a given DA'''

    # by default use the distributions.h5 file in the module directory
    if path is None:
        moduledir = os.path.dirname(os.path.realpath(__file__))
        path      = moduledir+'/distributions.h5'

    hdf = pd.HDFStore(path)
    physical_dgauss_params = hdf['physical_dgauss_params']
    
    ii=0
    loss = []
    for _, line in physical_dgauss_params.iterrows():
        ii+=1
        _a1, _sig2 = line['a1'], line['sig2']
        
        # initialize the gaussian
        dg = dgauss(a1=_a1,sig1=1,sig2=_sig2)
        
        # print("Checking distribution number {0}".format(ii),end='\r',flush=True)
        # append the relevant quantities
        # loss.append([_a1,_sig2,dg.integrate([[da,np.inf]])])
        loss.append([_a1,_sig2,0.5*dg.tailcontent(da)])
        
    loss = pd.DataFrame(loss,columns=['a1','sig2','loss'])

    if which=='min':
        return loss['loss'].min()
    elif which=='max':
        return loss['loss'].max()
    elif which=='mean':
        return loss['loss'].mean()
    elif which=='all':
        return loss
    elif which=='summary':
        return loss['loss'].min(), loss['loss'].max() , loss['loss'].mean()      



class dgauss:
    '''Class to handle a double Gaussian distribution'''
    def __init__(self,a1,sig1,sig2,a2=None,mux1=0,mux2=0):
        if a2 is None:
            a2 = 1-a1
        
        self.a1, self.a2, self.sig1, self.sig2, self.mux1, self.mux2 = a1,a2,sig1,sig2,mux1,mux2
        self.std = self.integrate([[-np.inf,np.inf]],moment=2)**0.5
        
    def density(self,x):
        '''Returns the probability density function value at the position x for the double Gaussian'''
        return (self.a1*gaussian(x,self.mux1,self.sig1)+
                self.a2*gaussian(x,self.mux2,self.sig2))
    
    def integrate(self,boundaries,moment=0):
        '''Integrate the double Gaussian in the specified limits'''
        def g(x):
            return self.density(x)*x**moment
        return integrate.nquad(g,boundaries)[0]
    
    def losses_above_nsig(self,nsig):
        '''Return the tail content function from N sigma to infinity'''
        return 2*self.integrate([[nsig*self.std,np.inf]])

    def losses_above_nsig1(self,nsig):
        '''Return the tail content function from N sigma1 to infinity'''
        return 2*self.integrate([[nsig*self.sig1,np.inf]])
    
    def tailcontent(self,S):
        return self.a1*(1+erf(-S/(np.sqrt(2)*self.sig1))) + self.a2*(1+erf(-S/(np.sqrt(2)*self.sig2)))    
