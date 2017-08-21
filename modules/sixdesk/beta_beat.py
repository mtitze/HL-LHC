import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def betabeat(df1,df2,quantity='betx'):
    '''Returns the beta-beat from two twiss dataframes'''
    return (df2[quantity]-df1[quantity])/df1[quantity]

def load_twiss(filename,column=['name','s','betx','bety','alfx','alfy']):
    '''Load twiss parameters from a MAD-X tfs file.'''
    return pd.read_csv(filename, skiprows=47, delim_whitespace=True,
                       names=column)

def get_bb_all_seeds(nametemp, ref, plane, seedi=1,seedf=60,verbose=False):
    bb      = pd.Series()
    bb_seed = {}                       # beta-beating for the individual seed

    for seed in range(seedi,seedf+1):
        name  = nametemp.format(seed)
        try:
            twiss = load_twiss(name)
            if verbose:
                print('Loaded Seed {0}'.format(seed))
        except IOError:
            if verbose:
                print('Cant load seed {0}'.format(seed))
            continue

        bb_seed[seed] = betabeat(ref,twiss,quantity=plane)
        
        bb = bb.append(bb_seed[seed])
    return bb, bb_seed
