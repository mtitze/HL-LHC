import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
import sqlite3
from scipy import integrate


class davsang:
    '''Tools to load the dynamic aperture vs angle from a SixDeskDB'''
    
    def __init__(self,studyname,directory='',emit=None,verbose=False):
        # bind some variables and initialize the sqlite3 connection
        self.study     = studyname
        self.directory = directory
        conn           = sqlite3.connect(directory+studyname+'.db')
        self._c        = conn.cursor()
        self.verbose   = verbose
        
        # load basic quantities from sixdb
        self._get_emit()
        self._get_trackdir()
        self.get_max_da()
        self._newemit  = emit
        self.get_DA()

    def _get_emit(self):
        '''Get the emittance used in the SixDesk study'''
        self._c.execute("select value from env WHERE keyname='emit';")
        data           = np.array(self._c.fetchall())
        self.emit      = data[0][0]

    def _get_trackdir(self):
        '''Get the tracking directory for the study'''
        self._c.execute("select value from env WHERE keyname='trackdir';")
        data           = np.array(self._c.fetchall())
        self.trackdir  = data[0][0]
        
    def get_max_da(self):
        '''Get the maximum da used for the scan in SixDesk'''
        self._c.execute("select value from env WHERE keyname='ns2l';")
        data           = np.array(self._c.fetchall())
        self.max_da    = data[0][0]
        
    def get_DA(self):
        '''Loads the DA from the sixdb database'''
        self._c.execute('SELECT seed,angle,alost1 FROM da_post;')
        data              = np.array(self._c.fetchall())
        data              = pd.DataFrame(data,columns=['seed','angle','da'])
        
        # bind the object self.dynap to data
        # make negative values positive and replace zeros with the max da used in the simulation
        self.dynap        = data
        self.dynap['da']  = self.dynap['da'].abs()    
        self.dynap['da']  = self.dynap['da'].replace(to_replace=0.0,value=self.max_da)
        
        # if a different emittance is required, adjust it here
        # if required move old emittance values to new column
        if self._newemit is not None:
            self.dynap.assign(da_oldemit=self.dynap['da'])             
            self.dynap['da'] = self.dynap['da']*((self.emit)/(self._newemit))
            
        # get the summary of the DA 
        self.get_DA_summary()
            
        # finally find seeds with potentially bad results
        self._find_bad_seeds()
        
        
    def get_DA_summary(self):
        '''Summarize the DA for all seeds into the most relevant quantities'''
        _summarydata = []
        for angle in self.dynap['angle'].unique():
            minDA = self.dynap[self.dynap.angle==angle].da.min()
            maxDA = self.dynap[self.dynap.angle==angle].da.max()
            avgDA = self.dynap[self.dynap.angle==angle].da.mean()
            _summarydata.append([angle,minDA,maxDA,avgDA])
#         self.dasum = np.array(summarydata)
        self.dasum = pd.DataFrame(_summarydata,columns=['angle','minda','maxda','avgda'])
        

    def plotDA(self,axis=None,fmt='o-',label=None, capsize=3):
        if not label:
            label = self.directory
        if axis:
            axis.errorbar(self.dasum.angle,self.dasum.avgda,
                         yerr=[(self.dasum.avgda-self.dasum.minda),(self.dasum.maxda-self.dasum.avgda)],
                          fmt=fmt, label=label, capsize=capsize)
            axis.set_xlabel('Angle [deg]')
            axis.set_ylabel(r'DA [$\sigma$]')
        else:
            plt.errorbar(self.dasum.angle,self.dasum.avgda, 
                         yerr=[(self.dasum.avgda-self.dasum.minda),(self.dasum.maxda-self.dasum.avgda)],
                         fmt=fmt, label=label, capsize=capsize)
            plt.xlabel('Angle [deg]')
            plt.ylabel(r'DA [$\sigma$]')

            
    def _find_bad_seeds(self):
        '''Identify seeds with unphysical results. Typically these deliver results close to the 
        transition points with DA very close to an integer.'''
        self.badseeds = self.dynap[(self.dynap.da-self.dynap.da.round()).abs()<1e-4]
        if len(self.badseeds)>0 and self.verbose:
            print('Found %i bad jobs for study %s' % (len(self.badseeds), self.study))
            print('All bad seeds are saved in self.badseeds')




class davst:    
    '''Tools to handle dynamic aperture vs. turn from SixDeskDB'''
    
    def __init__(self,filename):
        self.data = self._get_data(filename)
        self.revf = 11245.5                                 # LHC revolution frequency
        
    def _get_data(self,filename):
        '''Get the da_vst data from the SixDeskDB'''
        conn = sqlite3.connect(filename)
        daf  = pd.read_sql_query("select seed,dawsimp,dawsimperr,nturn from da_vst;", conn)    
        return daf
    
    def dafunction(self,x,d,b,k):
        return d + (b/(np.log10(x)**k))
    
    def _dafunction(self,kappa):
        '''Define the function to be used for the fit. Its better to fit with two 
        free parameters, due to unstable results when using three parameters. 
        Therefore we fix kappa for the fitting.'''
        def dafunc(x, d, b):
            return d + (b/(np.log10(x)**kappa))
        return dafunc
    
    def _da_after_time(self,kappa,d,b,minutes):
        '''Extract the dynamic aperture for a given time in minutes for a known 
        function defined by kappa,d and b.'''
        turns = self.revf*60*minutes
        return d + (b/(np.log10(turns)**kappa))
    
    def _dafunction_error(self,function,xdata,ydata,d,b):
        '''Return the squared error of the function'''
        return ((function(xdata,d,b)-ydata)**2).sum()
    
    def _fit_single_seed_kappa(self,kappa,seed,debug=False):
        '''Fit DA for a single seed with a defined kappa'''
        
        # select data for the desired seed
        data             = self.data[self.data.seed==seed]      
        
        # get data for horizontal axis [turn] and vertical axis [DA] and DA error [dawsimperr]
        xdata            = data['nturn'][1:]          
        ydata            = data['dawsimp'][1:]
        yerr             = data['dawsimperr'][1:]
        
        # generate the da function for the given kappa and perform the fit
        dafunc           = self._dafunction(kappa)                                   
        db, pcov         = curve_fit(dafunc, xdata, ydata,sigma=yerr)               
        
        # calculate the squared error
        d,b              = db
        sqerr            = self._dafunction_error(dafunc,xdata,ydata,d,b)            
        
        return d,b,kappa,sqerr
        
    def fit_single_seed(self,ki,kf,dk,seed,minutes=30,debug=False):
        '''Fit for a single seed with a range of kappa and identify the function with the least squared error.
        
        Example usage: 
            self._fit_single_seed(-5,5,0.1,3, minutes=60)
        will fit the DA with kappa in the range between -5 and 5 in steps of 0.1 for seed 3 and return
        the parameters with the extrapolated DA.
        
        returns seed, d, b, k, extrapolated_da
        '''
        # fit for the different values of kappa
        # take the kappa with the smallest squared error
        out = []
        for kk in np.arange(ki,kf,dk):
            out.append(self._fit_single_seed_kappa(kk,seed))
        out = np.array(out)
        d,b,k,_         = out[np.argmin(out[:,3])]                        
        
        # calculate the extrapolated DA for the given time
        extrapolated_da = self._da_after_time(k,d,b,minutes)
        
        return seed,d,b,k,extrapolated_da
    
    
    def _fit_multiple_seeds(self,ki,kf,kd,seeds,minutes=30,verbose=False):
        '''Fit the DA for multiple seeds'''
        out = []
        for s in seeds:
            if verbose:
                print('\rFitting seed {0}'.format(s),)
            try:
                out.append(self.fit_single_seed(ki,kf,kd,s,minutes=minutes))
            except TypeError:
                if verbose:
                    print(' ')
                    print('Fitting error for seed {0}'.format(s))
                continue
#         out = np.array(out)
        out = pd.DataFrame(out,columns=['seed','d','b','k','exda'])
        return out
    
        
    def fitda(self,ki,kf,kd,steps=1,seeds=None,minutes=30,verbose=False,**kwargs):
        '''Fit the DA vs. turns for a given number of seeds.
        Example Usage:
            self.fitda(-5,5,0.1,steps=2,seed=[1,60],minutes=30)
        '''
        
        # if no seeds are specified, use the standard set between 1 and 60
        if seeds is None:
            seeds = range(1,61)

        # fit for all seeds in the given range
        df = self._fit_multiple_seeds(ki,kf,kd,seeds,minutes=minutes,verbose=verbose)
        
        # return the result if only one step is requested
        if steps==1:
            return df
        
        # iterate over the steps if multiple steps are requested
        for _ in range(steps-1):
            try:
                ki,kf,kd = df.k.mean()-kd, df.k.mean()+kd, kd/10.
            except NameError:
                pass
            df = self._fit_multiple_seeds(ki,kf,kd,seeds,minutes=minutes,verbose=verbose)
        return df

    def plot_simulated_davst(self,seed,axis=None,label='Fit'):
        '''Plot the simulated data'''
        data = self.data[self.data.seed==seed]
        if axis is None:
            plt.errorbar(data.nturn,data.dawsimp,yerr=data.dawsimperr, fmt='o',label=label)
            plt.xlabel('Turn')
            plt.ylabel(r'DA [$\sigma$]')
        else:
            axis.errorbar(data.nturn,data.dawsimp,yerr=data.dawsimperr, fmt='o',label=label)
            axis.set_xlabel('Turn')
            axis.set_ylabel(r'DA [$\sigma$]')            
