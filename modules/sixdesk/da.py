import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
import sqlite3
from scipy import integrate


def davst_function(turn,d,b,kappa):
    '''Returns the da versus turn for a given set of d,b,kappa.'''
    return d + (b/(np.log10(turn)**kappa))

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
            self.dynap['da'] = self.dynap['da']*((self.emit)/(self._newemit))**0.5
            
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
    
    def __init__(self,filename,emit):
        self.filename = filename
        self.data     = self._get_data(filename)
        self._get_emit(filename)
        if emit is not None:
            self._adjust_emittance(self.emit, emit)
        self.revf = 11245.5                                 # LHC revolution frequency


    def _get_emit(self, filename):
        '''Get the emittance used in the SixDesk study'''
        conn           = sqlite3.connect(filename)
        c              = conn.cursor() 
        c.execute("select value from env WHERE keyname='emit';")
        data           = np.array(c.fetchall())
        self.emit      = data[0][0]
        
    def _get_data(self,filename):
        '''Get the da_vst data from the SixDeskDB'''
        conn = sqlite3.connect(filename)
        daf  = pd.read_sql_query("select seed,dawsimp,dawsimperr,nturn from da_vst;", conn)    
        return daf

    def _adjust_emittance(self, old, new):
        '''Adjust to the emittance selected by the user.'''
        self.data['dawsimp']    = self.data['dawsimp']*np.sqrt((old)/(new))
        self.data['dawsimperr'] = self.data['dawsimperr']*np.sqrt((old)/(new))        
    
    def dafunction(self,x,d,b,k):
        '''depreciated'''
        return d + (b/(np.log10(x)**k))
    
    def _dafunction(self,kappa):
        '''Define the function to be used for the fit. Its better to fit with two 
        free parameters, due to unstable results when using three parameters. 
        Therefore we fix kappa for the fitting.'''
        def dafunc(x, d, b):
            return d + (b/(np.log10(x)**kappa))
        return dafunc

    def davst_function(self,turn,d,b,kappa):
        '''Returns the da versus turn for a given set of d,b,kappa.'''
        return d + (b/(np.log10(turn)**kappa))
    
    def _da_after_time(self,kappa,d,b,minutes):
        '''Extract the dynamic aperture for a given time in minutes for a known 
        function defined by kappa,d and b.'''
        turns = self.revf*60*minutes
        return d + (b/(np.log10(turns)**kappa))
    
    def _chi_square(self,y_obs,y_exp):
        '''Return the chi square of a fitted function.
        
        Parameters
        ----------
        y_obs : list of float or int, 
            observed values
        y_exp : list float or int,
            expected values by the model

        Returns
        ----------
        result : float
        '''

        # convert to numpy array
        y_obs = np.array(y_obs)
        y_exp = np.array(y_exp)

        return ((y_obs - y_exp)**2/(y_exp)).sum()

    def _generate_function(self,kappa=None):
        '''Generate the da function depending on whether kappa is fixed or not'''
        if kappa is None:
            def dafunc(x,d,b,k):
                return d + (b/(np.log10(x)**k))
        else:
            def dafunc(x,d,b):
                return d + (b/(np.log10(x)**kappa))
        return dafunc

    def _fit_davst_single_seed(self,seed,kappa=None):
        '''Fits the DA for a single seed with or without defined kappa.

        Parameters
        ----------
        seed : the seed for which the data should be fitted
        kappa : the kappa which should be used, default value ='None', 
            in this case kappa will be fitting variable. kappa can be None, int/float or 
            iterable of int/float. If a list is given, the result will correspond to the 
            kappa with the smallest chi squared.
        '''
        
        # select data for the desired seed
        data             = self.data[self.data.seed==seed]      
        
        # get data for horizontal axis [turn] and vertical axis [DA] and DA error [dawsimperr]
        xdata            = data['nturn'][1:]          
        ydata            = data['dawsimp'][1:]
        yerr             = data['dawsimperr'][1:]
        
        # create list to iterate over
        try:
            kappas = [x for x in kappa]
        except TypeError:
            kappas = [kappa]

        # initialize the minimum chi square
        minimum_chisq = [np.infty]

        # loop over all possible kappas
        for kappa in kappas:
            dafunc = self._generate_function(kappa=kappa)

            # perform the fit                               
            db, pcov         = curve_fit(dafunc, xdata, ydata, sigma=yerr)  

            if kappa is None:           
                d,b,k        = db
            else:
                d,b          = db
                k            = kappa

            # calculate the chi square of the fit

            y_exp = self.davst_function(xdata,d,b,k)
            y_obs = ydata
            chisq = self._chi_square(y_obs, y_exp)

            if chisq<minimum_chisq:
                minimum_chisq  = chisq
                best_params    = (d,b,k,chisq), pcov
        
        return best_params

    def fit_davst(self,seeds,kappa=None):
        '''Fits the DA for a single seed with or without defined kappa.

        Parameters
        ----------
        seeds : the seeds for which the data should be fitted
        kappa : the kappa which should be used, default value ='None', 
            in this case kappa will be fitting variable. kappa can be None, int/float or 
            iterable of int/float. If a list is given, the result will correspond to the 
            kappa with the smallest chi squared.
        '''

        # create list to iterate over
        try:
            seeds = [x for x in seeds]
        except TypeError:
            seeds = [seeds]

        output = []
        for seed in seeds:
            try:
                params, pcov = self._fit_davst_single_seed(seed,kappa=kappa)
                pcov         = [pcov[i,i]**0.5 for i in range(len(pcov))] # take the diagonal elements
                output.append([seed]+list(params)+pcov)
            except TypeError:
                continue

        cols = ['seed','d','b','k','chisq']
        for i in range(len(pcov)):
            cols.append('std_{0}'.format(i))

        output = pd.DataFrame(output,columns=cols)
        self.fit = output
        return output

    def extrapolate_fit(self, time):
        '''Adds a column to self.fit with the extrapolated DA at the indicated time.'''
        turns = self.revf*60*time
        d,b,k = self.fit['d'].values,self.fit['b'].values,self.fit['k'].values

        self.fit['ex_da_{}min'.format(time)] = self.davst_function(turns,d,b,k)        

    
    def _fit_da_given_kappa(self,kappa,seed,debug=False):
        '''Fits the DA for a single seed with a defined kappa.'''
        
        # select data for the desired seed
        data             = self.data[self.data.seed==seed]      
        
        # get data for horizontal axis [turn] and vertical axis [DA] and DA error [dawsimperr]
        xdata            = data['nturn'][1:]          
        ydata            = data['dawsimp'][1:]
        yerr             = data['dawsimperr'][1:]
        
        # generate the da function for the given kappa 
        dafunc           = self._dafunction(kappa)     

        # perform the fit                               
        db, pcov         = curve_fit(dafunc, xdata, ydata, sigma=yerr)               
        d,b              = db

        # calculate the chi square of the fit

        y_exp = self.dafunction(xdata,d,b,kappa)
        y_obs = ydata
        chisq = self._chi_square(y_obs, y_exp)
        
        return (d,b,kappa,chisq), pcov
        
    def fit_single_seed(self,k_start,k_stop,k_step,seed):
        '''Fit for a single seed with a range of kappa and identify the function with the 
        smallest chi^2. 

        Parameters
        ----------

        k_start : starting point for evenly distributed set of kappa values to be studied
        k_stop : end point for evenly distributed set of kappa values to be studied
        k_step : step size for evenly distributed set of kappa values to be studied
        seed : seed to be studied

        Examples     
        ----------   
        >>> self._fit_single_seed(-5,5,0.1,3)
        fit the DA with kappa in the range between -5 and 5 in steps of 0.1 for seed 3.
        '''

        # fit for the different values of kappa
        # take the kappa with the smallest squared error
        out = []
        for kappa in np.arange(k_start, k_stop, k_step):
            (d,b,kappa,chisq),pcov = self._fit_da_given_kappa(kappa,seed)
            out.append([d, b, kappa, chisq, pcov[0,0], pcov[1,1]])
        out = np.array(out)

        d, b, kappa, chisq, pcovd, pcovb =  out[np.argmin(out[:,3])]
        return d, b, kappa, chisq, pcovd, pcovb


    def _dafunction_error(self,function,xdata,ydata,d,b):
        '''DEPRECEATED: Return the squared error of the function'''
        return (((function(xdata,d,b)-ydata)**2)/(ydata)).sum()

    def _fit_single_seed_kappa(self,kappa,seed,debug=False):
        '''DEPRECIATED: Fit DA for a single seed with a defined kappa'''
        
        # select data for the desired seed
        data             = self.data[self.data.seed==seed]   
        data   = data[(data['nturn'].diff()>2000) | (data['dawsimp'].diff()<-0.4)]
        
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
        
        return d,b,kappa,pcov[0,0],pcov[1,1],sqerr
    
    def fit_single_seed_old(self,ki,kf,dk,seed,minutes=30,debug=False):
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
        # d,b,k,_         = out[np.argmin(out[:,3])]                        
        d,b,k,dd,db,sqerr = out[np.argmin(out[:,3])]                        
        
        # calculate the extrapolated DA for the given time
        extrapolated_da = self._da_after_time(k,d,b,minutes)
        
        return seed,d,b,k,dd,db,sqerr,extrapolated_da


    
    def _fit_multiple_seeds(self,ki,kf,kd,seeds,minutes=30,verbose=False):
        '''Fit the DA for multiple seeds'''
        out = []
        for s in seeds:
            if verbose:
                print('\rFitting seed {0}'.format(s),)
            try:
                out.append(self.fit_single_seed_old(ki,kf,kd,s,minutes=minutes))
            except TypeError:
                if verbose:
                    print(' ')
                    print('Fitting error for seed {0}'.format(s))
                continue
#         out = np.array(out)
        out = pd.DataFrame(out,columns=['seed','d','b','k','ud','ub','sqerr','exda'])
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


                


class beamloss:
    def __init__(self):
        self.load_loss_da()
        pass
    def load_loss_da(self):
        '''Load the losses vs da into a dictionary'''
        lossda = {}
        for da in np.arange(0.1,20.1,0.1):
            da = round(da,1)
            basedir = '/afs/cern.ch/work/p/phermes/public/loss_vs_da'
            ldf = pd.read_csv("{0}/loss_da_{1:04.1f}.dat".format(basedir, da),names=['loss'])    
            lossda['{0:04.1f}'.format(da)] = np.array(ldf)
        self.lossda = lossda      

    def _get_loss_from_single_da(self,da):
        da = round(da,1)
        return self.lossda['{0:04.1f}'.format(da)]

    def get_loss_from_da_series(self,da_series):
        output = []
        for da in da_series:
            output.append(self._get_loss_from_single_da(da))
        output = np.array(output).flatten()
        return output

    def loss_from_fit_params(self, fit_params, realizations=1, turns=None, time=None):

        if turns is None and time is None:
            raise ValueError('Calculation of loss from fit parameters requires to indicate either the revolution time or the number of turns to take into account')
            return
        elif turns is None and time is not None:
            turns = 11245.5*time

        if realizations==1:
            nfit_params  = fit_params
        else:
            nfit_params  = get_fitting_params_distribution(fit_params,realizations)

        da_series = get_da_from_fit_output(nfit_params, turns)
        loss      = self.get_loss_from_da_series(da_series)
        return loss


def get_fitting_params_distribution(df,size):
    '''Returns the distribution of fitting parameters using the fitting errors'''
    output = []
    for row in df[['d','b','k','std_0','std_1']].iterrows():
        row     = row[1]
        # get the fit parameters
        d_dist  = np.random.normal(row['d'],scale=row['std_0'],size=size)
        b_dist  = np.random.normal(row['b'],scale=row['std_1'],size=size)
        k       = row['k']
        for i in range(len(d_dist)):
            output.append([d_dist[i], b_dist[i], k])
    output = pd.DataFrame(output,columns=['d','b','k'])
    return output   

def get_da_from_fit_output(fit_output,turns):
    '''Calculates the DA for a given number of turns from the fit output'''
    output = []
    for params in fit_output.iterrows():
        params  = params[1]
        k, b, d = params['k'], params['b'], params['d']
        output.append(davst_function(turns, d,b,k))
    return pd.Series(output)

def get_extrapolated_da(fit_params, turns, realizations=1):
    '''Calculate the extrapolated da '''
    if realizations==1:
        return get_da_from_fit_output(fit_params,turns)
    fit_params = get_fitting_params_distribution(fit_params,realizations)    
    da_series  = get_da_from_fit_output(fit_params, turns)
    return da_series 