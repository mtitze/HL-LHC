import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
import sqlite3
from scipy import integrate

import warnings

warnings.simplefilter('always', DeprecationWarning)


def deprecation(message):
    warnings.warn(message, DeprecationWarning, stacklevel=2)


class InsufficientDataError(Exception):
    pass


def emittance_growth_gauss(da):
    '''
	Emittance growth of a Gaussian distribution as a function of the dynamic aperture (DA).

	Parameters
	----------
	da : dynamic aperture in terms of standard deviations of the transverse particle density distribution

	Examples
	----------
	>>> growth = emittance_growth_gauss(12.)

	'''
    return (da ** 2 * np.exp(-da ** 2 / 2)) / (2 * (1 - np.exp(-da ** 2 / 2)))


def davst_function(turn, d, b, kappa):
    '''Returns the da versus turn for a given set of d,b,kappa using the functionality described in
	https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.15.024001'''
    return d + (b / (np.log10(turn) ** kappa))


class davsang:
    '''Tools to load the dynamic aperture vs angle from a SixDeskDB'''

    def __init__(self, studyname, directory='', emit=None, verbose=False):
        # bind some variables and initialize the sqlite3 connection
        self.study = studyname
        self.directory = directory
        conn = sqlite3.connect(directory + studyname + '.db')
        self._c = conn.cursor()
        self.verbose = verbose

        # load basic quantities from sixdb
        self._get_emit()
        self._get_trackdir()
        self.get_max_da()
        self._newemit = emit
        self.get_DA()

    def _get_emit(self):
        '''Get the emittance used in the SixDesk study'''
        self._c.execute("select value from env WHERE keyname='emit';")
        data = np.array(self._c.fetchall())
        self.emit = data[0][0]

    def _get_trackdir(self):
        '''Get the tracking directory for the study'''
        self._c.execute("select value from env WHERE keyname='trackdir';")
        data = np.array(self._c.fetchall())
        self.trackdir = data[0][0]

    def get_max_da(self):
        '''Get the maximum da used for the scan in SixDesk'''
        self._c.execute("select value from env WHERE keyname='ns2l';")
        data = np.array(self._c.fetchall())
        self.max_da = data[0][0]

    def get_DA(self):
        '''Loads the DA from the sixdb database'''
        self._c.execute('SELECT seed,angle,alost1 FROM da_post;')
        data = np.array(self._c.fetchall())
        data = pd.DataFrame(data, columns=['seed', 'angle', 'da'])

        # bind the object self.dynap to data
        # make negative values positive and replace zeros with the max da used in the simulation
        self.dynap = data
        self.dynap['da'] = self.dynap['da'].abs()
        self.dynap['da'] = self.dynap['da'].replace(to_replace=0.0, value=self.max_da)

        # if a different emittance is required, adjust it here
        # if required move old emittance values to new column
        if self._newemit is not None:
            self.dynap.assign(da_oldemit=self.dynap['da'])
            self.dynap['da'] = self.dynap['da'] * ((self.emit) / (self._newemit)) ** 0.5

        # get the summary of the DA
        self.get_DA_summary()

        # finally find seeds with potentially bad results
        self._find_bad_seeds()

    def get_DA_summary(self):
        '''Summarize the DA for all seeds into the most relevant quantities'''
        _summarydata = []
        for angle in self.dynap['angle'].unique():
            minDA = self.dynap[self.dynap.angle == angle].da.min()
            maxDA = self.dynap[self.dynap.angle == angle].da.max()
            avgDA = self.dynap[self.dynap.angle == angle].da.mean()
            _summarydata.append([angle, minDA, maxDA, avgDA])
        #         self.dasum = np.array(summarydata)
        self.dasum = pd.DataFrame(_summarydata, columns=['angle', 'minda', 'maxda', 'avgda'])

    def plotDA(self, axis=None, fmt='o-', label=None, capsize=3):
        if not label:
            label = self.directory
        if axis:
            axis.errorbar(self.dasum.angle, self.dasum.avgda,
                          yerr=[(self.dasum.avgda - self.dasum.minda), (self.dasum.maxda - self.dasum.avgda)],
                          fmt=fmt, label=label, capsize=capsize)
            axis.set_xlabel('Angle [deg]')
            axis.set_ylabel(r'DA [$\sigma$]')
        else:
            plt.errorbar(self.dasum.angle, self.dasum.avgda,
                         yerr=[(self.dasum.avgda - self.dasum.minda), (self.dasum.maxda - self.dasum.avgda)],
                         fmt=fmt, label=label, capsize=capsize)
            plt.xlabel('Angle [deg]')
            plt.ylabel(r'DA [$\sigma$]')

    def _find_bad_seeds(self):
        '''Identify seeds with unphysical results. Typically these deliver results close to the
		transition points with DA very close to an integer.'''
        self.badseeds = self.dynap[(self.dynap.da - self.dynap.da.round()).abs() < 1e-4]
        if len(self.badseeds) > 0 and self.verbose:
            print('Found %i bad jobs for study %s' % (len(self.badseeds), self.study))
            print('All bad seeds are saved in self.badseeds')


class davst:
    '''Tools to handle dynamic aperture vs. turn from SixDeskDB'''

    def __init__(self, filename, emit):
        self.filename = filename
        self.data = self._get_data(filename)  # load da vs turn data from sixdeskdb
        self._get_emit(filename)  # get emittance from sixdeskdb
        if emit is not None:
            self._adjust_emittance(self.emit, emit)  # adjust emittance if necessary
        self.revf = 11245.5                # LHC revolution frequency
        self._min_datapoints = 8           # min. number of data points required for fit
        self.check_monotonicity = False    # check monotonicity
        try:                               # load existing fit parameters
            self._get_fit_params_from_db()
        except:
            pass

    def _get_emit(self, filename):
        '''Get the emittance used in the SixDesk study'''
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        c.execute("select value from env WHERE keyname='emit';")
        data = np.array(c.fetchall())
        self.emit = data[0][0]

    def _get_data(self, filename):
        '''Get the da_vst data from the SixDeskDB'''
        conn = sqlite3.connect(filename)
        query = "select * from da_vst;"
        daf = pd.read_sql_query(query, conn)
        conn.close()
        return daf

    def _get_fit_params_from_db(self):
        '''Read fit parameters and extrapolated da from database'''

        # read fit parameters
        try:
            conn = sqlite3.connect(self.filename)
            daf1 = pd.read_sql_query("SELECT * FROM fit_parameters;", conn)
            conn.close()
            self.fit_params = daf1[:]
        except:
            pass
        # # read extrapolated da
        # try:
        #     conn = sqlite3.connect(self.filename)
        #     daf2 = pd.read_sql_query("SELECT * FROM exda;", conn)
        #     conn.close()
        #     self.extrapolated_da = daf2[:]
        # except:
        #     self.extrapolated_da = daf1[:]

    def _adjust_emittance(self, old, new):

        '''Adjust to the emittance selected by the user.
        Usage self._adjust_emittance(old,new).
        The formula to calculate the DA to the new emittance base on the old one is:

        da_new = da_old * sqrt{emittance_old/emittance_new}
        '''

        self.data['dawsimp'] = self.data['dawsimp'] * np.sqrt((old) / (new))
        self.data['dawsimperr'] = self.data['dawsimperr'] * np.sqrt((old) / (new))
        self.emit = new

    def save_fit_params(self):
        '''Saves the fit parameters in the sixdeskdb database'''
        conn = sqlite3.connect(self.filename)
        self.fit_params.to_sql('fit_parameters', conn, if_exists='replace', index=False)
        conn.close()

    # def save_exda(self):
    #     '''Saves the fit parameters in the sixdeskdb database'''
    #     conn = sqlite3.connect(self.filename)
    #     self.extrapolated_da.to_sql('exda', conn, if_exists='replace', index=False)
    #     conn.close()

    def get_extrapolated_da(self, minutes=0, seconds=0, hours=0, realizations=1, save=False):
        '''Calculate the extrapolated DA and save it to self.extrapolated_da

		Parameters
		----------
		hours : int,
			number of hours, default=0
		minutes : int,
			number of minutes, default=0
		seconds : int,
			number of seconds, default=0
		'''
        turns = self.revf * (seconds + 60 * minutes + 3600 * hours)
        sec = hours * 60. * 60. + minutes * 60. + seconds

        exda = self.fit_params

        # randomly calculate possible realizations of the extrapolated da using the fitting error
        if realizations>1:
            exda = get_fitting_params_distribution(exda,realizations)

        exda_key = 'exda_{0}_sec'.format(int(sec))

        self.extrapolated_da = exda
        self.extrapolated_da[exda_key] = davst_function(turns, exda['d'], exda['b'], exda['k'])

        # adjust emittance
        if self.emit != self.extrapolated_da['emit'][0]:
            emittance_factor = np.sqrt(self.extrapolated_da['emit'][0]/self.emit)
            self.extrapolated_da['emit']   = np.ones(len(self.extrapolated_da))*self.emit
            self.extrapolated_da[exda_key] = self.extrapolated_da[exda_key]*emittance_factor


    def clean_data_for_seed(self, seed, dacol='dawsimp', daerrcol='dawsimperr', angle=None):
        '''Returns a data frame with the cleaned data for a single seed.
        Only points with the correct monotonicity will be selected.
        The algorithm will check if the minimum possible value for a data point B is
        smaller than the maximum possible value of the previous data point A (nturn(B)>nturn(A)).
        The minimum and maximum possible value is evaluated using the error of dawsimp.'''

        # extract data for selected seed
        dseed = self.data[self.data['seed'] == seed]

        if angle is not None:
            dseed = dseed[dseed['angle']==angle]

        # lastrow = dseed.iloc[-1] # always keep the last row
        dseed = dseed.assign(dawsimpmerr=dseed[dacol] - dseed[daerrcol])
        dseed = dseed.assign(dawsimpperr=dseed[dacol] + dseed[daerrcol])

        if self.check_monotonicity:
            # ensure the monotonicity
            iterations = 0
            while True:
                initial_df_length = len(dseed)
                # dseed = dseed[~(dseed['dawsimp'].diff()>=0.) & ~(dseed['tlossmin'].diff()<=0.)]
                b = dseed[daerrcol]  # dawsimp minus error
                a = dseed[daerrcol].shift(1)  # dawsimp plus error
                dseed = dseed[~(a - b < 0)]
                final_df_length = len(dseed)
                if initial_df_length == final_df_length:
                    break
                iterations += 1

        dseed = dseed.assign(nturnavg=(dseed['nturn'] + dseed['tlossmin']) / 2)

        return dseed

    def _chi_square(self, xdata, y_obs, y_exp, regularization=0):
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

        return ((1 + regularization * xdata) * (y_obs - y_exp) ** 2 / (y_exp)).sum()

    def _generate_dafunction(self, kappa):
        '''Define the function to be used for the fit. Its better to fit with two
		free parameters, due to unstable results when using three parameters.
		Therefore we fix kappa for the fitting.'''

        def dafunc(x, d, b):
            return d + (b / (np.log10(x) ** kappa))

        return dafunc

    def _fit_da_single_kappa(self, kappa, seed, xaxis, dacol='dawsimp', daerrcol='dawsimperr', angle=None, regularization=0):
        '''Fits the DA for a single seed with a defined kappa
        Returns (d,b,kappa,chisquared), pcov'''

        # select data for the desired seed
        data = self.clean_data_for_seed(seed, angle=angle, dacol=dacol, daerrcol=daerrcol)
        self.cleaned_data = data

        if len(data) < self._min_datapoints:
            raise InsufficientDataError(
                "_fit_da_single_kappa requires at least {0} data points".format(self._min_datapoints))

        # get data for horizontal axis [turn] and vertical axis [DA] and DA error [dawsimperr]
        xdata = data[xaxis][1:]
        ydata = data[dacol][1:]
        if angle is None:
            yerr = data[daerrcol][1:]
        else:
            yerr = None

        # generate the da function for the given kappa
        dafunc = self._generate_dafunction(kappa)

        # perform the fit
        try:
            db, pcov = curve_fit(dafunc, xdata, ydata, sigma=yerr)
        except RuntimeError:
            return 0, 0, 0, 1000, 0, 0

        d, b = db

        # calculate the chi square of the fit

        y_exp = davst_function(xdata, d, b, kappa)
        y_obs = ydata
        chisq = self._chi_square(xdata, y_obs, y_exp, regularization=regularization)

        return d, b, kappa, chisq, pcov[0, 0], pcov[1, 1]

    def fit(self, seeds=range(1, 61), kappas=[-5, 5, 0.1], angles = None, dacol='dawsimp', daerrcol='dawsimperr',
            xaxis='nturnavg', regularization=0, save=False):
        '''Fits the da for a given range of seeds and kappas.
		A fit is only carried out if more than 8 data points are available.
		The number of minimum datapoints required can be set by the self._min_datapoints variable.'''
        ki, kf, kstep = kappas
        results = []

        # columns of the output data frame
        if angles is None:
            cols = ['seed', 'angle', 'emit', 'd', 'b', 'k', 'chi', 'derr', 'berr']
            angles = [None]
        else:
            cols = ['seed', 'angle', 'emit', 'd', 'b', 'k', 'chi', 'derr', 'berr']

        # nested loop over seeds and possible values for kappa
        for angle in angles:
            for seed in seeds:
                try:
                    output = []
                    for k in np.arange(ki, kf, kstep):
                        d, b, kappa, chisq, derr, berr = self._fit_da_single_kappa(k, seed, xaxis,
                                                                                    dacol=dacol, daerrcol=daerrcol,
                                                                                    angle=angle,
                                                                                    regularization=regularization)

                        output.append([seed, angle, self.emit, d, b, kappa, chisq, derr, berr])

                    output = pd.DataFrame(output, columns=cols)

                    indx_bestfit = output['chi'].argmin()  # index of the best fit
                    dfslice = output.iloc[indx_bestfit]  # get fit parameters of the best fit
                    results.append(dfslice)
                except InsufficientDataError:
                    continue
        results = pd.DataFrame(results)
        results = results.reset_index(drop=True)
        self.fit_params = results[:]
        self.extrapolated_da = results[:]
        if save:
            self.save_fit_params()


class beamloss:
    def __init__(self):
        self.load_loss_da()
        pass

    def load_loss_da(self):
        '''Load the losses vs da into a dictionary'''
        lossda = {}
        for da in np.arange(0.1, 20.1, 0.1):
            da = round(da, 1)
            basedir = '/afs/cern.ch/work/p/phermes/public/loss_vs_da'
            ldf = pd.read_csv("{0}/loss_da_{1:04.1f}.dat".format(basedir, da), names=['loss'])
            lossda['{0:04.1f}'.format(da)] = np.array(ldf)
        self.lossda = lossda

    def _get_loss_from_single_da(self, da):
        da = round(da, 1)
        return self.lossda['{0:04.1f}'.format(da)]

    def get_loss_from_da_series(self, da_series):
        output = []
        for da in da_series:
            try:
                output.append(self._get_loss_from_single_da(da))
            except KeyError:
                pass
        output = np.array(output).flatten()
        return output

    def loss_from_fit_params(self, fit_params, realizations=1, turns=None, time=None):

        if turns is None and time is None:
            raise ValueError(
                'Calculation of loss from fit parameters requires to indicate either the revolution time or the number of turns to take into account')
            return
        elif turns is None and time is not None:
            turns = 11245.5 * time

        if realizations == 1:
            nfit_params = fit_params
        else:
            nfit_params = get_fitting_params_distribution(fit_params, realizations)

        da_series = get_da_from_fit_output(nfit_params, turns)
        loss = self.get_loss_from_da_series(da_series)
        return loss


def get_fitting_params_distribution(df, size):
    '''Returns the distribution of fitting parameters using the fitting errors'''
    output = []
    # for row in df[['d', 'b', 'k', 'derr', 'berr']].iterrows():
    for row in df.iterrows():

        row = row[1]
        # get the fit parameters
        d_dist = np.random.normal(row['d'], scale=row['derr'], size=size)
        b_dist = np.random.normal(row['b'], scale=row['berr'], size=size)
        k = row['k']
        for i in range(len(d_dist)):
            output.append([row['seed'], row['angle'], row['emit'], d_dist[i], b_dist[i], k, 0, 0, 0])
    output = pd.DataFrame(output, columns=['seed','angle', 'emit', 'd', 'b', 'k', 'chi', 'derr', 'berr'])
    return output


def get_da_from_fit_output(fit_output, turns):
    '''Calculates the DA for a given number of turns from the fit output'''
    output = []
    for params in fit_output.iterrows():
        params = params[1]
        k, b, d = params['k'], params['b'], params['d']
        output.append(davst_function(turns, d, b, k))
    return pd.Series(output)


def get_extrapolated_da(fit_params, turns, realizations=1):
    '''Calculate the extrapolated da '''
    if realizations == 1:
        return get_da_from_fit_output(fit_params, turns)
    fit_params = get_fitting_params_distribution(fit_params, realizations)
    da_series = get_da_from_fit_output(fit_params, turns)
    return da_series
