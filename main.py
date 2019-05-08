import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table, Column
import corner

import multiprocessing
from functools import partial

class grid:
	"""
	
	Estimate stellar parameters from evolutionary grid models.

	"""

	def __init__(self, read_models, tracks, outdir, observables, estimators,
	 stars_obs_qty, stars_obserr_qty, starname=None):

		"""
		Initialize a parameter estimation class.

		----------
		Input:

		read_models: function reference
			The function takes one argument ``atrack'' which is the path of
			a track, and returns a table-like object (like the one defined in
			``astropy'').

		tracks: array-like[Ntrack,]
			A list containing the paths of all tracks.

		outdir: str
			The directory to store output figures and tables.

		observables: array-like[Nobservable,]
			The observables which are matched between models and observations.
			Make sure each string is callable from the returning table of the
			``read_models'' function.

		estimators: array-like[Nestimator,]
			The parameters desired to estimate. Make sure each string is 
			callable from the returning table of the ``read_models'' function.
	
		stars_obs_qty: array-like[Nstar, Nobservable]
			The array which stores all observation quantities.

		stars_obserr_qty: array-like[Nstar, Nobservable]
			The array which stores all uncertainties of the observational 
			quantities.

		----------
		Optional input:

		starname: list of str, default: list of ascending natural numbers
			The starnames to differentiate folders.

		"""

		self.read_models = read_models
		self.tracks = tracks
		self.outdir = outdir
		self.observables = observables
		self.estimators = estimators
		self.stars_obs_qty = stars_obs_qty
		self.stars_obserr_qty = stars_obserr_qty

		if not os.path.exists(self.outdir):
			os.mkdir(self.outdir)


		self.starname = starname

	def quantile(self, x, q, weights=None):

		"""
		Compute sample quantiles with support for weighted samples.

		----------
		Input:

		x: array-like[nsamples,]
			The samples.

		q: array-like[nquantiles,]
			The list of quantiles to compute. These should all be in the range
			``[0, 1]``.

		----------
		Optional input:

		weights : Optional[array-like[nsamples,]]
			An optional weight corresponding to each sample. These

		----------
		Output:

		quantiles: array-like[nquantiles,]
			The sample quantiles computed at ``q``.

		----------
		Modified based on ``corner``.

		Copyright (c) 2013-2016 Daniel Foreman-Mackey
		All rights reserved.

		Redistribution and use in source and binary forms, with or without
		modification, are permitted provided that the following conditions are met:

		1. Redistributions of source code must retain the above copyright notice, this
		   list of conditions and the following disclaimer.
		2. Redistributions in binary form must reproduce the above copyright notice,
		   this list of conditions and the following disclaimer in the documentation
		   and/or other materials provided with the distribution.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
		ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
		WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
		DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
		ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
		(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
		LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
		ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
		(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
		SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

		The views and conclusions contained in the software and documentation are those
		of the authors and should not be interpreted as representing official policies,
		either expressed or implied, of the FreeBSD Project.
		
		"""

		x = np.atleast_1d(x)
		q = np.atleast_1d(q)

		if weights is None:
			return np.percentile(x, 100.0 * q)
		else:
			weights = np.atleast_1d(weights)
			idx = np.argsort(x, axis=0)

			res = []
			for i in range(x.shape[1]):
				sw = weights[idx[:,i]]
				cdf = np.cumsum(sw)[:-1]
				cdf /= cdf[-1]
				cdf = np.append(0, cdf)
				res.append(np.interp(q, cdf, x[idx[:,i],i]))
			return np.array(res).T

	def assign_prob_to_models(self, starnames, tracks):

		Nobservable = len(self.observables)
		Nestimator = len(self.estimators)
		Nstar = len(starnames)
		Ntrack = len(tracks)

		model_prob = [np.array([]) for istar in range(Nstar)]
		model_estimator = [[np.array([]) for iestimator in range(Nestimator)] for istar in range(Nstar)] 

		for itrack in range(Ntrack): 

			# read in itrack
			table = self.read_models(tracks[itrack])
			Nmodel = len(table) - 2

			# calculate posterior
			for istar in range(Nstar):
				lifespan = (table["star_age"][2:] - table["star_age"][0:-2])/2.0
				prior = lifespan/np.sum(lifespan)
				lnprior = np.log(prior)
				chi2 = np.zeros(Nmodel)
				for iobservable in range(Nobservable):
					chi2 += (self.stars_obs_qty[istar][iobservable]-table[self.observables[iobservable]][1:-1])**2.0/(self.stars_obserr_qty[istar][iobservable]**2.0)
				lnlikelihood = -chi2/2.0
				lnprob = lnprior + lnlikelihood
				prob = np.exp(lnprob)

				# posterior
				model_prob[istar] = np.append(model_prob[istar], prob)

				# estimators
				for iestimator in range(Nestimator):
					model_estimator[istar][iestimator] = np.append(model_estimator[istar][iestimator], table[self.estimators[iestimator]][1:-1])

		return model_prob, model_estimator


	def output_results(self, model_prob, model_estimator, starnames):

		Nstar = len(starnames)

		for istar in range(Nstar):
			toutdir = self.outdir + starnames[istar] + '/'
			if not os.path.exists(toutdir):
				os.mkdir(toutdir)
			weights = model_prob[istar]
			index = weights>=0.0
			weights = weights[index]
			samples = np.array(model_estimator[istar]).T[index,:]
			if samples.shape[0] <= samples.shape[1]:
				f = open(toutdir+starnames[istar]+'_log.txt', 'w')
				f.write("Failed because samples.shape[0] <= samples.shape[1].")
				f.close()
			else:
				fig = corner.corner(samples, labels=self.estimators, quantiles=(0.16, 0.5, 0.84), weights=weights)
				fig.savefig(toutdir+starnames[istar]+"_triangle.png")
				plt.close()
				results = self.quantile(samples, (0.16, 0.5, 0.84), weights=weights)
				ascii.write(Table(results, names=self.estimators), toutdir+starnames[istar]+"_summary.txt",format="csv", overwrite=True)
		return 

	def estimate_parameters(self, Nthread=1, verbose=True):

		"""
		Initialize a parameter estimation class.

		----------
		Optional input:

		Nthread: int
			The number of available threads to enable parallel computing.


		"""

		Ntrack, Nestimator, Nstar = len(self.tracks), len(self.estimators), len(self.starname)

		# assign prob to models
		Ntrack_per_thread = int(Ntrack/Nthread)+1
		arglist = [self.tracks[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(Nthread)]
		pool = multiprocessing.Pool(processes=Nthread)
		part_func = partial(self.assign_prob_to_models, self.starname)
		result_list = pool.map(part_func, arglist)
		pool.close()

		# merge probs from different threads
		model_prob = [np.array([]) for istar in range(Nstar)]
		model_estimator = [[np.array([]) for iestimator in range(Nestimator)] for istar in range(Nstar)] 
		for ithread in range(Nthread):
			for istar in range(Nstar):
				model_prob[istar] = np.append(model_prob[istar], result_list[ithread][0][istar])
				for iestimator in range(Nestimator):
					model_estimator[istar][iestimator] = np.append(model_estimator[istar][iestimator], result_list[ithread][1][istar][iestimator])

		# normalize probs
		for istar in range(Nstar):
			model_prob[istar] /= np.sum(model_prob[istar])

		# output results
		# assign prob to models
		Nstar_per_thread = int(Nstar/Nthread)+1
		arglist = [(model_prob[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
				model_estimator[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
				self.starname[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread]) for ithread in range(Nthread)]
		pool = multiprocessing.Pool(processes=Nthread)
		pool.starmap(self.output_results, arglist)
		pool.close()

		return

	def tar_outputs(self):

		"""
		Tar the output directory.

		"""

		os.system("tar zcvf output.tar.gz " + self.outdir)
		return


