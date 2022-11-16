import os
import numpy as np
import pandas as pd

from src import grid
from astropy.table import Table
import h5py

import time

### step 1: define a tracking reading function

def read_models(filepath):
	try:
		f = h5py.File(filepath, 'r')
		print(filepath)
		cols = ['index','star_age', 'star_mass','luminosity','radius','Teff','Yinit','Xinit','Zinit',
		'log_g','amlt','FeH','delta_nu_scaling','numax_scaling','acoustic_cutoff', 'profile_number', 'flag_seismo']

		atrack = Table([f[col][:] for col in cols], names=cols)
		atrack = atrack[atrack['flag_seismo']==1]

		atrack['mode_l'] = np.zeros(len(atrack), dtype=object)
		atrack['mode_n'] = np.zeros(len(atrack), dtype=object)
		atrack['mode_freq'] = np.zeros(len(atrack), dtype=object)
		atrack['mode_inertia'] = np.zeros(len(atrack), dtype=object)

		# # replace the acoustic_cutoff from mesa with numax_scaling
		atrack['acoustic_cutoff'] = atrack['numax_scaling']

		for i in range(len(atrack)):
			# historyIndex = atrack[i]['index']
			profileIndex = atrack[i]['profile_number']
			atrack['mode_l'][i] = f['profile{:0.0f}/{:s}'.format(profileIndex, 'l')][:][0]
			atrack['mode_n'][i] = f['profile{:0.0f}/{:s}'.format(profileIndex, 'n_pg')][:][0]
			atrack['mode_freq'][i] = f['profile{:0.0f}/{:s}'.format(profileIndex, 'freq')][:][0]
			atrack['mode_inertia'][i] = f['profile{:0.0f}/{:s}'.format(profileIndex, 'E_norm')][:][0]
		return atrack

	except KeyError:
		return None

### step 2: provide a list of files, each of which is the path of an evolutionary track
tracks = ['test_models/'+f for f in os.listdir('test_models/') if (f.endswith('.h5') & f.startswith('index000'))]

### step 3: set up various parameters in params.py
from params import user_setup_params

if __name__ == '__main__':

	start_time = time.time()

	### step 4: grid modelling
	# initialize a grid modelling class
	g = grid(read_models, tracks, user_setup_params)

	# run!
	g.run()

	end_time = time.time() 

	with open('time_consumption.txt', 'a') as file :
		file.write('start_time: {:0.2f}, use_time: {:0.2f}\n'.format(start_time, end_time-start_time))
	