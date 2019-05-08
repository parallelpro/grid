import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table, Column

from main import grid


def read_mist_ms_tracks(atrack):
	# read the main sequence part of a mist track
	table = ascii.read(atrack, converters={'#':' '}, header_start=11)
	table = table[table['phase'] == 0]
	
	# add observables in the table that do not exist: feh
	z_surf_solar, x_surf_solar = 0.017, 0.735 # gs98
	feh = np.log10((1.-table['surface_h1']-table['surface_he4']-
		table['surface_he3'])/table['surface_h1'])-np.log10(z_surf_solar/x_surf_solar)
	table.add_column(Column(feh, name='feh'))

	# add observables that are power10 versions of the original columns
	cols = ['log_Teff', 'log_L', 'log_R']
	newcols = ['Teff', 'L', 'R']
	for icol in range(len(cols)):
		table.add_column(10.0**table[cols[icol]], name=newcols[icol])

	return table


# search mist tracks
databasedir = '../dwarf_mist/'
for parentdir, dirname, filenames in os.walk(databasedir):
	a = [filenames[i].endswith('.track.eep') for i in range(0,len(filenames))]
	f = filenames
a, f = np.array(a), np.array(f)
tracks = f[a]
tracks = np.array([databasedir+tracks[itrack] for itrack in range(len(tracks))])

outdir = 'output/'
observables = np.array(['Teff', 'feh', 'L', 'log_g'])
estimators = np.array(['star_mass', 'star_age', 'mass_conv_core', 
						'Teff', 'L', 'R', 'log_g', 'delta_nu', 
						'delta_Pg', 'nu_max'])
stars = ascii.read('Mc14_table1_input_parameters.txt', format='csv', delimiter="|")[0:20]
stars_obs_qty = np.array([stars['teff_lamost_lasp'], 
						stars['feh_lamost_lasp'], 
						stars['Lum_gaia'], 
						stars['logg_lamost_lasp']]).T
stars_obserr_qty = np.array([stars['teff_lamost_lasp_err'], 
						stars['feh_lamost_lasp_err'], 
						(stars['E_Lum']+stars['e_Lum'])/2.0, 
						stars['logg_lamost_lasp_err']]).T
starnames = np.array(stars['kicnumber'], dtype='str')

mygrid=grid(read_mist_ms_tracks, tracks, outdir, 
	observables, estimators, stars_obs_qty, stars_obserr_qty, starname=starnames)
mygrid.estimate_parameters(Nthread=20)
