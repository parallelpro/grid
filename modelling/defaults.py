import numpy as np 

compulsory_global_params = [
    "observables", 
    "estimators", 
    "filepath_stellar_params", 
    "col_starIDs" ]

default_global_params = {
    "col_obs_freq": "fc",
    "col_obs_e_freq": "e_fc",
    "col_obs_l": "l",
    "col_obs_Dnu": "Dnu",
    "col_obs_numax": "numax",
    "if_plot": True, 
    "estimators_to_plot": True, 
    "if_data": True, 
    "Nthread": 1, 
    "filepath_output": 'output/',
    "if_classical": True, 
    "weight_classical": 1, 
    "if_add_timstep_prior": False,
    "col_age_name": "star_age", 
    "if_seismic": False, 
    "weight_seismic": 1, 
    "col_mode_freq": "mode_freq",  
    "col_mode_l": "mode_l", 
    "col_mode_inertia": "mode_inertia", 
    "col_acoustic_cutoff": "acoustic_cutoff", 
    "col_mode_n": "mode_n", 
    "if_correct_surface": True, 
    "surface_correction_formula": "cubic", 
    "if_rescale": True, 
    "rescale_percentile": 0, 
    "if_regularize": False, 
    "weight_reg": 1, 
    "Nreg": 0,
    "if_reduce_seis_chi2": False, 
}

