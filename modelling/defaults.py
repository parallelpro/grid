import numpy as np 

compulsory_global_params = [
    "observables", 
    "estimators", 
    "estimators_to_plot",
    "filepath_stellar_params", 
    "col_starIDs" ]

default_global_params = {
    # 1 - pass data
    "observables": None,
    "estimators": None,

    "filepath_stellar_params": None,
    "col_starIDs": None,
    "col_obs_Dnu": "Dnu",
    "col_obs_numax": "numax",

    "filepath_stellar_freqs": None,
    "col_obs_freq": "fc",
    "col_obs_e_freq": "e_fc",
    "col_obs_l": "l",

    # 2 - output settings
    "if_plot": True, 
    "estimators_to_plot": None,
    "if_data": True, 

    "Nthread": 1, 

    "filepath_output": 'output/',

    # 3 - classical constraints
    "if_classical": True, 
    "weight_classical": 1, 
    "col_age_name": "star_age", 
    
    # 4 - seismic constraints
    "if_seismic": False, 
    "weight_seismic": 1, 
    "col_mode_freq": "mode_freq",  
    "col_mode_l": "mode_l", 
    "col_mode_inertia": "mode_inertia", 
    "col_acoustic_cutoff": "acoustic_cutoff", 
    "col_mode_n": "mode_n", 

    "if_reduce_seis_chi2": True, 

    "if_correct_surface": True, 
    "surface_correction_formula": "cubic", 

    "if_add_model_error": True,
    "add_model_error_method": 2,
    "rescale_percentile": 0, 
    "col_model_error": "mod_e_freq",

    "if_regularize": False, 
    "weight_reg": 1, 
    "Nreg": 0,
    
    "if_reduce_seis_reg_chi2": True,
}

