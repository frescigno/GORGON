"""
Created on Thu Jul 27 14:26:26 2023

@author: frescigno

Launcher for solar rv temp-dependent analysis
"""

import os
os.environ["OMP_NUM_THREADS"] = "5"
import numpy as np
import matplotlib.pyplot as plt
import time as tm
from pathlib import Path
import os
import pandas as pd

import auxiliary as aux
import GP_solar_multi as gp
import plotting as plot
import mass_calc as mc
from MCMC import ben_run_MCMC as run
from MCMC import get_model


###### DATA IMPORT ##### Needs to be changed
filedir = "/home/fr307/star_data/Sun_temp/"
savedir = "/home/fr307/TempRVs_analysis/decrease/chunk1/"
filename = 'vrad_Sun_T6_decrease_upper.csv' #need to change based on file used
datacut=[57250,57390] #change based on chosen cut (numbers can be approximations)
iterations = 50000
numb_chains = 100
numb_cores=5

# important imports and info
filepath = Path(filedir+filename)
data = pd.read_csv(filepath)
time = data['time'].values
bins = list(data.columns.values)
list_vrads=[d for d in bins if 'val' in d]
list_errs=[d for d in bins if 'err' in d]
list_bins=list_vrads
list_bins=[d.replace('vrad_val_','') for d in list_bins]

list_vrads=sorted(list_vrads)
list_errs=sorted(list_errs)
list_bins=sorted(list_bins)


###### SAVING FUNCTIONS #####
def saving_params(params):
    '''3d arrays cannot be saved, get rid of chains'''
    params = np.array(params)
    shapes = params.shape
    numb_chains = shapes[0]
    nparam = shapes[1]
    depth = shapes[2]
    param = np.zeros((((depth)*numb_chains),nparam))
    for p in range(nparam):
        numb=0
        for c in range(numb_chains):
            for i in range(depth):
                param[numb][p] = params[c][p][i]
                numb += 1
    return param



# Recover and save each posterior separately
def posterior_recovery(params, kernel=True):
    '''Function that isolates and recovers all the posteriors for the individual parameters '''
    params = np.array(params)
    shapes = params.shape
    numb_chains = shapes[0]
    nparam = shapes[1]
    depth = shapes[2]
    
    param = np.zeros((((depth)*numb_chains),nparam))
    for p in range(nparam):
        numb=0
        for c in range(numb_chains):
            for i in range(depth):
                param[numb][p] = params[c][p][i]
                numb += 1
    if kernel:
        param1 = param[:,0] # gp per
        param2 = param[:,1] # gp perlength
        param3 = param[:,2] # gp explength (evolution)
        param4 = param[:,3] # gp amp
        param5 = param[:,4] # jitter
        return param1, param2, param3, param4, param5
    else:
        param1 = param[:,0] # a0
        param2 = param[:,1] # a1
        param3 = param[:,2] # a2
        param4 = param[:,3] # a3
        return param1, param2, param3, param4
    


# using this one takes away the burn in
def posterior_recovery_noburnin(params, burnin, kernel=True):
    params = np.array(params)
    shapes = params.shape
    numb_chains = shapes[0]
    nparam = shapes[1]
    depth = shapes[2]
    
    param = np.zeros((((depth-burnin)*numb_chains),nparam))
    for p in range(nparam):
        numb=0
        for c in range(numb_chains):
            for i in range(depth):
                if i >= burnin:
                    param[numb][p] = params[c][p][i]
                    numb += 1
    if kernel:
        param1 = param[:,0] # gp per
        param2 = param[:,1] # gp perlength
        param3 = param[:,2] # gp explength (evolution)
        param4 = param[:,3] # gp amp
        param5 = param[:,4] # jitter
        return param1, param2, param3, param4, param5
    else:
        param1 = param[:,0] # a0
        param2 = param[:,1] # a1
        param3 = param[:,2] # a2
        param4 = param[:,3] # a3
        return param1, param2, param3, param4



# Set up function, including all the initial guesses and priors, can be amended if needed
def setup(time, vrad, vrad_err):
    
    hparam = gp.Par_Creator.create("JitterQuasiPer")
    hparam['gp_per'] = gp.Parameter(value=27., error=5.)
    hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.05)
    hparam['gp_explength'] = gp.Parameter(value=30, error=5.)
    hparam['gp_amp'] = gp.Parameter(value=np.nanstd(vrad), error=np.mean(vrad_err)*5)
    hparam['gp_jit'] = gp.Parameter(value=np.mean(vrad_err), error=.1)

    prior_list = []
    prior_param1 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param1["mu"] = hparam['gp_per'].value
    prior_param1["sigma"] = 5.
    prior_list.append(("gp_per", "Gaussian", prior_param1))

    prior_param2 = gp.Prior_Par_Creator.create("Uniform")  
    prior_param2["minval"] = 0.
    prior_param2["maxval"] = 1.
    prior_list.append(("gp_perlength", "Uniform", prior_param2))

    prior_param3 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param3["mu"] = hparam['gp_perlength'].value
    prior_param3["sigma"] = 0.1
    prior_list.append(("gp_perlength", "Gaussian", prior_param3))

    prior_param4 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param4["mu"] = hparam['gp_explength'].value
    prior_param4["sigma"] = 15.
    prior_list.append(("gp_explength", "Gaussian", prior_param4))

    prior_jit = gp.Prior_Par_Creator.create("Gaussian")
    prior_jit["mu"] = np.mean(vrad_err)
    prior_jit["sigma"] = 0.1
    prior_list.append(("gp_jit", "Gaussian", prior_jit))

    
    model_list = ["No"]
    Model_Par_Creator = gp.Model_Par_Creator()
    model_par = Model_Par_Creator.create(model_list)
    model_par["no"] = gp.Parameter(value=0., error=0., vary=False)
    model_y = get_model(model_list, time, model_par)

    xpred = np.arange(time[0]-1., time[-1]+1., 0.3)

    loglik = gp.GPLikelyhood(time, vrad, model_y, vrad_err, hparam, model_par, "JitterQuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)


    plt.plot(xpred, GP_rv)
    plt.scatter(time, vrad)
    
    return hparam, model_par, model_y, prior_list, xpred, logL, GP_rv, GP_err





###### LETS LOOP OVER ALL BINS #######
for i in range(len(list_vrads)):
    
    print(f'Beginning analysis on the {list_bins[i]} bin')
    
    saving_folder = Path(f'{savedir}/{list_bins[i]}/')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    time = data["time"].values
    vrad, vrad_err = data[list_vrads[i]].values,data[list_errs[i]].values
    
    file_check = os.path.join(saving_folder, "check.txt")
    check = open(file_check, "w+")
    check.write(list_bins[i].__str__())
    check.write(list_vrads[i].__str__())
    check.write(list_errs[i].__str__())
    check.close()
    
    #start = (np.abs(time-datacut[0])).argmin()
    #end = (np.abs(time-datacut[1])).argmin()
    #time,vrad,vrad_err = time[start:end],vrad[start:end], vrad_err[start:end]
    datacut_idx = (time >= datacut[0]) & (time <= datacut[1])
    time,vrad,vrad_err = time[datacut_idx],vrad[datacut_idx], vrad_err[datacut_idx]
    
    hparam, model_par, model_y, prior_list, xpred, logL, GP_rv, GP_err = setup(time, vrad, vrad_err)
    
    ########## SAVING INITIAL CONDITIONS ##########
    file1 = os.path.join(saving_folder, "initial_conditions.txt")
    initial_cond_file = open(file1, "w+")
    initial_cond_file.write("\nInitial Hyperparameters:\n")
    initial_cond_file.write(hparam.__str__())
    initial_cond_file.write("\n\nInitial Log Likelihood:\n")
    initial_cond_file.write(logL.__str__())
    initial_cond_file.write("\n\nMaximum iterations:\n")
    initial_cond_file.write(iterations.__str__())
    initial_cond_file.write("\n\nNumber of chains:\n")
    initial_cond_file.write(numb_chains.__str__())
    initial_cond_file.close()
    
    print('Set-up completed. Starting MCMC')
    
    ######## RUNNING MCMC ########
    logL_chain, fin_hparams, fin_model_param, completed_iterations = run(iterations, time, vrad, vrad_err, hparam, "JitterQuasiPer", model_par, ["No"], prior_list, numb_chains=numb_chains, plot_convergence=True, saving_folder=saving_folder, numb_cores=numb_cores)
    print("MCMC completed in ", completed_iterations, " iterations")
    burnin = int(completed_iterations*0.2)
    print("Burn in: ", burnin, " iterations")
    
    ###### SOME PLOTTING ######
    plot.mixing_plot(completed_iterations, numb_chains, fin_hparams, "JitterQuasiPer", fin_model_param, ["No"], logL_chain, save_folder=saving_folder)
    final_param_values, final_param_erru, final_param_errd = plot.corner_plot(fin_hparams, "JitterQuasiPer", fin_model_param, ["No"], save_folder=saving_folder, errors=True)
    
    print('Finished plotting \nStarting saving posteriors.')
    
    ###### SOME SAVING ######
    fin_param2d = saving_params(fin_hparams)
    # Saving the full 3D set of MCMC parameters
    fileparams= os.path.join(saving_folder, "fin_param.txt")
    np.savetxt(fileparams, fin_param2d)
    
    post_gpper, post_gpharm, post_gpevo, post_gpamp, post_jit = posterior_recovery(fin_hparams)
    filegpP = os.path.join(saving_folder, "posterior_gp_P.txt")
    np.savetxt(filegpP, post_gpper)
    filegpharm = os.path.join(saving_folder, "posterior_gp_harm.txt")
    np.savetxt(filegpharm, post_gpharm)
    filegpevo = os.path.join(saving_folder, "posterior_gp_evo.txt")
    np.savetxt(filegpevo, post_gpevo)
    filegpamp = os.path.join(saving_folder, "posterior_gp_amp.txt")
    np.savetxt(filegpamp, post_gpamp)
    filejit= os.path.join(saving_folder, "posterior_gp_jit.txt")
    np.savetxt(filejit, post_jit)
    post_gpper_nB, post_gpharm_nB, post_gpevo_nB, post_gpamp_nB, post_jit_nB = posterior_recovery_noburnin(fin_hparams, burnin)
    filegpP = os.path.join(saving_folder, "posterior_gp_P_nB.txt")
    np.savetxt(filegpP, post_gpper_nB)
    filegpharm = os.path.join(saving_folder, "posterior_gp_harm_nB.txt")
    np.savetxt(filegpharm, post_gpharm_nB)
    filegpevo = os.path.join(saving_folder, "posterior_gp_evo_nB.txt")
    np.savetxt(filegpevo, post_gpevo_nB)
    filegpamp = os.path.join(saving_folder, "posterior_gp_amp_nB.txt")
    np.savetxt(filegpamp, post_gpamp_nB)
    filejit= os.path.join(saving_folder, "posterior_gp_jit_nB.txt")
    np.savetxt(filejit, post_jit_nB)
    
    
    print('All posteriors saved \nStarting final plotting and saving')
    
    ##### FINAL RESULTS PLOTTING #####
    hparam['gp_per'] = gp.Parameter(value=np.percentile(post_gpper_nB, 50), error=(np.percentile(post_gpper_nB, 84)-np.percentile(post_gpper_nB, 16))/2, vary=True)
    hparam['gp_perlength'] = gp.Parameter(value=np.percentile(post_gpharm_nB, 50), error=(np.percentile(post_gpharm_nB, 84)-np.percentile(post_gpharm_nB, 16))/2, vary=True)
    hparam['gp_explength'] = gp.Parameter(value=np.percentile(post_gpevo_nB, 50), error=(np.percentile(post_gpevo_nB, 84)-np.percentile(post_gpevo_nB, 16))/2, vary=True)
    hparam['gp_amp'] = gp.Parameter(value=np.percentile(post_gpamp_nB, 50), error=(np.percentile(post_gpamp_nB, 84)-np.percentile(post_gpamp_nB, 16))/2, vary=True)
    hparam['gp_jit'] = gp.Parameter(value=np.percentile(post_jit_nB, 50), error=(np.percentile(post_jit_nB, 84)-np.percentile(post_jit_nB, 16))/2, vary=True)
    
    
    model_y = get_model(["No"], time, model_par)
    
    loglik = gp.GPLikelyhood(time, vrad, model_y, vrad_err, hparam, model_par, "JitterQuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    fileGP = os.path.join(saving_folder, "GP_pred.txt")
    np.savetxt(fileGP, GP_rv)
    fileGPerr= os.path.join(saving_folder, "GP_err.txt")
    np.savetxt(fileGPerr, GP_err)
    fileGPx= os.path.join(saving_folder, "GP_x.txt")
    np.savetxt(fileGPx, xpred)
    
    plot.GP_plot(time, vrad, vrad_err, model_y, xpred, GP_rv, GP_err, residuals=True, xlabel='Time [BJD]', ylabel='RV [m/s]', save_folder=saving_folder, savefilename='GP_plot')
    
    ########## SAVING Final CONDITIONS ##########
    file2 = os.path.join(saving_folder, "final_conditions.txt")
    final_cond_file = open(file2, "w+")
    final_cond_file.write("\nFinal Hyperparameters:\n")
    final_cond_file.write(hparam.__str__())
    final_cond_file.write("\n\nFinal Log Likelihood:\n")
    final_cond_file.write(logL.__str__())
    final_cond_file.write("\nTotal iterations:\n")
    final_cond_file.write(completed_iterations.__str__())
    final_cond_file.write("\n\nBurn in iterations:\n")
    final_cond_file.write(burnin.__str__())
    final_cond_file.write("\n\nNumber of chains:\n")
    final_cond_file.write(numb_chains.__str__())
    final_cond_file.close()
    
    print(f'All steps completed for bin {list_bins[i]}')

