"""
Created on Aug 1 14:44:19 2023

@author: frescigno

Launcher for HD489848 analysis using MAGPy parallelised
"""


n_cores = 5
iterations = 5000
n_chains = 700

import os
os.environ["OMP_NUM_THREADS"] = "5"
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import ascii
import scipy.interpolate as interp

import auxiliary as aux
import plotting
import GP_solar_multi as gp
from MCMC import ben_run_MCMC as run
from MCMC import get_model
import mass_calc as mc

current_run = "GPonly_medium2"


Mstar = 0.686

saving_folder = Path("/home/fr307/HD48948_analysis/" + current_run)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

data = ascii.read("/home/fr307/star_data/HD48948/mockdata_medium.rdb").to_pandas()
#data = ascii.read("/home/fr307/star_data/HD48948/HD48948_rhk_withoutmag.rdb").to_pandas()
JD = data["jdb"].values + 2.4e6
rv = data["vrad"].values 
err_rv = data["svrad"].values
rv_offset = np.mean(rv)
rv = rv - rv_offset
#logrhk = data["rhk"].values 
#err_logrhk = data["sig_rhk"].values
#logrhk_offset = np.mean(logrhk)
#logrhk = logrhk - logrhk_offset
#logrhk = data["na"].values 
#err_logrhk = data["sig_na"].values
#logrhk_offset = np.mean(logrhk)
#logrhk = logrhk - logrhk_offset

prior_list=[]


######## KERNEL HYPERPARAMETERS ########

hparam = gp.Par_Creator.create("JitterQuasiPer")
hparam['gp_per'] = gp.Parameter(value=45., error=3., vary=True)
hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.01, vary=True)
hparam['gp_explength'] = gp.Parameter(value=90., error=10., vary=True)
hparam['gp_amp'] = gp.Parameter(value=np.std(rv), error=np.std(rv)/2., vary=True)
hparam['gp_jit'] = gp.Parameter(value=np.mean(err_rv), error=np.mean(err_rv)/3., vary=True)

######## Priors
prior_jit = gp.Prior_Par_Creator.create("Uniform")
prior_jit["minval"] = 0.
prior_jit["maxval"] = 2.
prior_list.append(("gp_jit", "Uniform", prior_jit))


'''prior_harm = gp.Prior_Par_Creator.create("Gaussian")
prior_harm["mu"] = 0.5
prior_harm["sigma"] = 0.05
prior_list.append(("gp_perlength", "Gaussian", prior_harm))'''

prior_harm2 = gp.Prior_Par_Creator.create("Uniform") 
prior_harm2["minval"] = 0.
prior_harm2["maxval"] = 1.
prior_list.append(("gp_perlength", "Uniform", prior_harm2))

prior_per = gp.Prior_Par_Creator.create("Uniform")
prior_per["minval"] = 40.
prior_per["maxval"] = 47.
prior_list.append(("gp_per", "Uniform", prior_per))


######## MODEL PARAMETERS ########

models_list = ["No"]
Model_Par_Creator = gp.Model_Par_Creator()
model_par = Model_Par_Creator.create(models_list)


######## Planet c
model_par['no'] = gp.Parameter(value=1., error=0, vary=False)

model_y = get_model(models_list, JD, model_par, to_ecc=False)

loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "JitterQuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0] - 5.0, JD[-1] + 5.0, 1.0)
GP_rv, GP_err = loglik.predict(xpred)


'''plotting.GP_plot(
    JD,
    rv,
    err_rv,
    model_y,
    xpred,
    GP_rv,
    GP_err,
    residuals=True,
    save_folder=saving_folder,
    savefilename="Initial_GP_plot")

smooth_model_y = get_model(models_list, xpred, model_par, to_ecc=False)
smooth_model_end = smooth_model_y + GP_rv
plotting.data_plot(
    JD,
    rv,
    err_y=err_rv,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_end,
    model_y=model_y,
    save_folder=saving_folder,
    savefilename="Initial_data_plot")'''


########## SAVING INITIAL CONDITIONS ##########
file1 = os.path.join(saving_folder, "initial_conditions.txt")
initial_cond_file = open(file1, "w+")
initial_cond_file.write("\nUsed priors:\n")
initial_cond_file.write(prior_list.__str__())
initial_cond_file.write("\nInitial Hyperparameters:\n")
initial_cond_file.write(hparam.__str__())
initial_cond_file.write("\n\nInitial Log Likelihood:\n")
initial_cond_file.write(logL.__str__())
initial_cond_file.close()



######## RUNNING MCMC ########
logL_chain, fin_hparams, fin_model_param, completed_iterations = run(
    iterations,
    JD,
    rv,
    err_rv,
    hparam,
    "JitterQuasiPer",
    model_par,
    models_list,
    prior_list,
    numb_chains=n_chains,
    plot_convergence=True,
    numb_cores=n_cores)

print("MCMC completed in ", completed_iterations, " iterations")
burnin = int(completed_iterations*0.2)
print("Burn in: ", burnin, " iterations")

######## Plotting basic MCMC health plots
plotting.mixing_plot(
    iterations,
    n_chains,
    fin_hparams,
    "JitterQuasiPer",
    fin_model_param,
    models_list,
    logL_chain,
    save_folder=saving_folder)

final_param_values, final_param_erru, final_param_errd = plotting.corner_plot(
    fin_hparams,
    "JitterQuasiPer",
    fin_model_param,
    models_list,
    save_folder=saving_folder,
    errors=True)


########## SAVING POSTERIORS ##########
######## Saving functions
def posterior_recovery(params, kernel=False):
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
        param1 = param[:,0] # Period1
        param2 = param[:,1] # K1
        param3 = param[:,2] # Sk1
        param4 = param[:,3] # Ck1
        param5 = param[:,4] # to1
        return param1, param2, param3, param4, param5


def posterior_recovery_noburnin(params, burnin, kernel=False):
    '''Function that isolates and recovers all the posteriors for the individual parameters '''
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
        param1 = param[:,0] # Period1
        param2 = param[:,1] # K1
        param3 = param[:,2] # Sk1
        param4 = param[:,3] # Ck1
        param5 = param[:,4] # to1
        return param1, param2, param3, param4, param5

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


######## Saving hyperparameters
post_gpper, post_gpharm, post_gpevo, post_gpamp, post_jit = posterior_recovery(fin_hparams, kernel=True)
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

post_gpper_nB, post_gpharm_nB, post_gpevo_nB, post_gpamp_nB, post_jit_nB = posterior_recovery_noburnin(fin_hparams, burnin, kernel=True)
filegpP_nB = os.path.join(saving_folder, "posterior_gp_P_nB.txt")
np.savetxt(filegpP_nB, post_gpper_nB)
filegpharm_nB = os.path.join(saving_folder, "posterior_gp_harm_nB.txt")
np.savetxt(filegpharm_nB, post_gpharm_nB)
filegpevo_nB = os.path.join(saving_folder, "posterior_gp_evo_nB.txt")
np.savetxt(filegpevo_nB, post_gpevo_nB)
filegpamp_nB = os.path.join(saving_folder, "posterior_gp_amp_nB.txt")
np.savetxt(filegpamp_nB, post_gpamp_nB)
filejit_nB= os.path.join(saving_folder, "posterior_gp_jit_nB.txt")
np.savetxt(filejit_nB, post_jit_nB)

######## Saving the entire arrays ready for a corner plot
fin_hparams2d = saving_params(fin_hparams)
filehparams= os.path.join(saving_folder, "fin_hparams.txt")
np.savetxt(filehparams, fin_hparams2d)






######## Reconstruct GP info and get basic keplerian plots ########
hparam2 = gp.Par_Creator.create("JitterQuasiPer")
hparam2["gp_per"] = gp.Parameter(value=np.percentile(post_gpper_nB, 50), error=(np.percentile(post_gpper_nB, 84)-np.percentile(post_gpper_nB, 16))/2, vary=True)
hparam2["gp_perlength"] = gp.Parameter(value=np.percentile(post_gpharm_nB, 50), error=(np.percentile(post_gpharm_nB, 84)-np.percentile(post_gpharm_nB, 16))/2, vary=True)
hparam2["gp_explength"] = gp.Parameter(value=np.percentile(post_gpevo_nB, 50), error=(np.percentile(post_gpevo_nB, 84)-np.percentile(post_gpevo_nB, 16))/2, vary=True)
hparam2["gp_amp"] = gp.Parameter(value=np.percentile(post_gpamp_nB, 50), error=(np.percentile(post_gpamp_nB, 84)-np.percentile(post_gpamp_nB, 16))/2, vary=True)
hparam2['gp_jit'] = gp.Parameter(value=np.percentile(post_jit_nB, 50), error=(np.percentile(post_jit_nB, 84)-np.percentile(post_jit_nB, 16))/2, vary=True)


Model_Par_Creator = gp.Model_Par_Creator()

model_y = get_model(models_list, JD, model_par, to_ecc=False)
smooth_model_y = get_model(models_list, xpred, model_par, to_ecc=False)
loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam2, model_par, "JitterQuasiPer")
logL = loglik.LogL(prior_list)
GP_rv, GP_err = loglik.predict(xpred)

######## Saving GP arrays
fileGP = os.path.join(saving_folder, "GP_pred.txt")
np.savetxt(fileGP, GP_rv)
fileGPerr= os.path.join(saving_folder, "GP_err.txt")
np.savetxt(fileGPerr, GP_err)
fileGPx= os.path.join(saving_folder, "GP_x.txt")
np.savetxt(fileGPx, xpred)



'''
######## Phase plots prep: phasefolding the data and re-ordering it
def ordering(phase,y, err=None):
    order = np.argsort(phase)
    phase = phase[order]
    y = y[order]
    if err is not None:
        err = err[order]
        return phase, y, err
    else:
        return phase,y


#activity = interp.interp1d(xpred, GP_rv, kind='cubic')
pl0_only_smooth = GP_rv+smooth_planet_0_model #we only have pl1 left
pl1_only_smooth = GP_rv+smooth_planet_1_model #we only have pl0 left
pl1_f = interp.interp1d(xpred, pl1_only_smooth, kind='cubic')
pl0_f = interp.interp1d(xpred, pl0_only_smooth, kind='cubic')
pred_pl0 = pl0_f(JD) #here we have GP and pl0
pred_pl1 = pl1_f(JD) #here we have GP and pl1

activity = interp.interp1d(xpred, GP_rv, kind='cubic')
GP_rv_data = activity(JD)
allbut_pl0 = GP_rv_data+planet_1_model
allbut_pl1 = GP_rv_data+planet_0_model

phased_x_pl0 = aux.phasefold(JD,model_par["P_0"].value, model_par["t0_0"].value)
#phased_x_pl0, phased_rv_pl0, phased_err_rv_pl0 = ordering(phased_x_pl0, np.array(rv-allbut_pl0), err_rv)
phased_x_pl1 = aux.phasefold(JD,model_par["P_1"].value, model_par["t0_1"].value)
phased_x_pl1, phased_rv_pl1, phased_err_rv_pl1 = ordering(phased_x_pl1, np.array(rv-allbut_pl1), err_rv)

smooth_phased_x_pl0 = aux.phasefold(xpred,model_par["P_0"].value, model_par["t0_0"].value)
#smooth_phased_x_pl0, smooth_phased_rv_pl0 = ordering(smooth_phased_x_pl0, smooth_planet_0_model)
smooth_phased_x_pl1 = aux.phasefold(xpred,model_par["P_1"].value, model_par["t0_1"].value)
smooth_phased_x_pl1, smooth_phased_rv_pl1 = ordering(smooth_phased_x_pl1, smooth_planet_1_model)

#phased_pl0_mod = aux.phasefold(JD, model_par["P_0"].value, model_par["t0_0"].value)
phased_pl0_mod, phased_pl0_mod_rv = ordering(phased_x_pl0, planet_0_model)
#phased_pl1_mod = aux.phasefold(JD, model_par["P_1"].value, model_par["t0_1"].value)
phased_pl1_mod, phased_pl1_mod_rv = ordering(phased_x_pl1, planet_1_model)




phased_pl0 = plotting.phase_plot(phased_x_pl0,rv-GP_rv_data-planet_1_model,err_rv,model_y=planet_0_model,smooth_model_phase=smooth_phased_x_pl0, smooth_model_y=smooth_planet_0_model, residuals=True, save_folder=saving_folder, savefilename="final_pl0_phase")
phased_pl1 = plotting.phase_plot(phased_x_pl1,phased_rv_pl1,phased_err_rv_pl1,phased_pl1_mod_rv,smooth_phased_x_pl1, smooth_phased_rv_pl1, residuals=True, save_folder=saving_folder, savefilename="final_pl1_phase")
'''


########## SAVING Final CONDITIONS ##########
file2 = os.path.join(saving_folder, "final_conditions.txt")
final_cond_file = open(file2, "w+")
final_cond_file.write("\nUsed priors:\n")
final_cond_file.write(prior_list.__str__())
final_cond_file.write("\nFinal Hyperparameters:\n")
final_cond_file.write(hparam2.__str__())
final_cond_file.write("\n\nFinal Log Likelihood:\n")
final_cond_file.write(logL.__str__())
final_cond_file.write("\nTotal iterations:\n")
final_cond_file.write(completed_iterations.__str__())
final_cond_file.write("\n\nBurn in iterations:\n")
final_cond_file.write(burnin.__str__())
final_cond_file.write("\n\nNumber of chains:\n")
final_cond_file.write(n_chains.__str__())


    