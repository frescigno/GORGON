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

current_run = "noGP_3pl"

Mstar = 0.686

saving_folder = Path("/home/fr307/HD48948_analysis/" + current_run)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

data = ascii.read("/home/fr307/star_data/HD48948/HD48948_yarara_v2_timeseries.rdb").to_pandas()
JD = data["jdb"].values + 2.4e6
rv = data["vrad"].values
err_rv = data["svrad"].values
rv_offset = np.mean(rv)
rv = rv - rv_offset

prior_list=[]


######## KERNEL HYPERPARAMETERS ########

hparam = gp.Par_Creator.create("JitterQuasiPer")
hparam['gp_per'] = gp.Parameter(value=45., error=0., vary=False)
hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0., vary=False)
hparam['gp_explength'] = gp.Parameter(value=90., error=10., vary=False)
hparam['gp_amp'] = gp.Parameter(value=0., error=0., vary=False)
hparam['gp_jit'] = gp.Parameter(value=0.7, error=0.2, vary=False)

######## MODEL PARAMETERS ########

models_list = ["Kepler", "Kepler", "Kepler"]
Model_Par_Creator = gp.Model_Par_Creator()
model_par = Model_Par_Creator.create(models_list)

######## Planet b

model_par["P_0"] = gp.Parameter(value=7.34, error=0.01)
model_par["K_0"] = gp.Parameter(value=3., error=0.5)
model_par["ecc_0"] = gp.Parameter(value=0., error=0.01, vary=True)
model_par["omega_0"] = gp.Parameter(value=np.pi/2, error=0.1, vary=True)
model_par["t0_0"] = gp.Parameter(value=2456574, error=1.)

######## Priors for planet b
prior_P0 = gp.Prior_Par_Creator.create("Uniform")
prior_P0["minval"] = 7.33
prior_P0["maxval"] = 7.35
prior_list.append(("P_0", "Uniform", prior_P0))

prior_t00 = gp.Prior_Par_Creator.create("Uniform")
prior_t00["minval"] = 2456572
prior_t00["maxval"] = 2456576
prior_list.append(("t0_0", "Uniform", prior_t00))


######## Planet c
model_par['P_1'] = gp.Parameter(value=38., error=0.1)
model_par['K_1'] = gp.Parameter(value=3., error=1.)
model_par['ecc_1'] = gp.Parameter(value=0.0, error=0.01, vary = True)
model_par['omega_1'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)
model_par['t0_1'] = gp.Parameter(value=JD[0], error=0.1)

######## Priors for planet c
prior_P1 = gp.Prior_Par_Creator.create("Uniform")
prior_P1["minval"] = 37.
prior_P1["maxval"] = 40.
prior_list.append(("P_1", "Uniform", prior_P1))


######## Planet d
model_par['P_2'] = gp.Parameter(value=151., error=1.0)
model_par['K_2'] = gp.Parameter(value=3., error=1.)
model_par['ecc_2'] = gp.Parameter(value=0.0, error=0.01, vary = True)
model_par['omega_2'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)
model_par['t0_2'] = gp.Parameter(value=JD[0], error=0.1)

######## Priors for planet d
prior_P2 = gp.Prior_Par_Creator.create("Uniform")
prior_P2["minval"] = 130.
prior_P2["maxval"] = 170.
prior_list.append(("P_2", "Uniform", prior_P2))

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
    savefilename="Initial_data_plot")
'''

########## SAVING INITIAL CONDITIONS ##########
file1 = os.path.join(saving_folder, "initial_conditions.txt")
initial_cond_file = open(file1, "w+")
initial_cond_file.write("\nUsed priors:\n")
initial_cond_file.write(prior_list.__str__())
initial_cond_file.write("\nInitial Parameters:\n")
initial_cond_file.write(model_par.__str__())
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
        param6 = param[:,5] # Period2
        param7 = param[:,6] # K2
        param8 = param[:,7] # Sk2
        param9 = param[:,8] # Ck2
        param10 = param[:,9] # to2
        param11 = param[:,10] # Period3
        param12 = param[:,11] # K3
        param13 = param[:,12] # Sk3
        param14 = param[:,13] # Ck3
        param15 = param[:,14] # to3
        return param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14, param15

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
        param6 = param[:,5] # Period2
        param7 = param[:,6] # K2
        param8 = param[:,7] # Sk2
        param9 = param[:,8] # Ck2
        param10 = param[:,9] # to2
        param11 = param[:,10] # Period3
        param12 = param[:,11] # K3
        param13 = param[:,12] # Sk3
        param14 = param[:,13] # Ck3
        param15 = param[:,14] # to3
        return param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14, param15

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


######## Saving model parameters
post_P0_nB, post_K0_nB, post_sk0_nB, post_ck0_nB, post_t00_nB, post_P1_nB, post_K1_nB, post_sk1_nB, post_ck1_nB, post_t01_nB, post_P2_nB, post_K2_nB, post_sk2_nB, post_ck2_nB, post_t02_nB = posterior_recovery_noburnin(fin_model_param, burnin, kernel=False)
fileP0_nB = os.path.join(saving_folder, "posterior_P0_nB.txt")
np.savetxt(fileP0_nB, post_P0_nB)
fileK0_nB = os.path.join(saving_folder, "posterior_K0_nB.txt")
np.savetxt(fileK0_nB, post_K0_nB)
filesk0_nB = os.path.join(saving_folder, "posterior_sk0_nB.txt")
np.savetxt(filesk0_nB, post_sk0_nB)
fileck0_nB = os.path.join(saving_folder, "posterior_ck0_nB.txt")
np.savetxt(fileck0_nB, post_ck0_nB)
filet00_nB = os.path.join(saving_folder, "posterior_t00_nB.txt")
np.savetxt(filet00_nB, post_t00_nB)
fileP1_nB = os.path.join(saving_folder, "posterior_P1_nB.txt")
np.savetxt(fileP1_nB, post_P1_nB)
fileK1_nB = os.path.join(saving_folder, "posterior_K1_nB.txt")
np.savetxt(fileK1_nB, post_K1_nB)
filesk1_nB = os.path.join(saving_folder, "posterior_sk1_nB.txt")
np.savetxt(filesk1_nB, post_sk1_nB)
fileck1_nB = os.path.join(saving_folder, "posterior_ck1_nB.txt")
np.savetxt(fileck1_nB, post_ck1_nB)
filet01_nB = os.path.join(saving_folder, "posterior_t01_nB.txt")
np.savetxt(filet01_nB, post_t01_nB)
fileP2_nB = os.path.join(saving_folder, "posterior_P2_nB.txt")
np.savetxt(fileP2_nB, post_P2_nB)
fileK2_nB = os.path.join(saving_folder, "posterior_K2_nB.txt")
np.savetxt(fileK2_nB, post_K2_nB)
filesk2_nB = os.path.join(saving_folder, "posterior_sk2_nB.txt")
np.savetxt(filesk2_nB, post_sk2_nB)
fileck2_nB = os.path.join(saving_folder, "posterior_ck2_nB.txt")
np.savetxt(fileck2_nB, post_ck2_nB)
filet02_nB = os.path.join(saving_folder, "posterior_t02_nB.txt")
np.savetxt(filet02_nB, post_t02_nB)

post_P0, post_K0, post_sk0, post_ck0, post_t00, post_P1, post_K1, post_sk1, post_ck1, post_t01, post_P2, post_K2, post_sk2, post_ck2, post_t02 = posterior_recovery(fin_model_param, kernel=False)
fileP0 = os.path.join(saving_folder, "posterior_P0.txt")
np.savetxt(fileP0, post_P0)
fileK0 = os.path.join(saving_folder, "posterior_K0.txt")
np.savetxt(fileK0, post_K0)
filesk0 = os.path.join(saving_folder, "posterior_sk0.txt")
np.savetxt(filesk0, post_sk0)
fileck0 = os.path.join(saving_folder, "posterior_ck0.txt")
np.savetxt(fileck0, post_ck0)
filet00 = os.path.join(saving_folder, "posterior_t00.txt")
np.savetxt(filet00, post_t00)
fileP1 = os.path.join(saving_folder, "posterior_P1.txt")
np.savetxt(fileP1, post_P1)
fileK1 = os.path.join(saving_folder, "posterior_K1.txt")
np.savetxt(fileK1, post_K1)
filesk1 = os.path.join(saving_folder, "posterior_sk1.txt")
np.savetxt(filesk1, post_sk1)
fileck1 = os.path.join(saving_folder, "posterior_ck1.txt")
np.savetxt(fileck1, post_ck1)
filet01 = os.path.join(saving_folder, "posterior_t01.txt")
np.savetxt(filet01, post_t01)
fileP2 = os.path.join(saving_folder, "posterior_P2.txt")
np.savetxt(fileP2, post_P2)
fileK2 = os.path.join(saving_folder, "posterior_K2.txt")
np.savetxt(fileK2, post_K2)
filesk2 = os.path.join(saving_folder, "posterior_sk2.txt")
np.savetxt(filesk2, post_sk2)
fileck2 = os.path.join(saving_folder, "posterior_ck2.txt")
np.savetxt(fileck2, post_ck2)
filet02 = os.path.join(saving_folder, "posterior_t02.txt")
np.savetxt(filet02, post_t02)

######## Saving eccentricity and omega
post_ecc0 = []
post_omega0 = []
post_ecc1 = []
post_omega1 = []
post_ecc2 = []
post_omega2 = []
for a in range(len(post_sk0)):
    ecc0, omega0 = aux.to_ecc(post_sk0[a], post_ck0[a])
    post_ecc0.append(ecc0)
    post_omega0.append(omega0)
    ecc1, omega1 = aux.to_ecc(post_sk1[a], post_ck1[a])
    post_ecc1.append(ecc1)
    post_omega1.append(omega1)
    ecc2, omega2 = aux.to_ecc(post_sk2[a], post_ck2[a])
    post_ecc2.append(ecc2)
    post_omega2.append(omega2)

fileecc0 = os.path.join(saving_folder, "posterior_ecc0.txt")
np.savetxt(fileecc0, post_ecc0)
fileomega0 = os.path.join(saving_folder, "posterior_omega0.txt")
np.savetxt(fileomega0, post_omega0)
fileecc1 = os.path.join(saving_folder, "posterior_ecc1.txt")
np.savetxt(fileecc1, post_ecc1)
fileomega1 = os.path.join(saving_folder, "posterior_omega1.txt")
np.savetxt(fileomega1, post_omega1)
fileecc2 = os.path.join(saving_folder, "posterior_ecc2.txt")
np.savetxt(fileecc2, post_ecc2)
fileomega2 = os.path.join(saving_folder, "posterior_omega2.txt")
np.savetxt(fileomega2, post_omega2)

post_ecc0_nB = []
post_omega0_nB = []
post_ecc1_nB = []
post_omega1_nB = []
post_ecc2_nB = []
post_omega2_nB = []
for a in range(len(post_sk0_nB)):
    ecc0_nB, omega0_nB = aux.to_ecc(post_sk0_nB[a], post_ck0_nB[a])
    post_ecc0_nB.append(ecc0_nB)
    post_omega0_nB.append(omega0_nB)
    ecc1_nB, omega1_nB = aux.to_ecc(post_sk1_nB[a], post_ck1_nB[a])
    post_ecc1_nB.append(ecc1_nB)
    post_omega1_nB.append(omega1_nB)
    ecc2_nB, omega2_nB = aux.to_ecc(post_sk2_nB[a], post_ck2_nB[a])
    post_ecc2_nB.append(ecc2_nB)
    post_omega2_nB.append(omega2_nB)

fileecc0_nB = os.path.join(saving_folder, "posterior_ecc0_nB.txt")
np.savetxt(fileecc0_nB, post_ecc0_nB)
fileomega0_nB = os.path.join(saving_folder, "posterior_omega0_nB.txt")
np.savetxt(fileomega0_nB, post_omega0_nB)
fileecc1_nB = os.path.join(saving_folder, "posterior_ecc1_nB.txt")
np.savetxt(fileecc1_nB, post_ecc1_nB)
fileomega1_nB = os.path.join(saving_folder, "posterior_omega1_nB.txt")
np.savetxt(fileomega1_nB, post_omega1_nB)
fileecc2_nB = os.path.join(saving_folder, "posterior_ecc2_nB.txt")
np.savetxt(fileecc2_nB, post_ecc2_nB)
fileomega2_nB = os.path.join(saving_folder, "posterior_omega2_nB.txt")
np.savetxt(fileomega2_nB, post_omega2_nB)


######## Saving the entire arrays ready for a corner plot
fin_model_param2d = saving_params(fin_model_param)
fin_hparams2d = saving_params(fin_hparams)
fileparams= os.path.join(saving_folder, "fin_model_param.txt")
np.savetxt(fileparams, fin_model_param2d)





######## Reconstruct GP info and get basic keplerian plots ########
Model_Par_Creator = gp.Model_Par_Creator()
model_par2 = Model_Par_Creator.create(models_list)

model_par2['P_0'] = gp.Parameter(value=np.percentile(post_P0_nB, 50), error=(np.percentile(post_P0_nB, 84)-np.percentile(post_P0_nB, 16))/2, vary=True)
model_par2['K_0'] = gp.Parameter(value=np.percentile(post_K0_nB, 50), error=(np.percentile(post_K0_nB, 84)-np.percentile(post_K0_nB, 16))/2, vary=True)
model_par2['ecc_0'] = gp.Parameter(value=np.percentile(post_ecc0_nB, 50), error=(np.percentile(post_ecc0_nB, 84)-np.percentile(post_ecc0_nB, 16))/2, vary=True)
model_par2['omega_0'] = gp.Parameter(value=np.percentile(post_omega0_nB, 50), error=(np.percentile(post_omega0_nB, 84)-np.percentile(post_omega0_nB, 16))/2, vary=True)
model_par2['t0_0'] = gp.Parameter(value=np.percentile(post_t00_nB, 50), error=(np.percentile(post_t00_nB, 84)-np.percentile(post_t00_nB, 16))/2, vary=True)

model_par2['P_1'] = gp.Parameter(value=np.percentile(post_P1_nB, 50), error=(np.percentile(post_P1_nB, 84)-np.percentile(post_P1_nB, 16))/2, vary=True)
model_par2['K_1'] = gp.Parameter(value=np.percentile(post_K1_nB, 50), error=(np.percentile(post_K1_nB, 84)-np.percentile(post_K1_nB, 16))/2, vary=True)
model_par2['ecc_1'] = gp.Parameter(value=np.percentile(post_ecc1_nB, 50), error=(np.percentile(post_ecc1_nB, 84)-np.percentile(post_ecc1_nB, 16))/2, vary=True)
model_par2['omega_1'] = gp.Parameter(value=np.percentile(post_omega1_nB, 50), error=(np.percentile(post_omega1_nB, 84)-np.percentile(post_omega1_nB, 16))/2, vary=True)
model_par2['t0_1'] = gp.Parameter(value=np.percentile(post_t01_nB, 50), error=(np.percentile(post_t01_nB, 84)-np.percentile(post_t01_nB, 16))/2, vary=True)

model_par2['P_2'] = gp.Parameter(value=np.percentile(post_P2_nB, 50), error=(np.percentile(post_P2_nB, 84)-np.percentile(post_P2_nB, 16))/2, vary=True)
model_par2['K_2'] = gp.Parameter(value=np.percentile(post_K2_nB, 50), error=(np.percentile(post_K2_nB, 84)-np.percentile(post_K2_nB, 16))/2, vary=True)
model_par2['ecc_2'] = gp.Parameter(value=np.percentile(post_ecc2_nB, 50), error=(np.percentile(post_ecc2_nB, 84)-np.percentile(post_ecc2_nB, 16))/2, vary=True)
model_par2['omega_2'] = gp.Parameter(value=np.percentile(post_omega2_nB, 50), error=(np.percentile(post_omega2_nB, 84)-np.percentile(post_omega2_nB, 16))/2, vary=True)
model_par2['t0_2'] = gp.Parameter(value=np.percentile(post_t02_nB, 50), error=(np.percentile(post_t02_nB, 84)-np.percentile(post_t02_nB, 16))/2, vary=True)


model_y = get_model(models_list, JD, model_par2, to_ecc=False)
smooth_model_y = get_model(models_list, xpred, model_par2, to_ecc=False)
loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par2, "JitterQuasiPer")
logL = loglik.LogL(prior_list)
GP_rv, GP_err = loglik.predict(xpred)

######## Saving GP arrays
fileGPx= os.path.join(saving_folder, "GP_x.txt")
np.savetxt(fileGPx, xpred)

######## Saving complete model arrays
filemodel= os.path.join(saving_folder, "model_fin.txt")
np.savetxt(filemodel, model_y)
filemodel= os.path.join(saving_folder, "smooth_model_fin.txt")
np.savetxt(filemodel, smooth_model_y)


######## Saving single keplerian arrays
######## Planet b
model_par_pl0 = Model_Par_Creator.create(["kepler"])
model_par_pl0['P'] = model_par['P_0']
model_par_pl0['K'] = model_par['K_0']
model_par_pl0['ecc'] = model_par['ecc_0']
model_par_pl0['omega'] = model_par['omega_0']
model_par_pl0['t0'] = model_par['t0_0']
planet_0_model = get_model(["Kepler"], JD, model_par_pl0, to_ecc=False)
smooth_planet_0_model = get_model(["kepler"], xpred, model_par_pl0, to_ecc=False)

filemodel= os.path.join(saving_folder, "planet_0_model.txt")
np.savetxt(filemodel, planet_0_model)
filemodel= os.path.join(saving_folder, "smooth_planet_0_model.txt")
np.savetxt(filemodel, smooth_planet_0_model)

######## Planet c
model_par_pl1 = Model_Par_Creator.create(["kepler"])
model_par_pl1['P'] = model_par['P_1']
model_par_pl1['K'] = model_par['K_1']
model_par_pl1['ecc'] = model_par['ecc_1']
model_par_pl1['omega'] = model_par['omega_1']
model_par_pl1['t0'] = model_par['t0_1']
planet_1_model = get_model(["Kepler"], JD, model_par_pl1, to_ecc=False)
smooth_planet_1_model = get_model(["kepler"], xpred, model_par_pl1, to_ecc=False)

filemodel= os.path.join(saving_folder, "planet_1_model.txt")
np.savetxt(filemodel, planet_1_model)
filemodel= os.path.join(saving_folder, "smooth_planet_1_model.txt")
np.savetxt(filemodel, smooth_planet_1_model)

######## Planet d
model_par_pl2 = Model_Par_Creator.create(["kepler"])
model_par_pl2['P'] = model_par['P_2']
model_par_pl2['K'] = model_par['K_2']
model_par_pl2['ecc'] = model_par['ecc_2']
model_par_pl2['omega'] = model_par['omega_2']
model_par_pl2['t0'] = model_par['t0_2']
planet_2_model = get_model(["Kepler"], JD, model_par_pl2, to_ecc=False)
smooth_planet_2_model = get_model(["kepler"], xpred, model_par_pl2, to_ecc=False)

filemodel= os.path.join(saving_folder, "planet_2_model.txt")
np.savetxt(filemodel, planet_2_model)
filemodel= os.path.join(saving_folder, "smooth_planet_2_model.txt")
np.savetxt(filemodel, smooth_planet_2_model)



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


allbut_pl0 = planet_1_model+planet_2_model
allbut_pl1 = planet_0_model+planet_2_model
allbut_pl2 = planet_0_model+planet_1_model

phased_x_pl0 = aux.phasefold(JD,model_par["P_0"].value, model_par["t0_0"].value)
phased_x_pl0, phased_rv_pl0, phased_err_rv_pl0 = ordering(phased_x_pl0, np.array(rv-allbut_pl0), err_rv)
phased_x_pl1 = aux.phasefold(JD,model_par["P_1"].value, model_par["t0_1"].value)
phased_x_pl1, phased_rv_pl1, phased_err_rv_pl1 = ordering(phased_x_pl1, np.array(rv-allbut_pl1), err_rv)
phased_x_pl2 = aux.phasefold(JD,model_par["P_2"].value, model_par["t0_2"].value)
phased_x_pl2, phased_rv_pl2, phased_err_rv_pl2 = ordering(phased_x_pl2, np.array(rv-allbut_pl2), err_rv)

smooth_phased_x_pl0 = aux.phasefold(xpred,model_par["P_0"].value, model_par["t0_0"].value)
smooth_phased_x_pl0, smooth_phased_rv_pl0 = ordering(smooth_phased_x_pl0, smooth_planet_0_model)
smooth_phased_x_pl1 = aux.phasefold(xpred,model_par["P_1"].value, model_par["t0_1"].value)
smooth_phased_x_pl1, smooth_phased_rv_pl1 = ordering(smooth_phased_x_pl1, smooth_planet_1_model)
smooth_phased_x_pl2 = aux.phasefold(xpred,model_par["P_2"].value, model_par["t0_2"].value)
smooth_phased_x_pl2, smooth_phased_rv_pl2 = ordering(smooth_phased_x_pl2, smooth_planet_2_model)

phased_pl0_mod = aux.phasefold(JD, model_par["P_0"].value, model_par["t0_0"].value)
phased_pl0_mod, phased_pl0_mod_rv = ordering(phased_pl0_mod, planet_0_model)
phased_pl1_mod = aux.phasefold(JD, model_par["P_1"].value, model_par["t0_1"].value)
phased_pl1_mod, phased_pl1_mod_rv = ordering(phased_pl1_mod, planet_1_model)
phased_pl2_mod = aux.phasefold(JD, model_par["P_2"].value, model_par["t0_2"].value)
phased_pl2_mod, phased_pl2_mod_rv = ordering(phased_pl2_mod, planet_2_model)

phased_pl0 = plotting.phase_plot(phased_x_pl0,phased_rv_pl0,phased_err_rv_pl0,phased_pl0_mod_rv,smooth_phased_x_pl0, smooth_phased_rv_pl0, residuals=True, save_folder=saving_folder, savefilename="final_pl0_phase")
phased_pl1 = plotting.phase_plot(phased_x_pl1,phased_rv_pl1,phased_err_rv_pl1,phased_pl1_mod_rv,smooth_phased_x_pl1, smooth_phased_rv_pl1, residuals=True, save_folder=saving_folder, savefilename="final_pl1_phase")
phased_pl2 = plotting.phase_plot(phased_x_pl2,phased_rv_pl2,phased_err_rv_pl2,phased_pl2_mod_rv,smooth_phased_x_pl2, smooth_phased_rv_pl2, residuals=True, save_folder=saving_folder, savefilename="final_pl2_phase")
'''

mass0 = []
mass1 = []
mass2 = []
for i in range(len(post_P0_nB)):
    mass0_chain = mc.mass_calc(post_P0_nB[i],post_K0_nB[i], post_sk0_nB[i], post_ck0_nB[i],Mstar)
    mass0.append(mass0_chain)
    mass1_chain = mc.mass_calc(post_P1_nB[i],post_K1_nB[i], post_sk1_nB[i], post_ck1_nB[i],Mstar)
    mass1.append(mass1_chain)
    mass2_chain = mc.mass_calc(post_P2_nB[i],post_K2_nB[i], post_sk2_nB[i], post_ck2_nB[i],Mstar)
    mass2.append(mass2_chain)
filemass0 = os.path.join(saving_folder, "mass0.txt")
np.savetxt(filemass0, mass0)
filemass1 = os.path.join(saving_folder, "mass1.txt")
np.savetxt(filemass1, mass1)
filemass2 = os.path.join(saving_folder, "mass2.txt")
np.savetxt(filemass2, mass2)
final_mass0, final_mass0_err = np.percentile(mass0,50), ((np.percentile(mass0,84)-np.percentile(mass0,16))/2)
final_mass1, final_mass1_err = np.percentile(mass1,50), ((np.percentile(mass1,84)-np.percentile(mass1,16))/2)
final_mass2, final_mass2_err = np.percentile(mass2,50), ((np.percentile(mass2,84)-np.percentile(mass2,16))/2)
    

########## SAVING Final CONDITIONS ##########
file2 = os.path.join(saving_folder, "final_conditions.txt")
final_cond_file = open(file2, "w+")
final_cond_file.write("\nUsed priors:\n")
final_cond_file.write(prior_list.__str__())
final_cond_file.write("\nFinal Parameters:\n") 
final_cond_file.write(model_par2.__str__())
final_cond_file.write("\n\nFinal Log Likelihood:\n")
final_cond_file.write(logL.__str__())
final_cond_file.write("\nTotal iterations:\n")
final_cond_file.write(completed_iterations.__str__())
final_cond_file.write("\n\nBurn in iterations:\n")
final_cond_file.write(burnin.__str__())
final_cond_file.write("\n\nNumber of chains:\n")
final_cond_file.write(n_chains.__str__())
final_cond_file.write("\n\nMasses (in earth mass):\n")
final_cond_file.write("\nPlanet 0:\n")
final_cond_file.write(final_mass0.__str__())
final_cond_file.write(" +- ")
final_cond_file.write(final_mass0_err.__str__())
final_cond_file.write("\nPlanet 1:\n")
final_cond_file.write(final_mass1.__str__())
final_cond_file.write(" +- ")
final_cond_file.write(final_mass1_err.__str__())
final_cond_file.write("\nPlanet 2:\n")
final_cond_file.write(final_mass2.__str__())
final_cond_file.write(" +- ")
final_cond_file.write(final_mass2_err.__str__())
final_cond_file.close()


    