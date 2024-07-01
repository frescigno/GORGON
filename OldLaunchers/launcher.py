#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:27:17 2022

@author: frescigno

Launcher for HD489848 analysis
"""


import numpy as np
import GP_solar_multi as gp
import matplotlib.pyplot as plt
import plotting
from pathlib import Path
import auxiliary


from MCMC_affine_multi_ben import ben_run_MCMC as run
from MCMC_affine_multi_old import get_model
import os

from astropy.io import ascii

current_run = "new_test"

n_cores = 2
iterations = 100
n_chains = 2000
timeit = True

saving_folder = Path("/home/bsl204/MAGPy/new/" + current_run)
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)


# inputfilenameDACE = "RVs_MMLSD1"
# myinput = Path(

# )
# inputfile = open(myinput, "r")
# TOI_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
# Skipp 2 header lines and the first row (bad datapoint)
# JD = TOI_all[:, 0] + 2400000
# rv = TOI_all[:, 3]
# err_rv = TOI_all[:, 4]

data = ascii.read("/home/bsl204/HD48948/Data/HD48948_drs_timeseries.rdb").to_pandas()

JD = data["jdb"].values + 2.4e6
rv = data["vrad"].values
err_rv = data["svrad"].values
rv_offset = np.mean(rv)
rv = rv - rv_offset


# plotting.data_plot(JD, rv, err_y=err_rv)


hparam = gp.Par_Creator.create("QuasiPer")
hparam["gp_per"] = gp.Parameter(value=45.0, error=2.0)
##### IDEA OF ROATION PERIOD, ANDREW, PERIODOGRAM, AND VSIN FOR UPPER LIMIT (FORMULA)
########### ASK ANDREW FOR RADIUS AND I
hparam["gp_perlength"] = gp.Parameter(value=0.5, error=0.05)
hparam["gp_explength"] = gp.Parameter(value=2 * hparam["gp_per"].value, error=5.0)
######### APRROXIMATE ACC2019 PAPER TO SAY FACULAE DOMINATED, ABOUT 2X ROTATION PERIOD
##### giles et al. 2017
hparam["gp_amp"] = gp.Parameter(value=np.nanstd(rv), error=2.0)


prior_list = []

prior_param3_b = gp.Prior_Par_Creator.create("Jeffrey")
prior_param3_b["minval"] = 0.1
prior_param3_b["maxval"] = 500.0
prior_list.append(("gp_explength", "Jeffrey", prior_param3_b))

prior_param2_b = gp.Prior_Par_Creator.create("Jeffrey")
prior_param2_b["minval"] = 0.1
prior_param2_b["maxval"] = 500.0
prior_list.append(("gp_per", "Jeffrey", prior_param2_b))

prior_param_b = gp.Prior_Par_Creator.create("Uniform")
prior_param_b["minval"] = 0.0
prior_param_b["maxval"] = 1.0
prior_list.append(("gp_perlength", "Uniform", prior_param_b))

prior_param4_b = gp.Prior_Par_Creator.create("Gaussian")
prior_param4_b["mu"] = hparam["gp_per"].value
prior_param4_b["sigma"] = 3.0
prior_list.append(("gp_per", "Gaussian", prior_param4_b))

"""prior_param5_b = gp.Prior_Par_Creator.create("Gaussian")  
prior_param5_b["mu"] = 0.5
prior_param5_b["sigma"] = 0.05
prior_list.append(("gp_perlength", "Gaussian", prior_param5_b))"""


models_list = ["Kepler", "Kepler"]
Model_Par_Creator = gp.Model_Par_Creator()
model_par = Model_Par_Creator.create(models_list)
########## ASK ANDREW FOR TESS CURVE ##########

model_par["P_0"] = gp.Parameter(value=7.34, error=0.05)
model_par["K_0"] = gp.Parameter(value=2.0, error=0.5)
###### COULD TAKE THE RMS ########
model_par["ecc_0"] = gp.Parameter(value=0.0, error=0.01, vary=True)
model_par["omega_0"] = gp.Parameter(value=np.pi / 2, error=0.1, vary=True)

model_par["t0_0"] = gp.Parameter(value=JD[0], error=1.0)
######## add understand 0

model_par["P_1"] = gp.Parameter(value=38.0, error=0.5)
model_par["K_1"] = gp.Parameter(value=1.7, error=0.5)
###### COULD TAKE THE RMS ########
model_par["ecc_1"] = gp.Parameter(value=0.0, error=0.01, vary=True)
model_par["omega_1"] = gp.Parameter(value=np.pi / 2, error=0.1, vary=True)

model_par["t0_1"] = gp.Parameter(value=JD[0], error=1.0)
print(model_par)

model_y = get_model(models_list, JD, model_par, to_ecc=False)


loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0] - 10.0, JD[-1] + 10.0, 1.0)
GP_rv, GP_err = loglik.predict(xpred)


plotting.GP_plot(
    JD,
    rv,
    err_rv,
    model_y,
    xpred,
    GP_rv,
    GP_err,
    residuals=True,
    save_folder=saving_folder,
    savefilename="Initial_GP_plot",
)


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
    savefilename="Initial_data_plot",
)


model_par_pl0 = Model_Par_Creator.create(["kepler"])
model_par_pl0["P"] = model_par["P_0"]
model_par_pl0["K"] = model_par["K_0"]
###### COULD TAKE THE RMS ########
model_par_pl0["ecc"] = model_par["ecc_0"]
model_par_pl0["omega"] = model_par["omega_0"]
model_par_pl0["t0"] = model_par["t0_0"]
planet_0_model = get_model(["Kepler"], JD, model_par_pl0, to_ecc=False)
smooth_model_y0 = get_model(["kepler"], xpred, model_par_pl0, to_ecc=False)

model_par_pl1 = Model_Par_Creator.create(["kepler"])

model_par_pl1["P"] = model_par["P_1"]
model_par_pl1["K"] = model_par["K_1"]
###### COULD TAKE THE RMS ########
model_par_pl1["ecc"] = model_par["ecc_1"]
model_par_pl1["omega"] = model_par["omega_1"]
model_par_pl1["t0"] = model_par["t0_1"]
planet_1_model = get_model(["Kepler"], JD, model_par_pl1, to_ecc=False)
smooth_model_y1 = get_model(["keple"], xpred, model_par_pl1, to_ecc=False)


pl0 = plotting.Keplerian_only_plot(
    JD,
    rv,
    err_rv,
    xpred,
    GP_rv + smooth_model_y1,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_y0,
    model_y=planet_0_model,
    residuals=True,
    save_folder=saving_folder,
    savefilename="Initial_kepl_plot0",
)
pl1 = plotting.Keplerian_only_plot(
    JD,
    rv,
    err_rv,
    xpred,
    GP_rv + smooth_model_y0,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_y1,
    model_y=planet_1_model,
    residuals=True,
    save_folder=saving_folder,
    savefilename="Initial_kepl_plot1",
)

phase0 = auxiliary.phasefold(JD, model_par["P_0"].value, model_par["t0_0"].value)
smooth_phase0 = auxiliary.phasefold(
    xpred, model_par["P_0"].value, model_par["t0_0"].value
)

import scipy.interpolate as interp

f = interp.interp1d(xpred, GP_rv, kind="cubic")
new_pred_y0 = f(JD)
planet_only_rv0 = rv - new_pred_y0 - planet_1_model

plotting.phase_plot(
    phase0,
    planet_only_rv0,
    err_rv,
    model_y=planet_0_model,
    smooth_model_phase=smooth_phase0,
    smooth_model_y=smooth_model_y0,
    residuals=True,
    xlabel="Time [BJD]",
    ylabel="RV [km s-1]",
    save_folder=saving_folder,
    savefilename="Initial_phase_plot0",
)

phase1 = auxiliary.phasefold(JD, model_par["P_1"].value, model_par["t0_1"].value)
smooth_phase1 = auxiliary.phasefold(
    xpred, model_par["P_1"].value, model_par["t0_1"].value
)
new_pred_y1 = f(JD)
planet_only_rv1 = rv - new_pred_y1 - planet_0_model
plotting.phase_plot(
    phase1,
    planet_only_rv1,
    err_rv,
    model_y=planet_1_model,
    smooth_model_phase=smooth_phase1,
    smooth_model_y=smooth_model_y1,
    residuals=True,
    xlabel="Time [BJD]",
    ylabel="RV [km s-1]",
    save_folder=saving_folder,
    savefilename="Initial_phase_plot1",
)


########## SAVING INITIAL CONDITIONS ##########


file1 = os.path.join(saving_folder, "initial_conditions.txt")
initial_cond_file = open(file1, "w+")
initial_cond_file.write("\nInitial Hyperparameters:\n")
initial_cond_file.write(hparam.__str__())
initial_cond_file.write("\nInitial Parameters:\n")
initial_cond_file.write(model_par.__str__())
initial_cond_file.write("\n\nInitial Log Likelihood:\n")
initial_cond_file.write(logL.__str__())
initial_cond_file.close()

if timeit:
    logL_chain, fin_hparams, fin_model_param, completed_iterations, itertimes = run(
        iterations,
        JD,
        rv,
        err_rv,
        hparam,
        "QuasiPer",
        model_par,
        models_list,
        prior_list,
        numb_chains=n_chains,
        numb_cores=n_cores,
        timeit=True,
    )

    import csv

    with open(f"{saving_folder}/itertimes.csv", "w") as f:
        writer = csv.writer(f)
        for itertime in itertimes:
            writer.writerow([itertime])

else:
    logL_chain, fin_hparams, fin_model_param, completed_iterations = run(
        iterations,
        JD,
        rv,
        err_rv,
        hparam,
        "QuasiPer",
        model_par,
        models_list,
        prior_list,
        numb_chains=n_chains,
        numb_cores=n_cores,
    )

# print(fin_model_param)


plotting.mixing_plot(
    iterations,
    n_chains,
    fin_hparams,
    "QuasiPer",
    fin_model_param,
    models_list,
    logL_chain,
    save_folder=saving_folder,
)

final_param_values = plotting.corner_plot(
    fin_hparams, "QuasiPer", fin_model_param, models_list, save_folder=saving_folder
)

print(np.mean(fin_hparams[:, 0, :]))

hparam2 = gp.Par_Creator.create("QuasiPer")
hparam2["gp_per"] = gp.Parameter(value=np.mean(fin_hparams[:, 0, :]))
hparam2["gp_perlength"] = gp.Parameter(value=np.mean(fin_hparams[:, 1, :]))
hparam2["gp_explength"] = gp.Parameter(value=np.mean(fin_hparams[:, 2, :]))
hparam2["gp_amp"] = gp.Parameter(value=np.mean(fin_hparams[:, 3, :]))

"""fin_model_param[:,2,:], fin_model_param[:,3,:] = 0.0,np.pi*1/2
fin_model_param[:,7,:], fin_model_param[:,8,:] = 0.0,np.pi*1/2"""


model_par2 = Model_Par_Creator.create(models_list)
model_par2["P_0"] = gp.Parameter(value=np.mean(fin_model_param[:, 0, :]))
model_par2["K_0"] = gp.Parameter(value=np.mean(fin_model_param[:, 1, :]))
model_par2["ecc_0"] = gp.Parameter(value=np.mean(fin_model_param[:, 2, :]))
model_par2["omega_0"] = gp.Parameter(value=np.mean(fin_model_param[:, 3, :]))
model_par2["t0_0"] = gp.Parameter(value=np.mean(fin_model_param[:, 4, :]))

(
    model_par2["ecc_0"].value,
    model_par2["omega_0"].value,
    model_par2["ecc_0"].error,
    model_par2["omega_0"].error,
) = auxiliary.to_ecc(
    model_par2["ecc_0"].value,
    model_par2["omega_0"].value,
    errSk=model_par2["ecc_0"].error,
    errCk=model_par2["omega_0"].error,
)

model_par2["P_1"] = gp.Parameter(value=np.mean(fin_model_param[:, 5, :]))
model_par2["K_1"] = gp.Parameter(value=np.mean(fin_model_param[:, 6, :]))
model_par2["ecc_1"] = gp.Parameter(value=np.mean(fin_model_param[:, 7, :]))
model_par2["omega_1"] = gp.Parameter(value=np.mean(fin_model_param[:, 8, :]))
model_par2["t0_1"] = gp.Parameter(value=np.mean(fin_model_param[:, 9, :]))

(
    model_par2["ecc_1"].value,
    model_par2["omega_1"].value,
    model_par2["ecc_1"].error,
    model_par2["omega_1"].error,
) = auxiliary.to_ecc(
    model_par2["ecc_1"].value,
    model_par2["omega_1"].value,
    errSk=model_par2["ecc_0"].error,
    errCk=model_par2["omega_0"].error,
)


model_fin = get_model(models_list, JD, model_par2, to_ecc=False)
loglik2 = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par2, "QuasiPer")
logL2 = loglik2.LogL(prior_list)
GP_rv, GP_err = loglik2.predict(xpred)


"""loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0]-10., JD[-1]+10., 1.)
GP_rv, GP_err = loglik.predict(xpred)"""


# plotting.GP_plot(JD, rv, err_rv, model_y, xpred, GP_rv, GP_err, residuals=True)

smooth_model_y = get_model(models_list, xpred, model_par2, to_ecc=False)
smooth_model_end = smooth_model_y + GP_rv
# plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_y)


model_par_pl0 = Model_Par_Creator.create(["kepler"])
model_par_pl0["P"] = model_par2["P_0"]
model_par_pl0["K"] = model_par2["K_0"]
###### COULD TAKE THE RMS ########
model_par_pl0["ecc"] = model_par2["ecc_0"]
model_par_pl0["omega"] = model_par2["omega_0"]
model_par_pl0["t0"] = model_par2["t0_0"]
planet_0_model = get_model(["Kepler"], JD, model_par_pl0, to_ecc=False)
smooth_model_y0 = get_model(["kepler"], xpred, model_par_pl0, to_ecc=False)

model_par_pl1 = Model_Par_Creator.create(["kepler"])
model_par_pl1["P"] = model_par2["P_1"]
model_par_pl1["K"] = model_par2["K_1"]
###### COULD TAKE THE RMS ########
model_par_pl1["ecc"] = model_par2["ecc_1"]
model_par_pl1["omega"] = model_par2["omega_1"]
model_par_pl1["t0"] = model_par2["t0_1"]
planet_1_model = get_model(["Kepler"], JD, model_par_pl1, to_ecc=False)
smooth_model_y1 = get_model(["keple"], xpred, model_par_pl1, to_ecc=False)


pl0 = plotting.Keplerian_only_plot(
    JD,
    rv,
    err_rv,
    xpred,
    GP_rv + smooth_model_y1,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_y0,
    model_y=planet_0_model,
    residuals=True,
    save_folder=saving_folder,
    savefilename="final_kepl_plot0",
)
pl1 = plotting.Keplerian_only_plot(
    JD,
    rv,
    err_rv,
    xpred,
    GP_rv + smooth_model_y0,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_y1,
    model_y=planet_1_model,
    residuals=True,
    save_folder=saving_folder,
    savefilename="final_kepl_plot1",
)


phase0 = auxiliary.phasefold(JD, model_par2["P_0"].value, model_par2["t0_0"].value)
smooth_phase0 = auxiliary.phasefold(
    xpred, model_par2["P_0"].value, model_par2["t0_0"].value
)

import scipy.interpolate as interp

f = interp.interp1d(xpred, GP_rv, kind="cubic")
new_pred_y0 = f(JD)
planet_only_rv0 = rv - new_pred_y0 - planet_1_model

plotting.phase_plot(
    phase0,
    planet_only_rv0,
    err_rv,
    model_y=planet_0_model,
    smooth_model_phase=smooth_phase0,
    smooth_model_y=smooth_model_y0,
    residuals=True,
    xlabel="Time [BJD]",
    ylabel="RV [km s-1]",
    save_folder=saving_folder,
    savefilename="final_phase_plot0",
)

phase1 = auxiliary.phasefold(JD, model_par2["P_1"].value, model_par2["t0_1"].value)
smooth_phase1 = auxiliary.phasefold(
    xpred, model_par2["P_1"].value, model_par2["t0_1"].value
)
new_pred_y1 = f(JD)
planet_only_rv1 = rv - new_pred_y1 - planet_0_model
plotting.phase_plot(
    phase1,
    planet_only_rv1,
    err_rv,
    model_y=planet_1_model,
    smooth_model_phase=smooth_phase1,
    smooth_model_y=smooth_model_y1,
    residuals=True,
    xlabel="Time [BJD]",
    ylabel="RV [km s-1]",
    save_folder=saving_folder,
    savefilename="final_phase_plot1",
)

smooth_model_y = get_model(models_list, xpred, model_par2, to_ecc=False)
smooth_model_end = smooth_model_y + GP_rv
plotting.data_plot(
    JD,
    rv,
    err_y=err_rv,
    smooth_model_x=xpred,
    smooth_model_y=smooth_model_end,
    model_y=model_y,
    save_folder=saving_folder,
    savefilename="final_data_plot",
)

plotting.GP_plot(
    JD,
    rv,
    err_rv,
    model_fin,
    xpred,
    GP_rv,
    GP_err,
    residuals=True,
    save_folder=saving_folder,
    savefilename="final_GP_plot",
)


file2 = os.path.join(saving_folder, "final_conditions.txt")
final_cond_file = open(file2, "w+")
final_cond_file.write("\nFinal Hyperparameters:\n")
final_cond_file.write(hparam2.__str__())
final_cond_file.write("\nFinal Parameters:\n")
final_cond_file.write(model_par2.__str__())
final_cond_file.write("\n\nFinal Log Likelihood:\n")
final_cond_file.write(logL2.__str__())
final_cond_file.close()


print()
print("Final LogL: ", logL)
print()
print("Final hyperparameters: ", hparam2)
print()
print("Final model parameters: ", model_par2)
