#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence test

Author: Federica Rescigno
Version: 17-01-2024
"""
import numpy as np
from astropy.io import ascii
from pathlib import Path

def gelman_rubin_calc(burn_in, hparameter_list, model_parameter_list):
        """
        Returns the Gelman-Rubin convergence statistic.

        Must be calculated for each parameter independently.
        """

        all_R = []

        try:
            J = np.shape(hparameter_list)[0]
            P = np.shape(hparameter_list)[1]
            L = np.shape(hparameter_list)[2] - burn_in
        except:
            import pdb

            pdb.set_trace()

        #  Calculate for hyperparams
        hp = True
        if hp:
            for hyper_param in range(P):
                chain_means = []
                intra_chain_vars = []
                for chain in range(J):
                    # Calculate chain mean
                    param_chain = hparameter_list[chain, hyper_param, burn_in:]

                    chain_means.append(np.nanmean(param_chain))
                    intra_chain_var = np.nanvar(param_chain, ddof=1)
                    intra_chain_vars.append(intra_chain_var)
                chain_means = np.array(chain_means)
                grand_mean = np.mean(chain_means)
                intra_chain_vars = np.array(intra_chain_vars)
                inter_chain_var = (
                    L / (J - 1) * np.sum(np.square(chain_means - grand_mean))
                )
                W = np.mean(intra_chain_vars)

                R = (1 - 1 / L) * W + inter_chain_var / L
                R /= W
                all_R.append(R)

        # Redefine for model params - Others should be unchanged
        P = np.shape(model_parameter_list)[1]

        #  Calculate for model_params
        for param in range(P):
            chain_means = []
            intra_chain_vars = []
            if (
                np.nanmax(model_parameter_list[:, param, :])
                - np.nanmin(model_parameter_list[:, param, :])
                == 0.0
            ):
                all_R.append(1.0)
                continue
            for chain in range(J):
                # Calculate chain mean
                param_chain = model_parameter_list[chain, param, burn_in:]

                chain_means.append(np.nanmean(param_chain))
                intra_chain_var = np.nanvar(param_chain, ddof=1)
                intra_chain_vars.append(intra_chain_var)
            chain_means = np.array(chain_means)
            grand_mean = np.mean(chain_means)
            intra_chain_vars = np.array(intra_chain_vars)
            inter_chain_var = L / (J - 1) * np.sum(np.square(chain_means - grand_mean))
            W = np.mean(intra_chain_vars)

            R = (1 - 1 / L) * W + inter_chain_var / L
            R /= W

            all_R.append(R)

        all_R = np.array(all_R)
        try:
            # assert len(all_R) == self.numb_param
            assert np.all(all_R >= 1.0)
        except:
            import pdb

            pdb.set_trace()

        return all_R



def rebuild_params(params, nparam, nchains, niter):
    params3d=np.zeros((nchains, nparam, niter))
    for i in range(nchains):
        #print(params3d[i,:,:].shape)
        #print(params[i*niter:(i+1)*niter,:].shape)
        for p in range(nparam):
            params3d[i,p,:]=params[i*niter:(i+1)*niter,p]
    return params3d


current_run='GP_3pl_corr_7x5ktesta'
myinput = Path("/home/fr307/HD48948_analysis/{}/fin_hparams.txt".format(current_run))
print('start')
HP_data = np.genfromtxt(myinput, delimiter=None)
#data = ascii.read("/home/fr307/HD48948_analysis/{}/fin_hparams.txt".format(current_run)).to_pandas()
print(HP_data.shape)
HP_data3d=rebuild_params(HP_data, 5, 700, 5000+1)
print(HP_data3d.shape)

myinput = Path("/home/fr307/HD48948_analysis/{}/fin_model_param.txt".format(current_run))
print('start 2')
MD_data = np.genfromtxt(myinput, delimiter=None)
print(MD_data.shape)
MD_data3d=rebuild_params(MD_data, 15, 700, 5000+1)
print(MD_data3d.shape)


Rs = gelman_rubin_calc(400, HP_data3d, MD_data3d)
print(Rs)