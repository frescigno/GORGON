"""
MCMC affine Base Code for multiple models
Made specifically for GP_solar_multi.py
Contains:
    Body of MCMC
    #Planetary Mass Calculator
    Running function for MCMC

Author: Federica Rescigno
Version: 20-04-2022"""

import os
import numpy as np
import random
import GP_solar_multi as gp
import math
import plotting as plot
import auxiliary as aux
import time
import mass_calc as mc


# Imports for multiprocessing
from itertools import repeat
from multiprocessing import Pool


def numb_param_per_model(model_name):
    """Function to give the number of expected parameters per model"""
    if model_name.startswith("no") or model_name.startswith("No"):
        model_param_number = 1
    if model_name.startswith("off") or model_name.startswith("Off"):
        model_param_number = 1
    if model_name.startswith("kepl") or model_name.startswith("Kepl"):
        model_param_number = 5
    if model_name.startswith("poly") or model_name.startswith("Poly"):
        model_param_number = 4
    return model_param_number


def get_model(model_name, time, model_par, to_ecc=True, flags=None):
    """
    Parameters
    ----------
    model_name : list of strings
        Name of model used
    time : array, floats
        Time array over which to calculate the model
    model_par : dictionary
        Set of parameters (within the parameter object) with which to compute the model

    Returns
    -------
    model_y : array, floats
        Radial velocity of the model
    """

    model_y = np.zeros(len(time))
    i = 0
    a = 0
    for mod in model_name:
        numb_param_mod = numb_param_per_model(mod)
        parameters = {
            key: value
            for key, value in model_par.items()
            if (
                list(model_par).index(key) >= i
                and list(model_par).index(key) < i + numb_param_mod
            )
        }
        if mod.startswith("no") or mod.startswith("No"):
            model = gp.No_Model(time, parameters)
            model_y += model.model()
        if mod.startswith("off") or mod.startswith("Off"):
            model = gp.Offset(flags, parameters)
            model_y += model.model()
        if mod.startswith("poly") or mod.startswith("Poly"):
            model = gp.Polynomial(time, parameters)
            model_y += model.model()
        if mod.startswith("kepl") or mod.startswith("Kepl"):
            if to_ecc:
                if len(model_name) == 1:
                    parameters["ecc"].value, parameters["omega"].value = aux.to_ecc(
                        parameters["ecc"].value, parameters["omega"].value
                    )
                else:
                    (
                        parameters["ecc_" + str(a)].value,
                        parameters["omega_" + str(a)].value,
                    ) = aux.to_ecc(
                        parameters["ecc_" + str(a)].value,
                        parameters["omega_" + str(a)].value,
                    )
            # print('check', parameters)
            model = gp.Keplerian(time, parameters)
            model_y += model.model()
            a += 1
        i += numb_param_mod

    parameter = None
    return model_y


def parameter_check(parameters, names, Rstar=None, Mstar=None):
    """Function to check if the parameters are within the bounds

    Parameters
    ----------
    parameters : array, floats
        Array of parameters for all models
    names : list of strings
        Names of all the models used, can be one or more
    Rstar : float, optional
        Radius of the star in solar radii. Default is None. Needed for the orbit check
    Mstar : float, optional
        Mass of the star in solar masses. Default is None. Needed for the orbit check

    Returns
    -------
    check : bool
        Are all paramaters physically possible?
    """
    check = True
    o = 0
    for name in names:
        numb_params = numb_param_per_model(name)

        if name.lower().startswith("kepl"):
            # Period, amplitude and t0 must be positive
            if (
                parameters[o] < 0.0
                or parameters[o + 1] < 0.0
                or parameters[o + 4] < 0.0
            ):
                check = False
                return check

            # Sk and Ck can be negative, but the sum of their squares must be less than 1
            if (
                parameters[o + 2] ** 2 + parameters[o + 3] ** 2
            ) > 1.0:  # or (parameters[o+2] < 0.) or (parameters[o+3] < 0.):
                check = False
                return check

            if Rstar is not None and Mstar is not None:
                # Check planet does not fall into the star
                ####### NOW COLLED STAR CROSS!!!!!
                orbit_check = aux.orbit_check(
                    parameters[o + 2], parameters[o + 3], Rstar, parameters[o], Mstar
                )
                if not orbit_check:
                    check = False
                    return check

        o += numb_params

    return check


def calculate_move(
    t,
    rv,
    rv_err,
    hparams,
    mparams,
    logL,
    hparams_out,
    mparams_out,
    kernel_name,
    model_name,
    a,
    hp_err,
    mp_err,
    hp_vary,
    mp_vary,
    prior_list,
    flags=None,
    Rstar=None,
    Mstar=None,
):
    np.random.seed()
    res_hp = []
    res_mp = []
    res_logL = []
    res_acc = []
    for idx in range(len(hparams)):
        hp0 = hparams[idx]
        mp0 = mparams[idx]

        randint = np.random.choice(range(len(hparams_out)))

        rand_hp = hparams_out[randint]
        rand_mp = mparams_out[randint]

        z = (np.random.rand() * (a - 1) + 1) ** 2 / a

        new_hp = rand_hp + z * (hp0 - rand_hp)
        new_mp = rand_mp + z * (mp0 - rand_mp)

        if np.any(new_hp < 0.) or (
            not parameter_check(new_mp, model_name, Rstar, Mstar)
        ):
            # Are there any other conditions that should be checked?
            res_hp.append(hp0)
            res_mp.append(mp0)
            res_logL.append(logL[idx])
            res_acc.append(False)
            continue

        param = gp.Par_Creator.create(kernel_name)

        for j, key in enumerate(param):
            param[key] = gp.Parameter(
                value=new_hp[j],
                error=hp_err[j],
                vary=hp_vary[j],
            )

        Model_Par_Creator = gp.Model_Par_Creator()
        model_param = Model_Par_Creator.create(model_name)

        for j, key in enumerate(model_param):
            model_param[key] = gp.Parameter(
                value=new_mp[j],
                error=mp_err[j],
                vary=mp_vary[j],
            )

        model_y = get_model(model_name, t, model_param, flags=flags)

        likelihood = gp.GPLikelyhood(
            t,
            rv,
            model_y,
            rv_err,
            param,
            model_param,
            kernel_name,
        )
        logL_chain = likelihood.LogL(prior_list)

        diff_logL_z = logL_chain - logL[idx] + np.log(z) * (len(hp0) + len(mp0) - 1)

        logr = np.log(np.random.rand())
        if diff_logL_z > 0 or logr <= diff_logL_z:
            # Update values
            res_hp.append(new_hp)
            res_mp.append(new_mp)
            res_logL.append(logL_chain)
            res_acc.append(True)
            # self.hp0[idx] = new_hp
            # self.modpar0[idx] = new_mp
            # self.logL0[idx] = logL_chain
            # self.accepted0[idx] = True
        else:
            res_hp.append(hp0)
            res_mp.append(mp0)
            res_logL.append(logL[idx])
            res_acc.append(False)
    assert len(res_acc) == len(hparams)
    return np.array(res_hp), np.array(res_mp), np.array(res_logL), np.array(res_acc)


class MCMC:
    def __init__(
        self,
        t,
        rv,
        rv_err,
        hparam0,
        kernel_name,
        model_par0,
        model_name,
        prior_list,
        numb_chains=100,
        a=1.2,
        flags=None,
        mass=False,
        numb_cores=1,
    ):
        """
        Parameters
        ----------
        t : array, floats
            Time array of the observations
        rv : array, floats
            Radial velocity array of the observations
        rv_err : array, floats
            Errors on the radial velocity array
        hparam0 : dictionary
            Set of hyper-parameters for the kernel, with value, error and vary
        kernel_name : string
            Name of chosen kernel
        model_par0 : dictionary
            Set of parameters for the model, with value, error and vary
        model_name : string
            Name of chosen model
        prior_list : list
            List inlcuding in order the name of parameter, the name of the prior and the dictionary of the prior parameters (see posterior function in GP_solar.py for better description)
        numb_chains : integer, optional
            Number of chains requested. The default is 100.
        """

        # Save unchanging inputs
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.kernel_name = kernel_name
        self.model_name = model_name

        self.a = a

        # a is used as an interator later in _init_
        del a

        if flags is not None:
            self.flags = flags
        else:
            self.flags = None
        # Check how many models are included
        if isinstance(self.model_name, list):
            self.numb_models = len(self.model_name)
        elif isinstance(self.model_name, str):
            self.numb_models = 1
            # Put it into list for easy computation in loops
            self.model_name = [model_name]
        else:
            raise TypeError("model_name must be a string or a list of strings")
        self.prior_list = prior_list
        self.numb_chains = int(numb_chains)

        # For parallelisation
        if not isinstance(numb_cores, int):
            self.numb_cores = 1
        else:
            self.numb_cores = numb_cores

        # Set up storing arrays
        self.hparameter_list = []
        self.model_parameter_list = []
        self.logL_list = []
        # self.accepted = []
        self.accepted = np.array([[True] for i in range(self.numb_chains)])
        self.mass = mass
        if self.mass:
            self.mass0_list = []
            self.mass1_list = []


        # Get initial guesses for hyperparameters and save them as 0
        # Save also errors for chains initial positions population
        self.single_hp0 = []
        self.hp_err = []
        self.hp_vary = []
        for key in hparam0.keys():
            self.single_hp0.append(hparam0[key].value)
            self.hp_err.append(hparam0[key].error)
            self.hp_vary.append(hparam0[key].vary)
        # Do the same for model parameters
        self.single_modpar0 = []
        self.modpar_err = []
        self.modpar_vary = []
        self.modpar_names = []
        # Also include a list of the names of the parameters, in the proper order (useful when have multiple models)
        for key in model_par0.keys():
            self.single_modpar0.append(model_par0[key].value)
            self.modpar_err.append(model_par0[key].error)
            self.modpar_vary.append(model_par0[key].vary)
            self.modpar_names.append(key)
        # Extend self.modpar_names to inlcude a 2nd row with info on which model do they belong to and a 3rd with the name of the model
        which_model = []
        name = []
        b = 0
        for mod in self.model_name:
            a = numb_param_per_model(mod)
            for a in range(a):
                which_model.append(b)
                name.append(mod + str(b))
            b += 1
        self.modpar_info = [[], [], []]
        self.modpar_info[0] = self.modpar_names
        self.modpar_info[1] = which_model
        self.modpar_info[2] = name

        # Save number of parameters
        self.k_numb_param = len(self.single_hp0)
        self.mod_numb_param = len(self.single_modpar0)
        self.numb_param = len(self.single_hp0) + len(self.single_modpar0)

        
        self.idx_splits0 = None
        self.idx_splits1 = None
        # If model is keplerian, substitute eccentricity and omega with Sk and Ck
        # Ford et al. 2006
        # ecc = Sk^2 + Ck^2
        # omega = arctan(Sk/Ck)
        # Sk = sqrt(ecc) * sin(omega), Ck = sqrt(ecc) * cos(omega)
        for g in range(len(self.single_modpar0)):
            if self.modpar_info[0][g].startswith("ecc") and (
                self.modpar_info[2][g].startswith("kepl")
                or self.modpar_info[2][g].startswith("Kepl")
            ):
                (
                    self.single_modpar0[g],
                    self.single_modpar0[g + 1],
                    self.modpar_err[g],
                    self.modpar_err[g + 1],
                ) = aux.to_SkCk(
                    self.single_modpar0[g],
                    self.single_modpar0[g + 1],
                    self.modpar_err[g],
                    self.modpar_err[g + 1],
                )

        # Check that you have a reasonable number of chains
        if self.numb_chains < (len(self.single_hp0) + len(self.single_modpar0)) * 2:
            return RuntimeWarning(
                "It is not advisable to conduct this analysis with less chains than double the amount of free parameters"
            )

        # Populate the positions based on those values for the starting point of all the chains
        self.hp0 = aux.initial_pos_creator(
            self.single_hp0, self.hp_err, self.numb_chains
        )
        self.modpar0 = aux.initial_pos_creator(
            self.single_modpar0, self.modpar_err, self.numb_chains
        )  # , param_names=self.modpar_info[0])

        # Append these first guesses (and their chains) to the storing arrays (check for right shape)
        self.hparameter_list.append(self.hp0)
        self.model_parameter_list.append(self.modpar0)
        self.hparameter_list = np.array(self.hparameter_list[0])
        self.hparameter_list.tolist()
        self.model_parameter_list = np.array(self.model_parameter_list[0])
        self.model_parameter_list.tolist()
        # print("check shape", self.hparameter_list)
        row = len(self.hparameter_list)
        column = len(self.hparameter_list[0])
        row = len(self.model_parameter_list)
        column = len(self.model_parameter_list[0])
        # Expected output: 2d array with nrows=numb_chains, ncols=number of hyperparameters
        # We will then use np.concatenate(a.b) to make it into a 3d array with ndepth=steps

        self.logL0 = np.array([])
        if self.mass:
            self.mass0_0 = []
            self.mass1_0 = []

        # Start looping over all chains to get initial models and logLs
        for chain in range(self.numb_chains):
            # Pick each row one at a time
            hp_chain = self.hp0[chain]
            modpar_chain = self.modpar0[chain]

            # Re-create parameter objects, kernel and model
            hparam_chain = gp.Par_Creator.create(self.kernel_name)
            for i, key in zip(range(len(hp_chain)), hparam_chain.keys()):
                hparam_chain[key] = gp.Parameter(
                    value=hp_chain[i], error=self.hp_err[i], vary=self.hp_vary[i]
                )
            Model_Par_Creator = gp.Model_Par_Creator()
            model_par_chain = Model_Par_Creator.create(self.model_name)
            for i, key in zip(range(len(modpar_chain)), model_par_chain.keys()):
                model_par_chain[key] = gp.Parameter(
                    value=modpar_chain[i],
                    error=self.modpar_err[i],
                    vary=self.modpar_vary[i],
                )

            if self.mass:
                mass0_chain = mc.mass_calc(
                    model_par_chain["P_0"].value,
                    model_par_chain["K_0"].value,
                    model_par_chain["omega_0"].value,
                    model_par_chain["ecc_0"].value,
                    0.743,
                )
                mass1_chain = mc.mass_calc(
                    model_par_chain["P_1"].value,
                    model_par_chain["K_1"].value,
                    model_par_chain["omega_1"].value,
                    model_par_chain["ecc_1"].value,
                    0.743,
                )
                # print("check", mass1_chain)
                # print(model_par_chain["P_1"].value, model_par_chain["K_1"].value, model_par_chain["omega_1"].value, model_par_chain["ecc_1"].value)
                self.mass0_0.append(mass0_chain)
                self.mass1_0.append(mass1_chain)

            # Compute model y as sum of models
            if flags is None:
                self.model_y0 = get_model(self.model_name, self.t, model_par_chain)
            if flags is not None:
                self.model_y0 = get_model(
                    self.model_name, self.t, model_par_chain, flags=self.flags
                )

            # for priors in ecc and omega need to go back momentarely

            self.likelyhood = gp.GPLikelyhood(
                self.t,
                self.rv,
                self.model_y0,
                self.rv_err,
                hparam_chain,
                model_par_chain,
                self.kernel_name,
            )
            logL_chain = self.likelyhood.LogL(self.prior_list)

            # logL is initial likelihood of this chain, append it to logL0 of all chains
            # self.logL0.append(logL_chain)

            self.logL0 = np.append(self.logL0, logL_chain)

            #### ATTENTION!!!! FOR SOME REASON AFTER GET_MODEL THE VALUES IN MODEL_PAR_CHAIN BECOME OMEGA AND ECCENTRICITY
            if self.mass:
                mass0_chain, mass1_chain = None, None
            hparam_chain, model_par_chain, logL_chain = None, None, None
        # Save this set of likelihoods as the first ones of the overall likelihood array
        self.logL_list.append(self.logL0)
        acc = np.array([True for i in range(self.numb_chains)])
        # self.accepted.append(acc)
        # self.accepted = np.append(self.accepted, acc)
        self.logL_list = aux.transpose(self.logL_list)
        # self.accepted = aux.transpose(self.accepted)
        if self.mass:
            self.mass0_list.append(self.mass0_0)
            self.mass0_list = aux.transpose(self.mass0_list)
            self.mass1_list.append(self.mass1_0)
            self.mass1_list = aux.transpose(self.mass1_list)

        # Verbose
        print("Initial hyper-parameter guesses: ")
        print(self.single_hp0)
        print()
        print(
            "Initial model parameter guesses (ecc and omega are replaced by Sk and Ck): "
        )
        print(self.single_modpar0)
        print()
        print("Initial Log Likelihood: ", self.logL0[0])
        print()
        print("Number of chains: ", self.numb_chains)
        print()

    def update_chains(self, pos_i, out_pos_m, out_pos_hp):
        pos_f = {}
        log_z_f = {}
        for idx, pos in pos_i.items():
            pos_m = pos[0]
            pos_hp = pos[1]

            # new_pos_m = out_pos_m[idx]

            rint = random.randint(0, len(out_pos_m) - 1)
            X_m = out_pos_m[rint]
            # Use same chain for model parameters as well
            X_hp = out_pos_hp[rint]
            # Compute the step and apply it
            # The random number is the same within a chain but different between chains

            # TODO: make a an attribute of the class
            # z = (np.random.rand() * (a - 1) + 1) ** 2 / a

            # new_pos_m = X_m + (pos_m - X_m) * z

            # new_pos_hp = X_hp + (pos_hp - X_hp) * z

            # TODO: make Rstar and Mstar attributes of the class

            # if self.Rstar is not None and self.Mstar is not None:
            #     model_param_check = parameter_check(
            #         Xk_new_mod, self.model_name, Rstar, Mstar
            #     )
            # else:
            #     model_param_check = parameter_check(Xk_new_mod, self.model_name)

            model_param_check = False
            while (np.min(new_pos_hp) < 0) or (not model_param_check):
                # FIXME: I think this viloates MCMC step assumptions...

                # Compute the step and apply it
                # The random number is the same within a chain but different between chains
                z = (np.random.rand() * (self.a - 1) + 1) ** 2 / self.a
                new_pos_m = X_m + (pos_m - X_m) * z
                new_pos_hp = X_hp + (pos_hp - X_hp) * z

                model_param_check = parameter_check(
                    new_pos_m, self.model_name, self.Rstar, self.Mstar
                )
                log_z = np.log(z)

                # Save as the new value for the chain, by finding where the initial step was positioned and putting it in the same position

                # for o in range(self.numb_chains):
                #     # Kernel and model parameters are still kept together after shuffling
                #     if (self.hp0[o][0] == S1_hp[chain][0]) and (
                #         self.modpar0[o][0] == S1_mod[chain][0]
                #     ):
                #         position = o

                # print(position)

            # for v in range(len(self.hp_vary)):
            #     if self.hp_vary[v]:
            #         self.hp[idx][v] = new_pos_hp[v]
            #     else:
            #         self.hp[idx][v] = pos_hp[v]
            # for v in range(len(self.modpar_vary)):
            #     if self.modpar_vary[v]:
            #         self.modpar[idx][v] = new_pos_m[v]
            #         # print("nope")
            #     else:
            #         self.modpar[idx][v] = pos_m[v]
            #     self.logz[idx] = log_z

            pos_f[idx] = [new_pos_m, new_pos_hp]
            log_z_f[idx] = log_z
        return (pos_f, log_z_f)

    def split_step(self, n_splits=2, a=2.0, Rstar=None, Mstar=None):
        """
        self, n_splits=2, a=1.25, Rstar=None, Mstar=None

        Parameters
        ----------
        n_splits : integer, optional
            Number of subsplits of the total number of chains. The default is 2.
        a : float, optional
            Adjustable scale parameter. The default is 1.25.
        Rstar : float, optional
            Radius of the host star in solar radii. The default is None.
        Mstar : float, optional
            Mass of the star in solar masses. The default is None.
        """

        # Split the chains into subgroups (Foreman-Mackey et al. 2013)
        all_inds = np.arange(self.numb_chains)
        random.shuffle(all_inds)
        # Use modulo operator (think clock)
        inds = all_inds % n_splits

        # Shuffle to allow for flexible statistics - I don't think this shuffles anything

        # Create empty array for the new steps
        self.hp = np.zeros_like(self.hp0)
        self.modpar = np.zeros_like(self.modpar0)
        self.logz = np.zeros(self.numb_chains)

        # Create empty arrays for considered split part = 1, and the rest = 2
        S1_hp = []
        S2_hp = []

        S1_mod = []
        S2_mod = []

        # Repeat how many splits you want
        for split in range(n_splits):
            # Do the separation
            # This is where we parallelise
            # for walker in range(len(inds)):
            #     if inds[walker] == split:
            #         S1_hp.append(self.hp0[walker])
            #         S1_mod.append(self.modpar0[walker])
            #     else:
            #         S2_hp.append(self.hp0[walker])
            #         S2_mod.append(self.modpar0[walker])

            S1_hp = self.hp0[inds == split]
            S2_hp = self.hp0[inds != split]
            S1_mod = self.modpar0[inds == split]
            S2_mod = self.modpar0[inds != split]

            inds1 = all_inds[inds == split]
            inds2 = all_inds[inds != split]

            # S1_hp = np.array(S1_hp)
            # S2_hp = np.array(S2_hp)
            # S1_mod = np.array(S1_mod)
            # S2_mod = np.array(S2_mod)
            N_chains1 = len(S1_hp)
            N_chains2 = len(S2_hp)

            row = len(S1_hp)
            column = len(S1_hp[0])
            # print("row = {}, column = {}".format(row, column))

            # Start looping over all chains in the set
            # for chain in range(N_chains1):

            # TODO: n_cores propagate through functions
            # TODO: Produce {idx: [new_pos_m, new_pos_hp]} objects

            n_cores = 1
            if n_cores == 1:
                pos_i = {inds1[j]: [S1_hp[j], S1_mod[j]] for j in range(N_chains1)}
                # Collate all the chains into a single pos_i
                pos_f, log_z_f = self.update_chains(
                    # Put in parameters here...
                )

            else:
                split_idxs = np.array_split(np.arange(N_chains1), n_cores)
                pos_f = {}
                log_z_f = {}

                for core in n_cores:
                    pos_i = {j: [S1_hp[j], S1_mod[j]] for j in split_idxs[core]}

                    (this_pos, this_log_z) = self.update_chains(
                        # Put in parameters here...
                    )

                    pos_f.update(this_pos)
                    log_z_f.update(this_log_z)

            # for o in range(self.numb_chains):
            #     # Kernel and model parameters are still kept together after shuffling
            #     if (self.hp0[o][0] == S1_hp[chain][0]) and (
            #         self.modpar0[o][0] == S1_mod[chain][0]
            #     ):
            #         position = o

            # print(position)

            for position in pos_f.keys():
                for v in range(len(self.modpar_vary)):
                    if self.modpar_vary[v]:
                        self.modpar[position][v] = pos_f[position][0][v]
                        # print("nope")
                    else:
                        self.modpar[position][v] = pos_i[position][0][v]
                for v in range(len(self.hp_vary)):
                    if self.hp_vary[v]:
                        self.hp[position][v] = pos_f[0][v]
                    else:
                        self.hp[position][v] = pos_i[position][0][v]
                self.logz[position] = log_z_f[position]

            # Once all elements of the first section have taken the step, reinitialise the S1,2 arrays
            # print(len(S1_mod), len(S2_mod))

            S1_hp = []
            S2_hp = []

            S1_mod = []
            S2_mod = []

            # print("hp", self.hp)

        # print(self.modpar)
        # print()
        # print()
        # Final result is self.hp which should be a 2d array, nrow=numb chains, ncol=numb parameters
        # print("full z", self.logz)

    def compute(self):
        """ """
        self.logL = []
        self.mass0 = []
        self.mass1 = []
        # print("hp", self.hp)
        # print("select", self.hp[0][1])

        # Start by going chain by chain
        for chain in range(self.numb_chains):
            # print("chain", chain)
            # Reset dictionary for the kernel hyperparameters and store new values
            param = None
            param = gp.Par_Creator.create(self.kernel_name)
            for i, key in zip(range(self.k_numb_param), param.keys()):
                param[key] = gp.Parameter(
                    value=self.hp[chain][i], error=self.hp_err[i], vary=self.hp_vary[i]
                )

            # Do the same for the model parameters
            model_param = None
            Model_Par_Creator = gp.Model_Par_Creator()
            model_param = Model_Par_Creator.create(self.model_name)

            for i, key in zip(range(self.mod_numb_param), model_param.keys()):
                model_param[key] = gp.Parameter(
                    value=self.modpar[chain][i],
                    error=self.modpar_err[i],
                    vary=self.modpar_vary[i],
                )

            if self.mass:
                # print(model_param["omega_0"].value, model_param["ecc_0"].value, model_param["omega_1"].value, model_param["ecc_1"].value)
                mass0_chain = mc.mass_calc(
                    model_param["P_0"].value,
                    model_param["K_0"].value,
                    model_param["omega_0"].value,
                    model_param["ecc_0"].value,
                    0.743,
                )
                mass1_chain = mc.mass_calc(
                    model_param["P_1"].value,
                    model_param["K_1"].value,
                    model_param["omega_1"].value,
                    model_param["ecc_1"].value,
                    0.743,
                )
                self.mass0.append(mass0_chain)
                self.mass1.append(mass1_chain)

            # print("param", param)
            # print("par model", model_param)
            # print("hyperpar", param)

            # self.model_y = None
            # Get new model
            if self.flags is None:
                self.model_y = get_model(
                    self.model_name, self.t, model_param, flags=None
                )
            if self.flags is not None:
                self.model_y = get_model(
                    self.model_name, self.t, model_param, flags=self.flags
                )
            # print("model", self.model_y[0:5])

            # print(model_param)
            # For some reason after going through get model we get the ecc and omega instead???

            self.likelyhood = None
            # logL_chain = None
            # Use current hp and model to compute the logL
            self.likelyhood = gp.GPLikelyhood(
                self.t,
                self.rv,
                self.model_y,
                self.rv_err,
                param,
                model_param,
                self.kernel_name,
            )
            logL_chain = self.likelyhood.LogL(self.prior_list)
            # print("h", logL_chain)

            self.logL.append(logL_chain)

        # Final output: a logL 2d array, ncols = 1, nrows = numb_chains

    def compare(self):
        # Create empty array to save decisions in to then concatenate to the list arrays
        hp_decision = []
        modpar_decision = []
        logL_decision = []
        self.acceptance_chain = []
        if self.mass:
            mass0_decision = []
            mass1_decision = []

        # Loop over all chains
        for chain in range(self.numb_chains):
            # print(chain)
            # print("z", self.logz[chain])
            # print("logL", self.logL[chain])
            # print("logL0", self.logL0[chain])
            # Compute the difference between the current and the previous likelyhood (include affine invariant normalisation)

            diff_logL_z = (
                self.logL[chain]
                - self.logL0[chain]
                + self.logz[chain] * (self.numb_param - 1)
            )
            # print(self.logL[chain])
            # diff_Lz = (np.exp(self.logz[chain]))**(self.numb_param - 1) * (np.exp(self.logL[chain])/np.exp(self.logL0[chain]))
            # print("diff", diff_Lz)

            hp = self.hp
            hp0 = self.hp0
            modpar = self.modpar
            modpar0 = self.modpar0

            # If the logL is larger than one, then it's exponent will definitely be larger than 1 and will automatically be accepted
            if diff_logL_z > 1:
                logL_decision.append(self.logL[chain])
                hp_decision.append(hp[chain])
                modpar_decision.append(modpar[chain])
                self.acceptance_chain.append(True)
                if self.mass:
                    mass0_decision.append(self.mass0[chain])
                    mass1_decision.append(self.mass1[chain])

            # If the logL is ver small (eg. smaller than -35), automatic refusal
            if diff_logL_z < -35.0:
                logL_decision.append(self.logL0[chain])
                hp_decision.append(hp0[chain])
                modpar_decision.append(modpar0[chain])
                self.acceptance_chain.append(False)
                if self.mass:
                    mass0_decision.append(self.mass0_0[chain])
                    mass1_decision.append(self.mass1_0[chain])

            if (diff_logL_z >= -35.0) and (diff_logL_z <= 1.0):
                # Generate random number from uniform distribution
                MH_rand = random.uniform(0, 1)
                # if diff_Lz is smaller than the number, accept the step
                if MH_rand <= (np.exp(diff_logL_z)):
                    logL_decision.append(self.logL[chain])
                    hp_decision.append(hp[chain])
                    modpar_decision.append(modpar[chain])
                    self.acceptance_chain.append(True)
                    if self.mass:
                        mass0_decision.append(self.mass0[chain])
                        mass1_decision.append(self.mass1[chain])
                # if it is larger than the number reject the step
                else:
                    logL_decision.append(self.logL0[chain])
                    hp_decision.append(hp0[chain])
                    modpar_decision.append(modpar0[chain])
                    self.acceptance_chain.append(False)
                    if self.mass:
                        mass0_decision.append(self.mass0_0[chain])
                        mass1_decision.append(self.mass1_0[chain])

        # Now concatenate all the 2D arrays into the 3D list arrays
        # Start with logL list and append, nrows = nchains, ncols = niterations
        self.logL_list = np.column_stack((self.logL_list, logL_decision))
        if self.mass:
            self.mass0_list = np.column_stack((self.mass0_list, mass0_decision))
            self.mass1_list = np.column_stack((self.mass1_list, mass1_decision))
        self.accepted = np.column_stack((self.accepted, self.acceptance_chain))
        # Rest of lists, nrows = nchains, ncols = nparam, ndepth = niterations
        self.hparameter_list = np.dstack((self.hparameter_list, hp_decision))
        self.model_parameter_list = np.dstack(
            (self.model_parameter_list, modpar_decision)
        )
        # print("before", hp_decision)
        # print("after", self.hparameter_list)

    def reset(self):
        """
        Returns
        -------
        logL_list : 3D array
            List of the logL of all accepted steps, nrow = n_chains, ncol = 1, ndep = iterations
        hparameters_list : 3D array
            List of the hyperparameters of all accepted steps, nrow = n_chains, ncol = n_parameters, ndep = iterations
        modelel_paramater_list : 3D array
            List of the model parameters of all accepted steps, nrow = n_chains, ncol = n_parameters, ndep = iterations
        accepted : 3D array
            List of True and Falses regarding the acceptance of each MCMC step, nrow = n_chains, ncol = 1, ndep = iterations
        """
        # Don't need burn in

        # Set the zero values for nex step
        for chain in range(self.numb_chains):
            if self.acceptance_chain[chain] is True:
                self.logL0[chain] = self.logL[chain]
                self.hp0[chain] = self.hp[chain]
                self.modpar0[chain] = self.modpar[chain]
                if self.mass:
                    self.mass0_0[chain] = self.mass0[chain]
                    self.mass1_0[chain] = self.mass1[chain]
            if self.acceptance_chain[chain] is False:
                self.logL0[chain] = self.logL0[chain]
                self.hp0[chain] = self.hp0[chain]
                self.modpar0[chain] = self.modpar0[chain]
                if self.mass:
                    self.mass0_0[chain] = self.mass0_0[chain]
                    self.mass1_0[chain] = self.mass1_0[chain]

        # IMPORTANT!! In model_parameter_list, if the model is a keplerian we have Sk and Ck, not ecc and omega

        if self.mass:
            return (
                self.logL_list,
                self.hparameter_list,
                self.model_parameter_list,
                self.accepted,
                self.mass0_list,
                self.mass1_list,
            )
        else:
            return (
                self.logL_list,
                self.hparameter_list,
                self.model_parameter_list,
                self.accepted,
            )

    def gelman_rubin_calc(self, burn_in):
        """
        Returns the Gelman-Rubin convergence statistic.

        Must be calculated for each parameter independently.
        """

        all_R = []

        try:
            J = np.shape(self.hparameter_list)[0]
            P = np.shape(self.hparameter_list)[1]
            L = np.shape(self.hparameter_list)[2] - burn_in
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
                    param_chain = self.hparameter_list[chain, hyper_param, burn_in:]

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
        P = np.shape(self.model_parameter_list)[1]

        #  Calculate for model_params
        for param in range(P):
            chain_means = []
            intra_chain_vars = []
            if (
                np.nanmax(self.model_parameter_list[:, param, :])
                - np.nanmin(self.model_parameter_list[:, param, :])
                == 0.0
            ):
                all_R.append(1.0)
                continue
            for chain in range(J):
                # Calculate chain mean
                param_chain = self.model_parameter_list[chain, param, burn_in:]

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

    def ben_update(self):
        self.accepted0 = np.array([False] * self.numb_chains)

        idx0 = np.arange(0, self.numb_chains // 2)
        idx1 = np.arange(self.numb_chains // 2, self.numb_chains)

        if self.numb_cores == 1:
            self.out_ensemble_idxs = idx1
            # new_hp0, new_mp0, new_logL0, accepted0 = self._ben_bundle_calc_lighter(idx0)
            new_hp0, new_mp0, new_logL0, accepted0 = calculate_move(
                self.t,
                self.rv,
                self.rv_err,
                self.hp0[idx0],
                self.modpar0[idx0],
                self.logL0[idx0],
                self.hp0[idx1],
                self.modpar0[idx1],
                self.kernel_name,
                self.model_name,
                self.a,
                self.hp_err,
                self.modpar_err,
                self.hp_vary,
                self.modpar_vary,
                self.prior_list,
                self.flags,
            )

            self.hp0[idx0] = new_hp0
            self.modpar0[idx0] = new_mp0
            self.logL0[idx0] = new_logL0
            self.accepted0[idx0] = accepted0

            self.out_ensemble_idxs = idx0
            # new_hp1, new_mp1, new_logL1, accepted1 = self._ben_bundle_calc_lighter(idx1)

            new_hp1, new_mp1, new_logL1, accepted1 = calculate_move(
                self.t,
                self.rv,
                self.rv_err,
                self.hp0[idx1],
                self.modpar0[idx1],
                self.logL0[idx1],
                self.hp0[idx0],
                self.modpar0[idx0],
                self.kernel_name,
                self.model_name,
                self.a,
                self.hp_err,
                self.modpar_err,
                self.hp_vary,
                self.modpar_vary,
                self.prior_list,
                self.flags,
            )

            self.hp0[idx1] = new_hp1
            self.modpar0[idx1] = new_mp1
            self.logL0[idx1] = new_logL0
            self.accepted0[idx1] = accepted1

        else:
            if self.idx_splits0 is None:
                self.idx_splits0 = np.array_split(idx0, self.numb_cores)
                self.idx_splits1 = np.array_split(idx1, self.numb_cores)

            self.out_ensemble_idxs = idx1

            arguments = zip(
                repeat(self.t),
                repeat(self.rv),
                repeat(self.rv_err),
                [self.hp0[i] for i in self.idx_splits0],
                [self.modpar0[i] for i in self.idx_splits0],
                [self.logL0[i] for i in self.idx_splits0],
                repeat(self.hp0[idx1]),
                repeat(self.modpar0[idx1]),
                repeat(self.kernel_name),
                repeat(self.model_name),
                repeat(self.a),
                repeat(self.hp_err),
                repeat(self.modpar_err),
                repeat(self.hp_vary),
                repeat(self.modpar_vary),
                repeat(self.prior_list),
                repeat(self.flags),
            )

            with Pool(self.numb_cores) as p:
                # results0 = p.map(self._ben_bundle_calc_lighter, self.idx_splits0)
                results0 = p.starmap(calculate_move, arguments)

            for i in range(self.numb_cores):
                self.hp0[self.idx_splits0[i]] = results0[i][0]
                self.modpar0[self.idx_splits0[i]] = results0[i][1]
                self.logL0[self.idx_splits0[i]] = results0[i][2]
                self.accepted0[self.idx_splits0[i]] = results0[i][3]

            self.out_ensemble_idxs = idx0

            arguments = zip(
                repeat(self.t),
                repeat(self.rv),
                repeat(self.rv_err),
                [self.hp0[i] for i in self.idx_splits1],
                [self.modpar0[i] for i in self.idx_splits1],
                [self.logL0[i] for i in self.idx_splits1],
                repeat(self.hp0[idx0]),
                repeat(self.modpar0[idx0]),
                repeat(self.kernel_name),
                repeat(self.model_name),
                repeat(self.a),
                repeat(self.hp_err),
                repeat(self.modpar_err),
                repeat(self.hp_vary),
                repeat(self.modpar_vary),
                repeat(self.prior_list),
                repeat(self.flags),
            )
            with Pool(self.numb_cores) as p:
                # results1 = p.map(self._ben_bundle_calc_lighter, self.idx_splits1)
                results1 = p.starmap(calculate_move, arguments)

            for i in range(self.numb_cores):
                self.hp0[self.idx_splits1[i]] = results1[i][0]
                self.modpar0[self.idx_splits1[i]] = results1[i][1]
                self.logL0[self.idx_splits1[i]] = results1[i][2]
                self.accepted0[self.idx_splits1[i]] = results1[i][3]

        self.hparameter_list = np.dstack((self.hparameter_list, self.hp0))
        self.model_parameter_list = np.dstack((self.model_parameter_list, self.modpar0))
        self.logL_list = np.column_stack((self.logL_list, self.logL0))
        self.accepted = np.column_stack((self.accepted, self.accepted0))
        # print(type(self.accepted))


def run_MCMC(
    iterations,
    t,
    rv,
    rv_err,
    hparam0,
    kernel_name,
    model_param0,
    model_name,
    prior_list,
    numb_chains=None,
    n_splits=2,
    a=1.2,
    Rstar=None,
    Mstar=None,
    flags=None,
    plot_convergence=False,
    saving_folder=None,
    mass=False,
):
    # from MCMC_affine_multi import MCMC

    gelman_rubin_limit = 1.1

    if numb_chains is None:
        numb_chains = 100

    _ = MCMC(
        t,
        rv,
        rv_err,
        hparam0,
        kernel_name,
        model_param0,
        model_name,
        prior_list,
        numb_chains,
        flags,
        mass,
    )

    # if mass:
    #     if flags is None:
    #         if numb_chains is None:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, mass=True)
    #             numb_chains=100
    #         else:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, mass=True)
    #     if flags is not None:
    #         if numb_chains is None:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, mass=True)
    #             numb_chains=100
    #         else:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, mass=True)
    # else:
    #     if flags is None:
    #         if numb_chains is None:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list)
    #             numb_chains=100
    #         else:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains)
    #     if flags is not None:
    #         if numb_chains is None:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags)
    #             numb_chains=100
    #         else:
    #             _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags)

    # Initialise progress bar
    print("Start Iterations")
    print()

    start = time.time()
    burn_in = min(200, iterations // 10)

    if plot_convergence:
        import matplotlib.pyplot as plt
        import seaborn as sns

        conv_f, ax = plt.subplots(figsize=(15,15))
        hparam_names = aux.hparam_names(kernel_name)
        model_param_names = aux.model_param_names(model_name)
        all_param_names = hparam_names + model_param_names
        #all_param_names = model_param_names
        conv_vals = {i: [] for i in all_param_names}
        conv_iters = []
        ax.set_prop_cycle("color", sns.color_palette("plasma",len(all_param_names)))

        if saving_folder is None:

            saving_folder = os.getcwd()

    # aux.printProgressBar(0, iterations, length = 50)

    for iteration in range(iterations):
        # if n_splits is None and a is None:
        #     _.split_step()
        # elif n_splits is not None and a is None:
        #     _.split_step(n_splits=n_splits)
        # elif n_splits is None and a is not None:
        #     _.split_step(a=a)
        # elif n_splits is not None and a is not None:
        #     _.split_step(n_splits=n_splits, a=a)

        _.split_step(n_splits=n_splits, a=a)

        _.compute()

        # if Rstar is not None and Mstar is not None:
        #     _.compare(Rstar=Rstar, Mstar=Mstar)
        # elif Rstar is None or Mstar is None:
        #     _.compare()

        _.compare(Rstar=Rstar, Mstar=Mstar)

        if mass:
            (
                logL_list,
                hparameter_list,
                model_parameter_list,
                accepted,
                mass0,
                mass1,
            ) = _.reset()
        else:
            logL_list, hparameter_list, model_parameter_list, accepted = _.reset()
        if (iteration % 2 == 0) or iteration == iterations - 1:
            aux.printProgressBar(iteration, iterations - 1, length=50)

        if plot_convergence:
            if (iteration >= burn_in) and (iteration - burn_in) % 20 == 0:
                # Check whether the chains have converged...
                R_list = _.gelman_rubin_calc(burn_in)
                # print(iteration)
                if plot_convergence:
                    assert len(R_list) == len(conv_vals)
                    for i, R in enumerate(R_list):
                        conv_vals[all_param_names[i]].append(R)
                    conv_iters.append(iteration)

                    # for param in conv_vals:
                    #     ax.plot(conv_iters, conv_vals[param], label=param)
                    # ax.legend()
                    # ax.axhline(gelman_rubin_limit, c='k', ls='--')
                    # ax.set_xlim(left=burn_in)
                    # ax.set_ylim(bottom = 1.)
                    # ax.set_yscale('log')
                    # conv_f.savefig(saving_folder+'/convergence_plot.png')

                if np.all(R_list < gelman_rubin_limit):
                    # Convergence reached...
                    completed_iterations = iteration + 1
                    print("CONVERGENCE REACHED!! at iteration {}".format(iteration))
                    break
    else:
        completed_iterations = iterations

    if plot_convergence:
        for param in conv_vals:
            ax.plot(conv_iters, conv_vals[param], label=param)
        ax.legend()
        ax.axhline(gelman_rubin_limit, c="k", ls="--")
        ax.set_xlim(left=burn_in)
        ax.set_ylim(bottom=1.0)
        ax.set_yscale("log")
        ax.set_ylim(0, 10)
        conv_f.savefig(str(saving_folder) + "/convergence_plot.png")

    print()
    print()
    if numb_chains is not None:
        print(
            "{} iterations have been completed with {} contemporaneous chains".format(
                iterations, numb_chains
            )
        )
    if numb_chains is None:
        print(
            "{} iterations have been completed with {} contemporaneous chains".format(
                iterations, 100
            )
        )
    print()

    # Compute the acceptance rate
    passed = 0
    rejected = 0
    accepted_flat = accepted.flatten()
    for m in range(len(accepted_flat)):
        # print(m)
        if accepted_flat[m]:
            passed += 1
        if not accepted_flat[m]:
            rejected += 1

    print(passed, rejected)
    ratio = passed / (iterations * numb_chains + numb_chains)
    print("Acceptance Rate = ", ratio)

    print(" ---- %s minutes ----" % ((time.time() - start) / 60))

    # Mixing plots
    # plot.mixing_plot(iterations+4, hparam_chain, hparam_names, model_param_chain, model_param_names, LogL_chain)
    if mass:
        return (
            logL_list,
            hparameter_list,
            model_parameter_list,
            mass0,
            mass1,
            completed_iterations,
        )

    else:
        return logL_list, hparameter_list, model_parameter_list, completed_iterations


def ben_run_MCMC(
    iterations,
    t,
    rv,
    rv_err,
    hparam0,
    kernel_name,
    model_param0,
    model_name,
    prior_list,
    numb_chains=100,
    n_splits=2,
    a=2.0,
    Rstar=None,
    Mstar=None,
    flags=None,
    plot_convergence=False,
    saving_folder=None,
    mass=False,
    numb_cores=1,
    timeit=False,
):
    start = time.time()

    if numb_cores > 1 and numb_chains < 500:
        print("Too few chains to parallelise, setting numb_cores to 1")
        numb_cores = 1

    _ = MCMC(
        t,
        rv,
        rv_err,
        hparam0,
        kernel_name,
        model_param0,
        model_name,
        prior_list,
        numb_chains,
        flags=flags,
        numb_cores=numb_cores,
    )

    """
    New version of the run_MCMC function that uses the parallelisation of the bundle step
    """
    if timeit:
        itertime = []
        itertime_start = time.time()
    for iteration in range(iterations):
        _.ben_update()
        if (iteration % 2 == 0) or iteration == iterations - 1:
            aux.printProgressBar(iteration, iterations - 1, length=50)

        if timeit:
            itertime_end = time.time()
            itertime.append(itertime_end - itertime_start)
            itertime_start = itertime_end

    print()
    print(f"Completed {iterations} iterations with {numb_chains} chains")

    passed = 0
    rejected = 0
    try:
        accepted_flat = _.accepted.copy().flatten()
    except:
        import pdb

        pdb.set_trace()
    for m in range(len(accepted_flat)):
        if accepted_flat[m]:
            passed += 1
        if not accepted_flat[m]:
            rejected += 1

    print(f"Accepted {passed} and rejected {rejected} proposals")
    print(f"Acceptance Rate = {passed / (iterations * numb_chains + numb_chains)}")

    print(" ---- %s minutes ----" % ((time.time() - start) / 60))

    # TODO: add convergence to this, want to let completed_iterations be the number of iterations
    completed_iterations = iterations
    if timeit:
        return (
            _.logL_list,
            _.hparameter_list,
            _.model_parameter_list,
            completed_iterations,
            itertime,
        )

    return _.logL_list, _.hparameter_list, _.model_parameter_list, completed_iterations
