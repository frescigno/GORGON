#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of auxiliary functions for the GP_solar and MCMC codes.

Contains:
    printProgressBar
    Creates the progress bar for the MCMC iterations

Author: Federica Rescigno
Version: 25-01-2022
"""

import numpy as np


def printProgressBar(
    iteration,
    total,
    prefix="Progress: ",
    suffix="Complete",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """Call in a loop to create terminal progress bar
    Parameters
    ----------
    iteration:  integer
        Current iteration
    total:      integer
        Total expected iterations
    prefix:     string, optional
        String before progress bar
    suffix:     string, optional
        String after percentage
    decimals:   integer, optional
        Number of decimals in the percetage
    length:     integer, optional
        Character lenght of the progress bar
    fill:       string, optional
        Bar fill character
    printEnd:   string, optional
        End character (e.g. "\r", "\r\n")
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def transit_to_periastron(t_tr, P, ecc, omega):
    """
    Parameters
    ----------
    t_tr : float
        Value of the time of transit (start of transit)
    P : float
        Period of the planet
    ecc : float
        Eccentricity of orbit
    omega : float
        Argument of periastron

    Returns
    -------
    t_0 : float
        Time of periastron

    """
    v_tr = np.pi / 2 - omega
    E_tr = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(v_tr / 2))
    t_0 = t_tr - (P / (2 * np.pi) * (E_tr - ecc * np.sin(E_tr)))

    return t_0


def periastron_to_transit(t_0, P, ecc, omega):
    """
    Parameters
    ----------
    t_0 : float
        Time of periastron
    P : float
        Period of the planet
    ecc : float
        Eccentricity of orbit
    omega : float
        Argument of periastron

    Returns
    -------
    t_tr : float
        Value of the time of transit (start of transit)

    """
    v_tr = np.pi / 2 - omega
    E_tr = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(v_tr / 2))
    t_tr = t_0 + (P / (2 * np.pi) * (E_tr - ecc * np.sin(E_tr)))

    return t_tr


def star_cross(Sk, Ck, Rstar, P, Mstar):
    """
    Parameters
    ----------
    Sk : float
        Sk value from MCMC
    Ck : float
        Ck value from MCMC
    Rstar : float
        Radius of the host star, in Solar Radii
    P : float
        Period of planet, in days
    Mstar : float
        Mass of the star, in Solar Masses

    Returns
    -------
    bool
        If True, the semi-major orbit axes does never fall into the star
        If False, the orbit falls into the star and the step should be dismissed

    """
    ecc_pl = Sk**2 + Ck**2
    Rsun = 6.95508e8  # in meters
    AU = 149597871e3  # in meters
    ratio = (Rstar * Rsun / AU) / (((P / 365.23) ** 2) * Mstar) ** (1 / 3)

    if ecc_pl < 1 - ratio:
        return True
    if ecc_pl >= 1 - ratio:
        return False


def to_SkCk(ecc, omega, ecc_err, omega_err):
    """

    Parameters
    ----------
    ecc : float
        Eccentricity
    omega : float, radians
        Angle of periastron
    ecc_err : float
        Error on the eccentricity
    omega_err : float
        Error on angle of periastron

    Returns
    -------
    Sk : float
        Sk value
    Ck : float
        Ck value
    Sk_err : float
        Error on Sk
    Ck_err : float
        Error on Ck

    """

    Sk = np.sqrt(ecc) * np.sin(omega)
    Ck = np.sqrt(ecc) * np.cos(omega)

    if ecc == 0.0:
        Sk_err = ecc_err
        Ck_err = ecc_err
    else:
        Sk_err = np.sqrt(
            (ecc_err**2 * (np.sin(omega)) ** 2 / (4 * ecc))
            + (omega_err**2 * ecc * (np.cos(omega)) ** 2)
        )
        Ck_err = np.sqrt(
            (ecc_err**2 * (np.cos(omega)) ** 2 / (4 * ecc))
            + (omega_err**2 * ecc * (np.sin(omega)) ** 2)
        )

    return Sk, Ck, Sk_err, Ck_err


def to_ecc(Sk, Ck, errSk=None, errCk=None):
    ecc = Sk**2 + Ck**2
    if ecc == 0.0:
        omega = np.pi / 2
    else:
        # omega = np.arctan(Sk/Ck)
        omega = np.arctan2(Sk, Ck)

    if errSk is not None and errCk is not None:
        errecc = np.sqrt(errSk**2 * 4 * Sk**2 + errCk**2 * 4 * Ck**2)
        erromega = np.sqrt(
            errSk**2
            * (
                Ck / (Ck**2 + Sk**2) ** 2
                + errCk**2 * (-Sk / (Ck**2 + Sk**2) ** 2)
            )
        )

        return ecc, omega, errecc, erromega

    return ecc, omega


def initial_pos_creator(
    param, param_err, numb_chains, allow_neg=True, param_names=None
):
    """

    Parameters
    ----------
    param : list, floats
        List of the initial guess parameters
    param_err : list, floats
        List of the errors on the initial guess parameters
    numb_chains : int
        Number of chains
    allow_neg : boolean
        Allow negative starting values. Default is False
    Returns
    -------
    chains_param : 2D list, floats
        2D array of

    """

    chains_param = []
    # For the first chain, use the guesses themselves
    chains_param.append(param)

    # For the rest create them by multipling a random number between -1
    for l in range(numb_chains - 1):
        pos = param + param_err * np.random.uniform(-1.0, 1.0, (1, len(param)))
        # Fix double parenthesis
        # print(pos)
        # pos.tolist()
        if not allow_neg:
            while np.min(pos) < 0:
                if param_names is None:
                    pos = param + param_err * np.random.uniform(
                        -1.0, 1.0, (1, len(param))
                    )
                elif param_names is not None:
                    # print("in")
                    for i in range(len(param_names)):
                        if param_names[i].startswith("ecc") or param_names[
                            i
                        ].startswith("omega"):
                            # print("ecc or omega")
                            pass
                        else:
                            while pos[0][i] < 0:
                                pos[0][i] = param[i] + param_err[i] * np.random.uniform(
                                    -1.0, 1.0, (1, len(param[i]))
                                )
                                # print("pos", pos)
        chains_param.append(pos[0])

    # chains_param should have on the horizontal the parameter values for each chain
    # on the vertical the number of chains
    return np.array(chains_param)


def mass_calc(model_param, Mstar):
    """
    Parameters
    ----------
    model_param : 2d array
        Array of all the model parameter in the mcmc (with Sk and Ck instead of ecc and omega)
    Mstar : float
        Stellar mass in solar masses

    Returns
    -------
    Mpl_sini : float
        Minimum mass of the planet in Jupiter masses

    """

    P, K, Ck, Sk, t0 = (
        model_param[-1][0],
        model_param[-1][1],
        model_param[-1][2],
        model_param[-1][3],
        model_param[-1][4],
    )
    ecc = Ck**2 + Sk**2
    omega = np.arctan(Sk / Ck)
    Mpl_sini = (
        4.9191
        * 10 ** (-3)
        * K
        * np.sqrt(1 - ecc**2)
        * P ** (1 / 3)
        * Mstar ** (2 / 3)
    )

    return Mpl_sini


def transpose(arr):
    """
    Parameters
    ----------
    arr : list
        List you want to transpose

    Returns
    -------
    trans2 : list
        Transposed list

    """
    arr2 = np.array(arr)
    trans = arr2.T
    trans2 = trans.tolist()

    return trans2


def model_param_names(model_list, SkCk=False):
    """
    Function to get model names

    Parameters
    ----------
    model_name : string or list
        Name of the model, or list of names of the models
    SkCk : boolean, optional
        If True, return the names of the Sk and Ck parameters. Default is False

    Returns
    -------
    param_names : list of strings
        Name of parameters
    """

    # Check if it's a single model
    if isinstance(model_list, str):
        model_list = [model_list]
    if isinstance(model_list, list) and len(model_list) == 1:
        numb = 1
    elif isinstance(model_list, list) and len(model_list) > 1:
        numb = len(model_list)
    else:
        raise ValueError("Model must be a string or a list of strings")

    # If it's a single model
    if numb == 1:
        if model_list[0].startswith("Kepler") or model_list[0].startswith("kepler"):
            param_names = ["P", "K", "Ck", "Sk", "t0"]
            
        if model_list[0].startswith("Poly") or model_list[0].startswith("poly"):
            param_names = ["a_0", "a_1", "a_2", "a_3"]
        
        if model_list[0].startswith("Sk") or model_list[0].startswith("sk"):
            param_names = ["amp", "per", "phase", "offset"]

        if (
            model_list[0].startswith("No_Model")
            or model_list[0].startswith("No")
            or model_list[0].startswith("no")
        ):
            param_names = ["no"]

        if model_list[0].startswith("Offset") or model_list[0].startswith("offset"):
            param_names = ["offset"]
    else:
        # Check how many times each model is called
        n_kep = 0
        n_no = 0
        n_off = 0
        n_skew=0
        param_names = []
        for mod_name in model_list:
            param_names_mods = None
            if mod_name.startswith("Kepler") or mod_name.startswith("kepler"):
                if SkCk:
                    param_names_mods = [
                        "P_" + str(n_kep),
                        "K_" + str(n_kep),
                        "Ck_" + str(n_kep),
                        "Sk_" + str(n_kep),
                        "t0_" + str(n_kep),
                    ]
                    param_names.extend(param_names_mods)
                if not SkCk:
                    param_names_mods = [
                        "P_" + str(n_kep),
                        "K_" + str(n_kep),
                        "ecc_" + str(n_kep),
                        "omega_" + str(n_kep),
                        "t0_" + str(n_kep),
                    ]
                    param_names.extend(param_names_mods)
                n_kep += 1
            if (
                mod_name.startswith("No_Model")
                or mod_name.startswith("No")
                or mod_name.startswith("no")
            ):
                param_names_mods = ["no_" + str(n_no)]
                param_names.extend(param_names_mods)
                n_no += 1
            if mod_name.startswith("Offset") or mod_name.startswith("offset"):
                param_names_mods = ["offset_" + str(n_off)]
                param_names.extend(param_names_mods)
                n_off += 1
            if mod_name.startswith("Sk") or mod_name.startswith("sk"):
                param_names_mods = [
                        "amp_" + str(n_kep),
                        "per_" + str(n_kep),
                        "phase_" + str(n_kep),
                        "offset_" + str(n_kep)]
                param_names.extend(param_names_mods)
                n_skew += 1

    return param_names


def hparam_names(kernel_name):
    """
    Function to get kernel hyperparameters names

    Parameters
    ----------
    kernel_name : string
        Name of the kernel

    Returns
    -------
    hparam_names : list of strings
        Name of hyperparameters
    """
    if kernel_name.startswith("Cos") or kernel_name.startswith("cos"):
        hparam_names = ["gp_amp", "gp_per"]
    if kernel_name.startswith("Exp") or kernel_name.startswith("exp"):
        hparam_names = ["gp_amp", "gp_length", "gp_per"]
    if kernel_name.startswith("Quas") or kernel_name.startswith("quas"):
        hparam_names = ["gp_per", "gp_perlegth", "gp_explength", "gp_amp"]
    if kernel_name.startswith("Jit") or kernel_name.startswith("jit"):
        hparam_names = ["gp_per", "gp_perlegth", "gp_explength", "gp_amp", "jitter"]
    if kernel_name.startswith("Mat") or kernel_name.startswith("mat"):
        hparam_names = ["gp_amp", "gp_timescale", "gp_jit"]

    return hparam_names


def phasefold(time, period, t0, zerocentre=True, returnepoch=False):
    """
    Function to phase-fold data

    Parameters
    ----------
    time : array, float
        Time array
    period : float
        Period of the orbit
    t0 : float
        Time of periastron passage
    zerocentre : boolean
        If True, the time array is shifted to the zero-centre. Default is True
    returnepoch : boolean
        Return the epoch of the phase-folded data. Default is False

    Returns
    -------
    true_phase : array, floats
        Phase array
    epoch : array, floats, optional
        Epoch of the orbit at all points. Starting from 0. Only returned if returnepoch is True.

    """
    phase = (time - t0) / period
    # Want phase to be between 0 and 1
    epoch = np.floor(phase)
    true_phase = phase - epoch

    if zerocentre:
        end = np.where(true_phase >= 0.5)[0]
        true_phase[end] -= 1.0

    if returnepoch:
        return true_phase, epoch
    else:
        return true_phase
