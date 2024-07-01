#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Process Regression Base Code
Contains:
    Kernels and Parameter Creators
    Priors and Parameter Creators
    Models and Parameter Creators
    GP Likelyhood
    Sum of Models

Author: Federica Rescigno
Version: 20-04-2022
"""

import scipy
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt


"""KERNELS"""

# List of implemented kernels with hyperparameters
KERNELS = {
    "Matern3": ["gp_amp", "gp_timescale", "gp_jit"],
    "Cosine": ["gp_amp", "gp_per"],
    "ExpSinSquared": ["gp_amp", "gp_length", "gp_per"],
    "QuasiPer": ["gp_per", "gp_perlength", "gp_explength", "gp_amp"],
    "JitterQuasiPer": ["gp_per", "gp_perlength", "gp_explength", "gp_amp", "gp_jit"],
}


def PrintKernelList():
    print("Implemented kernels:")
    print(KERNELS)


class Matern3:
    """Class that computes the Matern 3/2 matrix.

    Kernel formula:

        K =

    in which:
        H_1 = variance/amp
        H_2 = timescale
    """

    def __init__(self, hparams):
        """
           Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements
        """

        # Initialize
        self.covmatrix = None
        self.hparams = hparams

        # Check if we have the right amount of parameters
        assert len(self.hparams) == 3, (
            "Matern 3/2 kernel requires 3 hyperparameters:"
            + "'gp_amp', 'gp_timescale','gp_jit'"
        )

        # Check if all parameters are numbers
        try:
            self.hparams["gp_amp"].value
            self.hparams["gp_timescale"].value
            self.hparams["gp_jit"].value
        except KeyError:
            raise KeyError(
                "Matern 3/2 kernel requires 2 hyperparameters:" + "'amp', 'timescale'"
            )

    @property
    def name(self):
        return "Matern 3/2"

    @property
    def __repr__(self):
        """
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        """

        per = self.hparams["gp_timescale"].value
        amp = self.hparams["gp_amp"].value
        jit = self.hparams["gp_jit"].value

        message = "Matern 3/2 Kernel with amp: {}, timescale: {}".format(amp, per)
        print(message)

    def compute_distances(self, t1, t2):
        """
        Parameters
        ----------
        t1 : array or list, floats
            Array of the first time series
        t2 : array or list, floats
            Array of the second time series

        Returns
        -------
        self.dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t')
        self.dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2

        """
        T1 = np.array([t1]).T
        T2 = np.array([t2]).T

        self.dist_e = scipy.spatial.distance.cdist(T1, T2, "euclidean")
        self.dist_se = scipy.spatial.distance.cdist(T1, T2, "sqeuclidean")

        return self.dist_e, self.dist_se

    def compute_covmatrix(self, errors=None):
        """
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        """

        timescale = self.hparams["gp_timescale"].value
        amp = self.hparams["gp_amp"].value
        jit = self.hparams["gp_jit"].value

        r = self.dist_e / timescale
        r2 = self.dist_se / timescale**2

        # K = np.array(amp**2 * (1 + (np.sqrt(5)*self.dist_e/timescale) + (5*self.dist_se**2/3*timescale**2) * np.exp(-np.sqrt(5)*self.dist_e/timescale)))
        # K = np.array(amp**2 * (1 + (np.sqrt(5)*r) + (5/3 *r2) * np.exp(-np.sqrt(5)*r)))
        K = np.array(amp**2 * (1 + (np.sqrt(3) * r)) * np.exp(-np.sqrt(3) * r))

        sigma = np.identity(K.shape[0]) * jit**2

        # This is the covariance matrix
        try:
            self.covmatrix = K + sigma
        except ValueError:
            self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # if errors are not present or the array is non-square
            pass

        return self.covmatrix


class Cosine:
    """Class that computes the Cosine kernel matrix.

    Kernel formula:

        K = H_1^2 cos[(2pi . |t-t'|) / H_2]}

    in which:
        H_1 = variance/amp
        H_2 = per

    Arguments:
        hparams: dictionary with all the hyperparameters
        Should have 2 elements with possibly errors
    """

    def __init__(self, hparams):
        """
           Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with possibly errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 2 required parameters
        """

        # Initialize
        self.covmatrix = None
        self.hparams = hparams

        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, (
            "Periodic Cosine kernel requires 2 hyperparameters:" + "'gp_amp', 'gp_per'"
        )

        # Check if all parameters are numbers
        try:
            self.hparams["gp_amp"].value
            self.hparams["gp_per"].value
        except KeyError:
            raise KeyError("Cosine kernel requires 2 hyperparameters:" + "'amp', 'per'")

    def name(self):
        print("Cosine")
        return "Cosine"

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        """

        per = self.hparams["gp_per"].value
        amp = self.hparams["gp_amp"].value

        message = "Cosine Kernel with amp: {}, per: {}".format(amp, per)
        print(message)
        return message

    def compute_distances(self, x1, x2):
        """
        Parameters
        ----------
        x1 : array or list, floats

        x2 : array or list, floats
            DESCRIPTION.

        Returns
        -------
        self.dist_e : ???
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t') within sine
        self.dist_se : ???
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2
        """

        X1 = np.array([x1]).T
        X2 = np.array([x2]).T

        self.dist_e = scipy.spatial.distance.cdist(X1, X2, "euclidean")

        return self.dist_e

    def compute_covmatrix(self, errors):
        """
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        """

        per = self.hparams["gp_per"].value
        amp = self.hparams["gp_amp"].value

        K = np.array(amp**2 * np.cos(2 * np.pi * self.dist_e / per))

        self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # if errors are not present or the array is non-square
            pass

        return self.covmatrix


class ExpSinSquared:
    """Class that computes the Periodic kernel matrix.

    Kernel formula:

        K = H_1^2 . exp{-2/H_3^2 . sin^2[(pi . |t-t'|) / H_2]}

    in which:
        H_1 = variance/amp
        H_3 = recurrence timescale/length
        H_2 = period

    Arguments:
        hparams: dictionary with all the hyperparameters
        Should have 3 elements with errors
    """

    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 3 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 3 required parameters
        """

        # Initialize
        self.covmatrix = None
        self.hparams = hparams

        # Check if we have the right amount of parameters
        assert len(self.hparams) == 3, (
            "Periodic ExpSinSquared kernel requires 3 hyperparameters:"
            + "'gp_amp', 'gp_length', 'gp_per'"
        )

        # Check if all parameters are numbers
        try:
            self.hparams["gp_amp"].value
            self.hparams["gp_length"].value
            self.hparams["gp_per"].value
        except KeyError:
            raise KeyError(
                "Periodic ExpSinSquared kernel requires 3 hyperparameters:"
                + "'amp', 'length', 'per'"
            )

    def name(self):
        print("ExpSinSquared")
        return "ExpSinSquared"

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        """

        per = self.hparams["gp_per"].value
        amp = self.hparams["gp_amp"].value
        length = self.hparams["gp_length"].value

        message = (
            "Periodic ExpSinSquared Kernel with amp: {}, length: {}, per: {}".format(
                amp, length, per
            )
        )
        print(message)
        return message

    def compute_distances(self, x1, x2):
        """
        Parameters
        ----------
        x1 : array or list, floats

        x2 : array or list, floats
            DESCRIPTION.

        Returns
        -------
        self.dist_e : ???
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t') within sine
        self.dist_se : ???
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2
        """

        X1 = np.array([x1]).T
        X2 = np.array([x2]).T

        self.dist_e = scipy.spatial.distance.cdist(X1, X2, "euclidean")

        return self.dist_e

    def compute_covmatrix(self, errors):
        """
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        """

        per = self.hparams["gp_per"].value
        length = self.hparams["gp_length"].value
        amp = self.hparams["gp_amp"].value

        K = np.array(
            amp**2
            * np.exp((-2 / length**2) * (np.sin(np.pi * self.dist_e / per)) ** 2)
        )

        self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # if errors are not present or the array is non-square
            pass

        return self.covmatrix


class QuasiPer:
    """Class that computes the quasi-periodic kernel matrix.

    Kernel formula from Haywood Thesis 2016, Equation 2.14:

        K = H_1^2 . exp{ [-(t-t')^2 / H_2^2] - [ sin^2(pi(t-t')/H_3) / H_4^2] }

    in which:
        H_1 = amp
        H_2 = explength
        H_3 = per
        H_4 = perlength

    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors

    """

    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 4 required parameters
        """

        # Initialize final result
        self.covmatrix = None
        self.hparams = hparams

        # Check if we have the right amount of parameters
        assert len(self.hparams) == 4, (
            "QuasiPeriodic Kernel requires 4 hyperparameters:"
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'"
        )

        # Check if all hyperparameters are numbers
        try:
            self.hparams["gp_per"].value
            self.hparams["gp_perlength"].value
            self.hparams["gp_explength"].value
            self.hparams["gp_amp"].value
        except KeyError:
            raise KeyError(
                "QuasiPeriodic Kernel requires 4 hyperparameters:"
                + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'"
            )

    def name(self):
        print("QuasiPeriodic")
        return "QuasiPer"

    # String to call to see what the class is doing
    def __repr__(self):
        """
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        """
        per = self.hparams["gp_per"].value
        perlength = self.hparams["gp_perlength"].value
        explength = self.hparams["gp_explength"].value
        amp = self.hparams["gp_amp"].value

        message = "QuasiPeriodic Kernel with amp: {}, per length: {}, per: {}, exp length: {}".format(
            amp, perlength, per, explength
        )
        print(message)
        return message

    def compute_distances(self, t1, t2):
        """
        Parameters
        ----------
        y1 : array or list, floats
            DESCRIPTION.
        y2 : array or list, floats
            DESCRIPTION.

        Returns
        -------
        self.dist_e : ???
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t') within sine
        self.dist_se : ???
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2

        """
        X1 = np.array([t1]).T
        X2 = np.array([t2]).T

        self.dist_e = scipy.spatial.distance.cdist(X1, X2, "euclidean")
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, "sqeuclidean")

        return self.dist_e, self.dist_se

    def compute_covmatrix(self, errors):
        """
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the quasiperiodic kernel

        """

        per = self.hparams["gp_per"].value
        perlength = self.hparams["gp_perlength"].value
        explength = self.hparams["gp_explength"].value
        amp = self.hparams["gp_amp"].value

        # print("Paramters for this round:", self.hparams)

        K = np.array(
            amp**2
            * np.exp(-self.dist_se / (explength**2))
            * np.exp(-((np.sin(np.pi * self.dist_e / per)) ** 2) / (perlength**2))
        )

        # This is the covariance matrix
        self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # if errors are present or the array is non-square
            pass

        return self.covmatrix


class JitterQuasiPer:
    """Class that computes the quasi-periodic kernel matrix + jitter.

    Kernel formula from Haywood Thesis 2016, Equation 2.14:

        K = H_1^2 . exp{ [-(t-t')^2 / H_2^2] - [ sin^2(pi(t-t')/H_3) / H_4^2] } + delta_nm sigma

    in which:
        H_1 = amp
        H_2 = explength
        H_3 = per
        H_4 = perlength

    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors

    """

    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 4 required parameters
        """

        # Initialize final result
        self.covmatrix = None
        self.hparams = hparams

        # Check if we have the right amount of parameters
        assert len(self.hparams) == 5, (
            "QuasiPeriodic Kernel requires 4 hyperparameters:"
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp', 'gp_jit'"
        )

        # Check if all hyperparameters are numbers
        try:
            self.hparams["gp_per"].value
            self.hparams["gp_perlength"].value
            self.hparams["gp_explength"].value
            self.hparams["gp_amp"].value
            self.hparams["gp_jit"].value
        except KeyError:
            raise KeyError(
                "QuasiPeriodic Kernel requires 4 hyperparameters:"
                + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp', 'gp_jit'"
            )

    def name(self):
        print("QuasiPeriodic")
        return "QuasiPer"

    # String to call to see what the class is doing
    def __repr__(self):
        """
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        """
        per = self.hparams["gp_per"].value
        perlength = self.hparams["gp_perlength"].value
        explength = self.hparams["gp_explength"].value
        amp = self.hparams["gp_amp"].value
        jit = self.hparams["gp_jit"].value

        message = "QuasiPeriodic Kernel with amp: {}, per length: {}, per: {}, exp length: {}, jit: {}".format(
            amp, perlength, per, explength, jit
        )
        print(message)
        return message

    def compute_distances(self, t1, t2):
        """
        Parameters
        ----------
        y1 : array or list, floats
            DESCRIPTION.
        y2 : array or list, floats
            DESCRIPTION.

        Returns
        -------
        self.dist_e : ???
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t') within sine
        self.dist_se : ???
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2

        """
        X1 = np.array([t1]).T
        X2 = np.array([t2]).T

        self.dist_e = scipy.spatial.distance.cdist(X1, X2, "euclidean")
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, "sqeuclidean")

        return self.dist_e, self.dist_se

    def compute_covmatrix(self, errors):
        """
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the quasiperiodic kernel

        """

        per = self.hparams["gp_per"].value
        perlength = self.hparams["gp_perlength"].value
        explength = self.hparams["gp_explength"].value
        amp = self.hparams["gp_amp"].value
        jit = self.hparams["gp_jit"].value

        # print("Paramters for this round:", self.hparams)

        K = np.array(
            amp**2
            * np.exp(-self.dist_se / (explength**2))
            * np.exp(-((np.sin(np.pi * self.dist_e / per)) ** 2) / (perlength**2))
        )

        sigma = np.identity(K.shape[0]) * jit**2
        # This is the covariance matrix
        try:
            self.covmatrix = K + sigma
        except ValueError:
            self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # if errors are present or the array is non-square
            pass

        return self.covmatrix


######################################################
######### PARAMETER OBJECTS FOR KERNELS #########
######################################################


class Par_Creator:
    """Object to create the set of parameters necessary for the chosen kernel, now only one kernel.

    Returns:
    hparams: dictionary
        Dictionary of all necessary parameters for given kernel
    """

    def __init__(self, kernel):
        """
        Parameters
        ----------
        kernel : string
            Name of the implemented kernel.
        """
        self.kernel = kernel

    def create(kernel):
        if (
            kernel.startswith("QuasiPer")
            or kernel.startswith("quasiper")
            or kernel.startswith("Quasiper")
        ):
            hparams = dict(
                gp_per="gp_per",
                gp_perlength="gp_perlength",
                gp_explength="gp_explength",
                gp_amp="gp_amp",
            )

        if kernel.startswith("Jit") or kernel.startswith("jit"):
            hparams = dict(
                gp_per="gp_per",
                gp_perlength="gp_perlength",
                gp_explength="gp_explength",
                gp_amp="gp_amp",
                gp_jit="gp_jit",
            )

        if (
            kernel.startswith("Periodic")
            or kernel.startswith("periodic")
            or kernel.startswith("ExpSin")
            or kernel.startswith("expsin")
            or kernel.startswith("Expsin")
        ):
            hparams = dict(gp_amp="gp_amp", gp_length="gp_length", gp_per="gp_per")

        if kernel.startswith("Cos") or kernel.startswith("cos"):
            hparams = dict(gp_amp="gp_amp", gp_per="gp_per")

        if kernel.startswith("Matern3") or kernel.startswith("matern3"):
            hparams = dict(
                gp_amp="gp_amp", gp_timescale="gp_timescale", gp_jit="gp_jit"
            )

        return hparams


class Parameter:
    """Object to assign initial values to a parameter and define whether it is
    allowed to vary in the fitting
    """

    def __init__(self, value=None, error=None, vary=True):
        """
        Parameters
        ----------
        value : float, optional
            Assumed initial value of the chosen variable. The default is None.
        error : float, optional
            Error on the value. The default is None
        vary : True or False, optional
            Is the variable allowed to vary? The default is True.
        """

        self.value = value
        self.error = error
        self.vary = vary

        # If no error is given, compute the error as 20% of the value
        if self.error is None:
            self.error = 0.2 * self.value

    # String to see what the parameters look like
    def __repr__(self):
        """
        Returns
        -------
        message : string
            List of specific parameter value and characteristics
        """
        message = ("Parameter object: value = {}, error={} (vary = {}) \n").format(
            self.value, self.error, self.vary
        )
        return message

    def vary(self):
        return self.vary

    def value(self):
        return self.value

    def error(self):
        return self.error


################################################################################################
######## PRIORS, series of priors that the code can implement divided in classes ########
################################################################################################

PRIORS = {
    "Gaussian": ["hparam", "mu", "sigma"],
    "Jeffrey": ["hparam", "minval", "maxval"],
    "Modified_Jeffrey": ["hparam", "minval", "maxval", "kneeval"],
    "Uniform": ["hparam", "minval", "maxval"],
}


def PrintPriorList():
    print("Implemented priors:")
    print(PRIORS)


class Gaussian:
    """Gaussian prior computed as:

        -0.5 * ((x - mu) / sigma)**2 -0.5 * np.log(2*pi * sigma**2)

    Args:
        hparam (string): parameter label
        mu (float): centre of Gaussian Prior
        sigma (float): width of the Gaussian Prior
    """

    def __init__(self, hparam, mu, sigma):
        """
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        mu : float
            Center of the Gaussian prior
        sigma : float
            FWHM of the Gaussian prior
        """
        self.hparam = hparam
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Description of the prior.
        """
        message = (
            "Gaussian prior on the parameter {}, with mu = {} and sigma = {}"
        ).format(self.hparam, self.mu, self.sigma)
        print(message)
        return message

    def logprob(self, x):
        """
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        """
        logprob = -0.5 * ((x - self.mu) / self.sigma) ** 2 - np.log(
            np.sqrt(2 * np.pi) * self.sigma
        )
        return logprob


class Jeffrey:
    """Jeffrey prior computed as:

        p(x) proportional to  1/x
        with upper and lower bound to avoid singularity at x = 0

        and normalized as:
            1 / ln(maxval/minval)

    Args:
        hparam (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    """

    def __init__(self, hparam, minval, maxval):
        """
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        """
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval

        assert (
            self.minval < self.maxval
        ), "Minimum value {} must be smaller than the maximum value {}".format(
            self.minval, self.maxval
        )

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Description of the prior
        """
        message = (
            "Jeffrey prior on the paramter {}, with minimum and maximum values of x = ({}, {})"
        ).format(self.hparam, self.minval, self.maxval)
        print(message)
        return message

    def logprob(self, x):
        """
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        """
        if x < self.minval or x > self.maxval:
            # logprob = -np.inf
            logprob = -10e5
            return logprob
        else:
            normalisation = 1.0 / (np.log(self.maxval / self.minval))
            prob = normalisation * 1.0 / x
            logprob = np.log(prob)
            return logprob


class Modified_Jeffrey:
    """Modified Jeffrey prior computed as:

        p(x) proportional to  1/(x-x0)
        with upper bound


    Args:
        hparam (string): parameter label
        kneeval (float): x0, knee of the Jeffrey prior
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    """

    def __init__(self, hparam, minval, maxval, kneeval):
        """
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        kneeval: float
            Kneww value of prior (x0)
        """
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval
        self.kneeval = kneeval

        assert (
            self.minval < self.maxval
        ), "Minimum value {} must be smaller than the maximum value {}".format(
            self.minval, self.maxval
        )

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Description of the prior
        """
        message = (
            "Modified Jeffrey prior on the paramter {}, with minimum and maximum values of x = ({}, {}) and knee = {}"
        ).format(self.hparam, self.minval, self.maxval, self.kneeval)
        print(message)
        return message

    def logprob(self, x):
        """
        Parameters
        ----------
        x : float
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        """
        if x < self.minval or x > self.maxval:
            # logprob = -np.inf
            logprob = -10e5
            return logprob
        else:
            normalisation = 1.0 / (
                np.log((self.maxval + self.kneeval) / (self.kneeval))
            )
            prob = normalisation * 1.0 / (x - self.kneeval)
            logprob = np.log(prob)
            return logprob


class Uniform:
    """Uniform prior

    Args:
        hparam (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    """

    def __init__(self, hparam, minval, maxval):
        """
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        """
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval

        assert (
            self.maxval > self.minval
        ), "Minimum value {} must be smaller than the maximum value {}".format(
            self.minval, self.maxval
        )

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Description of the prior
        """
        message = (
            "Uniform prior on the paramter {}, with minimum and maximum values of x = ({}, {})"
        ).format(self.hparam, self.minval, self.maxval)
        print(message)
        return message

    def logprob(self, x):
        """
        Parameters
        ----------
        x : array
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        """
        if x < self.minval or x > self.maxval:
            # logprob = -np.inf
            logprob = -10e5
            return logprob
        else:
            logprob = 0
            return logprob


class Prior_Par_Creator:
    """Object to create the set of parameters necessary for the chosen prior, now only one prior.
    To assign value: prior_params["name"] = value

    Returns:
    hparams: dictionary
        Dictionary of all necessary prior parameters
    """

    def __init__(self, prior):
        """
        Parameters
        ----------
        prior : string
            Name of the implemented prior.
        """
        self.prior = prior

    def create(prior):
        if prior.startswith("Gauss") or prior.startswith("gauss"):
            prior_params = dict(mu="mu", sigma="sigma")

        if prior.startswith("Jeffrey") or prior.startswith("jeffrey"):
            prior_params = dict(minval="minval", maxval="maxval")

        if prior.startswith("Mod") or prior.startswith("mod"):
            prior_params = dict(minval="minval", maxval="maxval", kneeval="kneeval")

        if prior.startswith("Uni") or prior.startswith("uni"):
            prior_params = dict(minval="minval", maxval="maxval")

        return prior_params


########################
######## MODEL ########
########################

# This is what models the possible planets or offset


MODELS = {
    "No_Model": ["rvs"],
    "Offset": ["rvs", "offset"],
    "Polinomyal":["a_0","a_1","a_2","a_3"],
    "Keplerian": ["time", "P", "K", "ecc", "omega", "t0"],
    "Uncorr_Noise": ["to come"],
}
# will add Uncorrelated noise, FF' and others


def PrintModelList():
    print("Implemented models")
    print(MODELS)


class No_Model:
    """The subtracted model from the RV is null"""

    def __init__(self, y, no):
        """
        Parameters
        ----------
        y : array
            Observed RV or ys
        """

        self.y = y

    def model(self):
        """
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        """
        model_y = np.zeros(len(self.y))
        return model_y


class Offset:
    """The subtracted model from RV is a constant offset chosen explicitlty"""

    def __init__(self, flags, model_params):
        """
        Parameters
        ----------
        y : array
            Observed RV or y.
        offset : float
            Constant offset
        """

        self.flags = flags
        self.model_params = model_params

        # Check if we have the right amount of parameters
        assert len(self.model_params) == 1, (
            "Offset Model requires 2 parameter:" + "'offset'"
        )

        # Check if all hyperparameters are numbers
        try:
            self.offset = self.model_params["offset"].value
        except KeyError:
            for i in range(10):
                try:
                    self.offset = self.model_params["offset_" + str(i)].value
                    break
                except KeyError:
                    raise KeyError("Offset Model requires 1 parameter:" + "'offset'")

    def model(self):
        """
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        """
        model_y = []
        for i in range(len(self.flags)):
            if self.flags[i] > 0.5:
                model_y.append(self.offset)
            else:
                model_y.append(0.0)

        # model_y = [self.offset] * len(self.y)
        model_y = np.array(model_y)
        return model_y



class Polynomial:
    '''The subtracted model from the RV is null
    '''
    
    def __init__(self, time, model_params):
        '''
        Parameters
        ----------
        time : array
            Time or x axis array
        '''
        
        self.time = time
        self.model_params = model_params
        
        # Check if we have the right amount of parameters
        assert len(self.model_params) == 4, "Offset Model requires 4 parameters"
        
        # Check if all hyperparameters are numbers
        try:
            self.a_0 = self.model_params['a_0'].value
            self.a_1 = self.model_params['a_1'].value
            self.a_2 = self.model_params['a_2'].value
            self.a_3 = self.model_params['a_3'].value
        except KeyError:
            for i in range(10):
                try:
                    self.a_0 = self.model_params['a_0'+str(i)].value
                    self.a_1 = self.model_params['a_1'+str(i)].value
                    self.a_2 = self.model_params['a_2'+str(i)].value
                    self.a_3 = self.model_params['a_3'+str(i)].value
                    break
                except KeyError:
                    raise KeyError("Offset Model requires 4 parameters")
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y = self.a_0 + self.a_1*self.time + self.a_2*self.time**2 + self.a_3*self.time**3
        return model_y



class Keplerian:
    """The generalized Keplerian RV model (only use when dealing with RV observation
    of star with possible planet).
    If multiple planets are involved, the model parameters should be inputted as list (not fully implemented yet).

    """

    def __init__(self, time, model_params):
        """
        Parameters
        ----------
        time : array, floats
            Time array over which to compute the Keplerian.
        model_params: dictionary
            Dictionary of model parameters containing:
                P : float
                    Period of orbit in days.
                K : float
                    Semi-aplitude of the rv signal in meter per seconds.
                ecc : float
                    Eccentricity of orbit. Float between 0 and 0.99
                omega : float
                    Angle of periastron, in rad.
                t0 : float
                    Time of periastron passage of a planet
        """

        self.time = time
        self.model_params = model_params

        # Check if we have the right amount of parameters
        assert len(self.model_params) == 5, (
            "Keplerian Model requires 5 parameters:" + "'P', 'K', 'ecc', 'omega', 't0'"
        )

        # Check if all hyperparameters are number
        try:
            P = self.model_params["P"].value
            K = self.model_params["K"].value
            ecc = self.model_params["ecc"].value
            omega = self.model_params["omega"].value
            t0 = self.model_params["t0"].value
        except KeyError:
            for i in range(10):
                try:
                    self.model_params["P_" + str(i)].value
                    break
                except KeyError:
                    if i == 10:
                        raise KeyError(
                            "Keplerian Model requires 5 parameters:"
                            + "'P', 'K', 'ecc', 'omega', 't0'"
                        )
                    else:
                        continue

            P = self.model_params["P_" + str(i)].value
            K = self.model_params["K_" + str(i)].value
            ecc = self.model_params["ecc_" + str(i)].value
            omega = self.model_params["omega_" + str(i)].value
            t0 = self.model_params["t0_" + str(i)].value

        """P = self.model_params['P'].value
        K = self.model_params['K'].value
        ecc = self.model_params['ecc'].value
        omega = self.model_params['omega'].value
        t0 = self.model_params['t0'].value"""

        # P,K,ecc,omega,t0 = np.atleast_1d(P,K,ecc,omega,t0)
        # need for for loops

        self.P = P
        self.K = K
        self.ecc = ecc
        self.omega = omega
        self.t0 = t0

    def __repr__(self):
        """
        Returns
        -------
        message : string
            Description of the model
        """
        message = (
            "Keplerian RV model with the following set of parameters: \n",
            "P = {} \n".format(self.P),
            "K = {} \n".format(self.K),
            "ecc = {} \n".format(self.ecc),
            "omega = {} \n".format(self.omega),
            "t0 = {}".format(self.to),
        )
        print(message)
        return message

    def ecc_anomaly(self, M, ecc, max_itr=200):
        """
        ----------
        M : float
            Mean anomaly
        ecc : float
            Eccentricity, number between 0. and 0.99
        max_itr : integer, optional
            Number of maximum iteration in E computation. The default is 200.

        Returns
        -------
        E : float
            Eccentric anomaly
        """

        E0 = M
        E = M
        # print("E before = ", E)
        for i in range(max_itr):
            f = E0 - ecc * np.sin(E0) - M
            fp = 1.0 - ecc * np.cos(E0)
            E = E0 - f / fp

            # check for convergence
            if np.linalg.norm(E - E0, ord=1) <= 1.0e-10:
                return E
                break
            # if not convergence continue
            E0 = E

        # no convergence, return best estimate
        # print('Best estimate E = ',E[0:5])
        return E

    def true_anomaly(self, E, ecc):
        """
        Parameters
        ----------
        E : float
            Eccentric anomaly
        ecc : float
            Eccentricity

        Returns
        -------
        nu : float
            True anomaly

        """
        # print("ecc", ecc)
        # print()
        # print("E", E)
        nu = 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(E / 2.0))
        return nu

    def model(self):
        """
        Returns
        -------
        rad_vel : array, floats
            Radial volicity model derived by the number of planet include
        """
        rad_vel = np.zeros_like(self.time)

        # Need to add for multiple planets

        # Compute mean anomaly M
        # print("self.ecc = ",self.ecc[i],"/n")
        M = 2 * np.pi * (self.time - self.t0) / self.P

        # print("Model M = ", M,"/n")
        E = self.ecc_anomaly(M, self.ecc)
        # print("Model E = ", E,"/n")
        nu = self.true_anomaly(E, self.ecc)
        # print("Model nu = ", nu[0:5],"/n")

        rad_vel = rad_vel + self.K * (
            np.cos(self.omega + nu) + self.ecc * np.cos(self.omega)
        )

        # Add systemic velocity??
        model_keplerian = rad_vel
        # print("Model RV = ", rad_vel[0:5],"/n")
        # plt.plot(self.time, rad_vel)
        # plt.show()
        # print("Model done")
        return model_keplerian


"""class MultiplePlanetsModel:


    def __init__(self, numb_pl, """


##### PARAMETER OBJECTS FOR MODEL #####


class Model_Par_Creator:
    """Object to create the set of parameters necessary for the chosen model, now only one model.

    Returns:
    model_params: dictionary
        Dictionary of all necessary parameters
    """

    def __init__(self):
        """
        Parameters
        ----------
        model : string or list of strings
            Name of the implemented model, or list of the names of the models.
        """

    def model_number_check(self, models):
        """
        Get the number of implemented models
        """
        # Check if it's a single model
        if isinstance(models, str) or (isinstance(models, list) and len(models) == 1):
            numb = 1
        elif isinstance(models, list) and len(models) > 1:
            numb = len(models)
        else:
            raise ValueError("Model must be a string or a list of strings")
        return numb

    def create(self, model):
        # Check the number of models
        numb = self.model_number_check(model)

        # If it's a single model
        if numb == 1:
            if model[0].startswith("Kepler") or model[0].startswith("kepler"):
                model_params = dict(
                    P="period",
                    K="semi-amplitude",
                    ecc="eccentricity",
                    omega="angle of periastron",
                    t0="t of per pass",
                )

            if (
                model[0].startswith("No_Model")
                or model[0].startswith("No")
                or model[0].startswith("no")
            ):
                model_params = dict(no="no")

            if model[0].startswith("Offset") or model[0].startswith("offset"):
                model_params = dict(offset="offset")
            
            if model[0].startswith("Poly") or model[0].startswith("poly"):
                model_params = dict(a_0="a_0",a_1="a_1",a_2="a_2",a_3="a_3")
            
        else:
            # Check how many times each model is called
            n_kep = 0
            n_no = 0
            n_off = 0
            model_params = {}
            for mod_name in model:
                if mod_name.startswith("Kepler") or mod_name.startswith("kepler"):
                    model_params.update(
                        {
                            "P_" + str(n_kep): "period",
                            "K_" + str(n_kep): "semi-amplitude",
                            "ecc_" + str(n_kep): "eccentricity",
                            "omega_" + str(n_kep): "angle of periastron",
                            "t0_" + str(n_kep): "t of periastron passage",
                        }
                    )
                    n_kep += 1
                if (
                    mod_name.startswith("No_Model")
                    or mod_name.startswith("No")
                    or mod_name.startswith("no")
                ):
                    model_params.update({"no_" + str(n_no): "no"})
                    n_no += 1
                if mod_name.startswith("Offset") or mod_name.startswith("offset"):
                    model_params.update({"offset_" + str(n_off): "offset"})
                    n_off += 2
                if mod_name.startswith("Poly") or mod_name.startswith("poly"):
                    model_params.update({'a_0_'+str(n_off):'a_0',
                                         'a_1_'+str(n_off):'a_1',
                                         'a_2_'+str(n_off):'a_2',
                                         'a_3_'+str(n_off):'a_3'})
                    n_off += 1

        return model_params


#################################################################
##### LIKELYHOOD, or posterior when priors are involved #####
#################################################################


class GPLikelyhood:
    """Gaussian Process likelyhood"""

    def __init__(self, x, y, model_y, yerr, hparameters, model_param, kernel_name):
        """
        Parameters
        ----------
        x : array or list, floats
            Time series of the radial velocity
        y : array or list, floats
            Radial velocity values
        yerr : array or list, floats
            Radial velocity errors
        model_y: array, floats
            Array of y (rv) values from the model chosen
        hparamters: dictionary
            dictionary of all hyper parameters considered
        model_par: disctionary
            dictionary of all model parameters considered
        kernel_name: string
            name of the used kernel
        """
        self.x = np.array(x)  # Time series (must be array)
        self.y = np.array(y)  # Radial velocity array (must be array)
        self.yerr = np.array(yerr)  # Radial velocity error values (must be array)
        self.model_y = np.array(
            model_y
        )  # Coming from a model function, planet sinuisoidal or offset

        self.hparameters = hparameters  # Dictionary of all parameter each of Parameter class as: value, vary, mcmc scale
        self.hparam_names = list(hparameters.keys())
        self.hparam_values = []
        for key in hparameters.keys():
            self.hparam_values.append(hparameters[key].value)

        self.kernel_name = kernel_name

        self.model_param = model_param  # Dictionary of all parameters of the model
        self.model_param_names = list(model_param.keys())
        self.model_param_values = []
        for key in model_param.keys():
            self.model_param_values.append(model_param[key].value)

    def __repr__(self):
        """
        Returns
        -------
        message : string
        parameters : string
            List of all parameters with values
        """
        message = (
            "Gaussian Process Likelyhood object, computed with a {} kernel \n".format(
                self.kernel_name
            )
        )

        parameters = "Kernel parameters: \n"
        for i in range(len(self.hparam_values)):
            parameters += ("{} with initial value {} \n").format(
                self.hparam_names[i], self.hparam_values[i]
            )

        model_parameters = "Model parameters: \n"
        for i in range(len(self.model_param_values)):
            model_parameters += ("{} with initial value {} \n").format(
                self.model_param_names[i], self.model_param_values[i]
            )

        print(message)
        print(parameters)
        print(model_parameters)
        return message, parameters, model_parameters

    def compute_kernel(self, x1, x2):
        """
        Parameters
        ----------
        x1 : array or list, floats

        x2 : array or list, floats


        Returns
        -------
        covmatrix : array, floats
            Covariance matrix of the chosen set of varaibles x1 and x2
        """

        assert (
            self.kernel_name in KERNELS.keys()
        ), "Kernel not yet implemented. Pick from available kernels: " + str(
            KERNELS.keys()
        )

        x1 = np.array(x1)
        x2 = np.array(x2)
        yerr = self.yerr

        kernel_name = self.kernel_name
        if (
            kernel_name.startswith("QuasiPer")
            or kernel_name.startswith("quasiper")
            or kernel_name.startswith("Quasiper")
        ):
            self.kernel = QuasiPer(self.hparameters)
        if kernel_name.startswith("jit") or kernel_name.startswith("Jit"):
            self.kernel = JitterQuasiPer(self.hparameters)
        if (
            kernel_name.startswith("Periodic")
            or kernel_name.startswith("periodic")
            or kernel_name.startswith("ExpSin")
            or kernel_name.startswith("Expsin")
            or kernel_name.startswith("expsin")
        ):
            self.kernel = ExpSinSquared(self.hparameters)
        if kernel_name.startswith("Cos") or kernel_name.startswith("cos"):
            self.kernel = Cosine(self.hparameters)
        if (
            kernel_name.startswith("Matern3")
            or kernel_name.startswith("matern3")
            or kernel_name.startswith("Matern")
            or kernel_name.startswith("matern")
        ):
            self.kernel = Matern3(self.hparameters)

        self.kernel.compute_distances(x1, x2)
        if np.array_equal(x1, x2) is True and x1.all() == self.x.all():
            covmatrix = self.kernel.compute_covmatrix(yerr)
        else:
            covmatrix = self.kernel.compute_covmatrix(0.0)

        return covmatrix

    def internal_residuals(self):
        """
        Residuals internal to the computation:
            RV - RV_model

        Returns
        -------
        res : array
            New RVs for internal calculations
        """
        self.new_y = self.y - self.model_y
        res = self.new_y
        return res

    def residuals(self):
        """
        Residuals between the RV - model and the GP prediction for the plotting of GPs.
            RV - RV_model - predicted mean of GP noise model

        Returns
        -------
        res : array
            New RVs for GP plotting
        """
        mu_pred, _pred = self.predict(self.x)
        # self.predict comes from the predict function later on
        res = self.y - self.model_y - mu_pred
        return res

    def logprob(self):
        """
        Computes the natural logarith of the likelyhood of the gaussian fit.
        Following the equation:
            ln(L) = -n/2 ln(2pi) * -1/2 ln(det(K)) -1/2 Y.T dot K-1 dot Y

        Returns
        -------
        logL : float
            Ln of the likelihood
        """
        # Compute kernel covariance matrix and the y (rvs) to model
        K = self.compute_kernel(self.x, self.x)
        Y = self.internal_residuals()
        # Compute likelyhood, formula 2.28 in Raphie Thesis
        # Part 1: get ln of determinant
        sign, logdetK = np.linalg.slogdet(K)
        # Part 2: compute Y.T * K-1 * Y
        # print(np.linalg.eigvalsh(K))
        A = cho_solve(cho_factor(K), Y)
        alpha = np.dot(Y, A)
        # Part 3: all together
        N = len(Y)
        logprob = -(N / 2) * np.log(2 * np.pi) - 0.5 * logdetK - 0.5 * alpha
        self.logprob = logprob

        return logprob

    def priors(self, param_name, prior_name, prior_parameters):
        """
        Parameters
        ----------
        hparam_name : string
            Name of the hyperparameter on which to impose the prior
        prior_name : string
            Name of the prior to apply
        prior_parameters : dictionary
            Dictionary containing all necessary prior parameters
            (get with Prior_Par_Creator and assign values)

        Returns
        -------
        **logL_post : float
            Natural logarith of the likelihood after imposing the prior
        prior_logprob : float
            Natural logarith of the prior likelihood
        """

        try:
            self.prior_param = self.hparameters[param_name].value
        except KeyError:
            self.prior_param = self.model_param[param_name].value

        self.prior_name = prior_name
        self.prior_parameters = prior_parameters
        assert (
            self.prior_name in PRIORS.keys()
        ), "Prior not yet implemented. Pick from available priors: " + str(
            PRIORS.keys()
        )

        if prior_name.startswith("Gaussian") or prior_name.startswith("gaussian"):
            self.mu = self.prior_parameters["mu"]
            self.sigma = self.prior_parameters["sigma"]
            self.prior = Gaussian(param_name, self.mu, self.sigma)
            prior_logprob = self.prior.logprob(self.prior_param)

        if prior_name.startswith("Jeffrey") or prior_name.startswith("jeffrey"):
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.prior = Jeffrey(param_name, self.minval, self.maxval)
            prior_logprob = self.prior.logprob(self.prior_param)

        if prior_name.startswith("Modified") or prior_name.startswith("modified"):
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.kneeval = self.prior_parameters["kneeval"]
            self.prior = Modified_Jeffrey(
                param_name, self.minval, self.maxval, self.kneeval
            )
            prior_logprob = self.prior.logprob(self.prior_param)

        if prior_name.startswith("Uni") or prior_name.startswith("uni"):
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.prior = Uniform(param_name, self.minval, self.maxval)
            prior_logprob = self.prior.logprob(self.prior_param)

        return prior_logprob

    def LogL(self, prior_list):
        """
        Parameters
        ----------
        prior_list : list of sets of 3 objects
            List of the priors applied. Each item in the list should countain the following
            3 objects:
                String of the name of the parameter the prior is applied to
                String of the name of the prior
                Prior_Par_Creator dictionary of the prior

        Returns
        -------
        LogL : float
            Final ln of likelyhood after applying all posteriors from priors
        """

        LogL = self.logprob()
        for i in range(len(prior_list)):
            hparam = prior_list[i][0]
            name_prior = prior_list[i][1]
            prior_param = prior_list[i][2]
            LogL += self.priors(hparam, name_prior, prior_param)

        return LogL

    def predict(self, xpred, FullCov=False):
        """
        Parameters
        ----------
        xpred : array, floats
            X array over which to do the prediction
        FullCov : True/False, optional
            Want to return the full covariance? The default is False.

        Returns
        -------
        pred_mean: array, floats
            Predicted values of the y axis
        stdev: array, floats
            Standard deviation of the y points, to be used as error
        OR
        np.array(pred_cov): array, floats
            Full covariance of the data set
        """

        Y = self.internal_residuals()
        y = Y.T

        K = self.compute_kernel(self.x, self.x)
        Ks = self.compute_kernel(xpred, self.x)
        """kernel_name = self.kernel_name
        if kernel_name.startswith("QuasiPer") or kernel_name.startswith("quasiper") or kernel_name.startswith("Quasiper"):
            self.kernel = QuasiPer(self.hparameters)
        self.kernel.compute_distances(xpred, xpred)
        covmatrix = self.kernel.compute_covmatrix(0.)
        Kss = covmatrix
        #Kss = self.compute_kernel(xpred, xpred)
        print("Kss", Kss)"""
        Kss = self.compute_kernel(xpred, xpred)

        # Predicted mean = Ks * K-1 * y
        alpha = cho_solve(cho_factor(K), y)
        pred_mean = np.dot(Ks, alpha).flatten()
        pred_mean = np.array(pred_mean)

        # Predicted errors = Kss - Ks * K-1 * Ks.T
        beta = cho_solve(cho_factor(K), Ks.T)
        pred_cov = Kss - np.dot(Ks, beta)

        # print('Kss = ', Kss)
        # print("Ks", Ks)
        """Kinv = np.linalg.inv( np.matrix(K) )
        pred_cov2 = Kss - Ks * Kinv * Ks.T"""
        # print("second pred = ", pred_cov2)

        # print("pred cov,", pred_cov)

        plots = False
        if plots is True:
            """import matplotlib.gridspec as gridspec
            fig = plt.figure()
            spec = gridspec.GridSpec(ncols = 3, nrows = 1, figure = fig)
            ax1 = fig.add_subplot(spec[0,0])
            ax2 = fig.add_subplot(spec[0,1])
            ax3 = fig.add_subplot(spec[0,2])"""

            from matplotlib.colors import LogNorm

            fig, [ax1, ax2, ax3] = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(10, 6),
                gridspec_kw={"width_ratios": [1, 1, 1.7]},
            )

            vmin = 0
            vmax1 = max([max(l) for l in K])
            vmax2 = max([max(o) for o in Ks])
            vmax3 = max([max(p) for p in Kss])
            vmaxs = np.array([vmax1, vmax2, vmax3])
            vmaxs.sort()
            vmax = vmaxs[-1]

            ax1.imshow(K, vmin=vmin, vmax=vmax)
            ax1.title.set_text("K")
            ax2.imshow(Ks, vmin=vmin, vmax=vmax)
            ax2.title.set_text("Ks")
            im3 = ax3.imshow(Kss, vmin=vmin, vmax=vmax)
            ax3.title.set_text("Kss")

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            fig.colorbar(im3, cax=cax)
            plt.show()

        if FullCov is True:
            return pred_mean, np.array(pred_cov)
        else:
            var = np.array(np.diag(pred_cov)).flatten()
            # print("var", var)
            stdev = np.sqrt(var)
            # print("stdev", stdev)
            stdev = np.array(stdev)
            return pred_mean, stdev

        # imshow cov matrices
