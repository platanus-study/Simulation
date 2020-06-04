#!/usr/bin/env python3
#
# Pints Boundaries that limit the transition rates in the Beattie et al model.
#
from __future__ import division, print_function
import numpy as np
import pints

# Load project modules
import transformations


class Boundaries(pints.Boundaries):
    """
    Boundary constraints on the parameters.

    Arguments:

    ``lower_conductance``
        The lower bound on conductance to use. The upper bound will be set as
        ten times the lower bound.
        Set to ``None`` to use an 8-parameter boundary.
    ``search_transformation``
        A transformation on the parameter space.
        Calls to :meth:`check(p)` will assume ``p`` is in the transformed
        space. Similarly, :meth:`sample()` will return samples in the
        transformed space (although the type of sampling will depend on the
        ``sample_transformation``.
    ``sample_transformation``
        A transformation object, specifying the space to sample in.

    """
    def __init__(
            self, search_transformation, sample_transformation,
            lower_conductance=None):

        super(Boundaries, self).__init__()

        # Include conductance parameter
        self._conductance = (lower_conductance is not None)

        # Parameter transformations
        self._search_transformation = search_transformation
        self._sample_transformation_code = sample_transformation.code()

        # Conductance limits
        if self._conductance:
            self.lower_conductance = lower_conductance
            self.upper_conductance = 10 * lower_conductance

        # Limits on p1-p24
        self.lower_alpha = 1e-8             # Kylie: 1e-7
        self.upper_alpha = 1e3              # Kylie: 1e3
        self.lower_beta = 1e-8              # Kylie: 1e-7
        self.upper_beta = 3e-1               # Kylie: 0.4

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ]
        self.upper = [
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ]

        if self._conductance:
            self.lower.append(self.lower_conductance)
            self.upper.append(self.upper_conductance)

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-8
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 25 if self._conductance else 24

    def check(self, transformed_parameters):

        debug = False

        # Transform parameters back to model space
        parameters = self._search_transformation.detransform(
            transformed_parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug:
                print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug:
                print('Upper')
            return False

        # Check maximum rate constants
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, \
        p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, \
        p23, p24 = parameters[:24]

        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r2')
            return False
        r = p9 * np.exp(p10 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False
        r = p13 * np.exp(p14 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                 print('r4')
            return False
        r = p15 * np.exp(p16 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r5')
            return False
        r = p17 * np.exp(p18 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r6')
            return False
        r = p21 * np.exp(p22 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r7')
            return False


        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r8')
            return False
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r9')
            return False
        r = p11 * np.exp(-p12 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r10')
            return False
        r = p19 * np.exp(-p20 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r11')
            return False
        r = p23 * np.exp(-p24 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r12')
            return False 

        return True

    def _sample_partial(self, v):
        """
        Sample a pair of parameters, uniformly in the a-transformed space, that
        satisfy the maximum transition rate constraints.
        """
        if self._sample_transformation_code == 'a':
            for i in range(100):
                a = np.exp(np.random.uniform(
                        np.log(self.lower_alpha), np.log(self.upper_alpha)))
                b = np.random.uniform(self.lower_beta, self.upper_beta)
                r = a * np.exp(b * v)
                if r >= self.rmin and r <= self.rmax:
                    return a, b
            raise ValueError('Too many iterations')
        elif self._sample_transformation_code == 'n':
            for i in range(1000):
                a = np.random.uniform(self.lower_alpha, self.upper_alpha)
                b = np.random.uniform(self.lower_beta, self.upper_beta)
                r = a * np.exp(b * v)
                if r >= self.rmin and r <= self.rmax:
                    return a, b
        elif self._sample_transformation_code in ['f', 'k']:
            for i in range(100):
                a = np.exp(np.random.uniform(
                        np.log(self.lower_alpha), np.log(self.upper_alpha)))
                b = np.exp(np.random.uniform(
                        np.log(self.lower_beta), np.log(self.upper_beta)))
                r = a * np.exp(b * v)
                if r >= self.rmin and r <= self.rmax:
                    return a, b
            raise ValueError('Too many iterations')
        else:
            raise ValueError(
                'Unknown transformation code: '
                + str(self._sample_transformation_code))

    def _sample_conductance(self):
        """
        Samples a conductance.
        """
        if self._sample_transformation_code in ['a', 'n', 'k']:
            return np.random.uniform(
                self.lower_conductance, self.upper_conductance)
        elif self._sample_transformation_code == 'f':
            return np.exp(np.random.uniform(
                np.log(self.lower_conductance), np.log(self.upper_conductance)
                ))
        else:
            raise ValueError(
                'Unknown transformation code: '
                + str(self._sample_transformation_code))

    def sample(self, n=1):

        if n > 1:
            raise NotImplementedError

        p = np.zeros(25 if self._conductance else 24)

        # Sample forward rates
        p[0:2] = self._sample_partial(self.vmax)
        p[4:6] = self._sample_partial(self.vmax)
        p[8:10] = self._sample_partial(self.vmax)
        p[12:14] = self._sample_partial(self.vmax)
        p[14:16] = self._sample_partial(self.vmax)
        p[16:18] = self._sample_partial(self.vmax)
        p[20:22] = self._sample_partial(self.vmax)

        # Sample backward rates
        p[2:4] = self._sample_partial(-self.vmin)
        p[6:8] = self._sample_partial(-self.vmin)
        p[10:12] = self._sample_partial(-self.vmin)
        p[18:20] = self._sample_partial(-self.vmin)
        p[22:24] = self._sample_partial(-self.vmin)

        # Sample conductance
        if self._conductance:
            p[24] = self._sample_conductance()

        # Transform from model to search space
        p = self._search_transformation.transform(p)

        # The Boundaries interface requires a matrix ``(n, n_parameters)``
        p.reshape(1, 25 if self._conductance else 24)
        return p
