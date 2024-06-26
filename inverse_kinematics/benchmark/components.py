from abc import ABC, abstractmethod

import numpy as np
import scipy


class Component(ABC):
    """Base class for all components of the robot arm"""

    @abstractmethod
    def forward(self, params, start_coord):
        """Calculates the forward kinematics of the component

        Parameters
        ----------
        params: array of shape (n_samples, n_params)
            The parameters for the component
        coord: array of shape (n_samples, n_dim)
            The starting coordinate of the component

        Returns
        -------
        prior_draws: array of shape (n_sampes, n_params)
        """
        pass


class RobotArm:
    def __init__(self, components):
        self.components = components
        self._validate_components()

    def forward(self, params, start_coord=None, only_end=True):
        """
        Parameters
        ----------
        start           : (batch_size, start_coord_dim)
            Coordinates of where robot is fixated.
            (0, 0, 0) for a 2D robot indicates x=0, y=0, phi=0
        params          : (batch_size, n_params)
            Parameters for the components.
        start_coord     : array of shape (batch_size, n_dim), optional
            Start coordinate of the arm, default 0
        only_end        : bool, optional, default=True
            include only the endpoint, not intermediate positions

        Returns
        -------
        end   : (batch_size, n_end)
            Coordinates of the end-effector.
        """

        batch_size = params[0].shape[0]

        if start_coord is None:
            start_coord = np.zeros((batch_size, 3))
        else:
            assert start_coord.shape == (batch_size, 3)

        coord = start_coord.copy()

        if not only_end:
            coords = np.empty((start_coord.shape[0], len(params) + 1, start_coord.shape[1]))
            coords[:, 0, :] = coord

        for i, p in enumerate(params):
            coord = self.components[i].forward(p, coord)

            if not only_end:
                coords[:, i + 1, :] = coord

        if only_end:
            return coord
        return coords

    def sample_prior(self, n_samples, n_exclude_last=0):
        """Sample from the prior distributions of the components

        Parameters
        ----------
        n_samples       : int
            number of samples to draw
        n_exclude_last  : int, optional, default=0
            Do not include samples for the last n components

        Returns
        -------
        prior_samples: list of length (n_components-n_exclude_last) with an
            array of shape (n_samples, n_params) for each component
        """
        return [
            c.prior_dist.rvs((n_samples, c.n_params))
            for c in self.components[: (len(self.components) - n_exclude_last)]
        ]

    def sample(self, n_samples, only_end=True, n_exclude_last=0):
        """Sample from the forward model

        Parameters
        ----------
        n_samples       : int
            number of samples to draw
        only_end        : bool, optional, default=True
            include only the endpoint, not intermediate positions
        n_exclude_last  : int, optional, default=0
            Do not include prior samples for the last n components and stop
            forward process n components earlier.

        Returns
        -------
        prior_samples: list of length (n_components-n_exclude_last) with an
            array of shape (n_samples, n_params) for each component
        coords: array of shape (n_samples, n_dim)
            End points of the forward process.
        """
        prior_samples = self.sample_prior(n_samples, n_exclude_last=n_exclude_last)
        coords = self.forward(prior_samples, only_end=only_end)
        return prior_samples, coords

    def sample_posterior(self, x, n_samples, max_n_batches=1000):
        """Sample from the posterior distribution

        Parameters
        ----------
        x               : array of dimension (n_dim,)
            End point of the end effector
        n_samples       : int
            number of samples to draw
        max_n_batches   : int, optional, default=1000
            Upper limit to the number of batches computed. If sampling does
            not succeed even though the end point is reachable by the arm,
            increase this number.

        Returns
        -------
        posterior_samples: list of length n_components with an
            array of shape (n_samples, n_params) for each component
        """
        samples = [np.full((n_samples, c.n_params), np.nan) for c in self.components]
        total_accepted = 0
        for i in range(max_n_batches):
            # Sample params_1, params_2 from the prior
            accepted_red_prior_samples = self._sample_reduced_batched(
                x, np.maximum(100 * n_samples, 10_000), n_samples_per_batch=100 * n_samples, max_n_batches=max_n_batches
            )

            # Explicitly calculate preimages param_a and param_b of x given params_1, params_2
            params = self._get_full_params(accepted_red_prior_samples, x)

            # Weight the samples by their probability under the prior
            # Compute Jacobi determinant of the transformation

            deform_factor = self._get_jacobian_determinant(params, self.forward(params[:-2]))

            weights = self._complement_prior_pdf(params[-2:])[:, 0] * deform_factor
            num_missing = n_samples - total_accepted
            p = weights / np.max(weights)
            accepted_mask = (np.random.binomial(1, p) == 1).flatten()
            num_accepted = min(accepted_mask.sum(), num_missing)
            if num_accepted > 0:
                for i in range(len(samples)):
                    samples[i][total_accepted : (total_accepted + num_accepted)] = params[i][accepted_mask][
                        :num_accepted
                    ]
                total_accepted += num_accepted
            if total_accepted == n_samples:
                return samples
        raise RuntimeError(
            f"""Only {total_accepted} samples could be produced. Please increase n_samples_per_batch or max_n_batches
            """
        )

    def _validate_components(self):
        if not (isinstance(self.components[-1], Joint) and isinstance(self.components[-2], Joint)):
            print(
                "Warning: efficient posterior sample estimation currently only available if last two components are joints"
            )

    def _check_reachable(self, params_reduced, x):
        "Check if the end effector is in range for a batch of paramss and xs."
        z = self.forward(params_reduced)[:, :2]
        a, b = self.components[-2].length, self.components[-1].length
        gap = np.linalg.norm(x - z, axis=1)
        max_l = np.maximum(gap, np.maximum(a, b)[None,])
        return gap + a + b - max_l >= max_l

    def _get_full_params(self, params_reduced, x):
        "Get the full params from the reduced params and end-effector position x."
        z = self.forward(params_reduced)
        a, b = self.components[-2].length, self.components[-1].length
        gap = x - z[:, :2]
        eta = np.arcsin(gap[:, 1] / np.linalg.norm(gap, axis=1))
        eta = np.where(gap[:, 0] < 0, -eta - np.pi, eta)

        gamma = np.arccos((a**2 + np.linalg.norm(gap, axis=1) ** 2 - b**2) / (2 * a * np.linalg.norm(gap, axis=1)))

        gamma_prime = np.arccos(
            (b**2 + np.linalg.norm(gap, axis=1) ** 2 - a**2) / (2 * b * np.linalg.norm(gap, axis=1))
        )
        param_a = np.mod(-z[:, -1] + eta - gamma * np.array([1, -1])[:, None], 2 * np.pi)
        param_a = np.where(param_a > np.pi, param_a - 2 * np.pi, param_a)
        param_b = np.mod(-z[:, -1] - param_a + eta + gamma_prime * np.array([1, -1])[:, None], 2 * np.pi)
        param_b = np.where(param_b > np.pi, param_b - 2 * np.pi, param_b)

        # randomly choose one of the two possible solutions
        selection = np.random.randint(0, 2, size=(z.shape[0]))
        param_a = np.where(selection, param_a[0], param_a[1])
        param_b = np.where(selection, param_b[0], param_b[1])

        return params_reduced + [param_a[:, None], param_b[:, None]]

    def _get_jacobian_determinant(self, params, z):
        la, lb = self.components[-2].length, self.components[-1].length

        dy1da = -la * np.sin(z[:, 2] + params[-2][:, 0]) - lb * np.sin(z[:, 2] + params[-2][:, 0] + params[-1][:, 0])
        dy1db = -lb * np.sin(z[:, 2] + params[-2][:, 0] + params[-1][:, 0])
        dy2da = la * np.cos(z[:, 2] + params[-2][:, 0]) + lb * np.cos(z[:, 2] + params[-2][:, 0] + params[-1][:, 0])
        dy2db = lb * np.cos(z[:, 2] + +params[-2][:, 0] + params[-1][:, 0])

        return 1 / np.abs(dy1da * dy2db - dy2da * dy1db)

    def _complement_prior_pdf(self, complement_samples):
        n_components = len(self.components)
        n_complements = len(complement_samples)
        pdfs = [
            self.components[n_components - n_complements + i].prior_dist.pdf(complement_samples[i])
            for i in range(n_complements)
        ]
        return np.prod(pdfs, axis=0)

    def _sample_reduced_batched(self, x, n_samples, n_samples_per_batch=100_000, max_n_batches=100):
        samples = [np.full((n_samples, c.n_params), np.nan) for c in self.components[:-2]]
        total_accepted = 0
        for i in range(max_n_batches):
            red_prior_samples = self.sample_prior(n_samples_per_batch, n_exclude_last=2)

            # Reject samples for which the end effector is not in range
            n_missing = n_samples - total_accepted
            accepted_red_prior_samples = [
                red_prior_samples[i][self._check_reachable(red_prior_samples, x)][:n_missing]
                for i in range(len(red_prior_samples))
            ]
            num_accepted = accepted_red_prior_samples[0].shape[0]
            if num_accepted > 0:
                for i in range(len(accepted_red_prior_samples)):
                    samples[i][total_accepted : (total_accepted + num_accepted)] = accepted_red_prior_samples[i]
                total_accepted += num_accepted
            if total_accepted == n_samples:
                return samples
        raise RuntimeError(
            f"""Only {total_accepted} samples could be produced. If zero samples were produced,
            please ensure that the observation point is reachable for the given lengths.
            If too few samples were produced, please increase n_samples_per_batch or max_n_batches
            """
        )


class Rail(Component):
    def __init__(self, prior_std, fix_rot=0):
        self.fix_rot = fix_rot
        self.n_params = 1
        self.prior_dist = scipy.stats.norm(0, prior_std)

    def forward(self, params, start_coord):
        assert len(params.shape) == 2
        assert params.shape[1] == self.n_params
        batch_size = params.shape[0]
        assert start_coord.shape == (batch_size, 3)

        coord = start_coord.copy()

        coord[:, 0] -= np.sin(coord[:, 2]) * params[:, 0]
        coord[:, 1] += np.cos(coord[:, 2]) * params[:, 0]
        coord[:, 2] += self.fix_rot

        return coord


class Joint(Component):
    def __init__(self, length, prior_std):
        self.length = length
        self.n_params = 1
        loc, scale = 0, prior_std
        trunc_lower, trunc_upper = -np.pi, np.pi
        a, b = (trunc_lower - loc) / scale, (trunc_upper - loc) / scale
        self.prior_dist = scipy.stats.truncnorm(loc=loc, scale=scale, a=a, b=b)

    def forward(self, params, start_coord):
        assert len(params.shape) == 2
        assert params.shape[1] == self.n_params
        batch_size = params.shape[0]
        assert start_coord.shape == (batch_size, 3)

        coord = start_coord.copy()

        coord[:, 0] += self.length * np.cos(coord[:, 2] + params[:, 0])
        coord[:, 1] += self.length * np.sin(coord[:, 2] + params[:, 0])
        coord[:, 2] += params[:, 0]

        return coord
