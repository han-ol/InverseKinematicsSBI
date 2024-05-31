from functools import partial

import numpy as np
import scipy


class BenchmarkRobot:
    def __init__(self):
        self.l = np.array([0.5, 0.5, 1.0])
        self.sigmas = [[0.25, 0.5, 0.5, 0.5]]
        self.distribution = scipy.stats.norm(np.zeros_like(self.sigmas), self.sigmas)

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
        end   : np.array (batch_size, n_end)
            Coordinates of the end-effector.
        """
        return self.forward_reduced(params, [0, 1, 2, 3], only_end=only_end)

    def forward_reduced(self, params, indices, only_end=True):
        if not only_end:
            coords = np.empty((len(params), 4, 3))

        operations = [
            lambda pose_: np.stack((pose_[:, 0], pose_[:, 1] + params[:, 0], pose_[:, 2]), axis=1),
        ] + [
            partial(
                lambda pose_, i_: np.stack(
                    (
                        pose_[:, 0] + self.l[i_ - 1] * np.cos(pose[:, 2] + params[:, i_]),
                        pose_[:, 1] + self.l[i_ - 1] * np.sin(pose[:, 2] + params[:, i_]),
                        pose_[:, 2] + params[:, i_],
                    ),
                    axis=1,
                ),
                i_=i,
            )
            for i in range(1, 4)
        ]

        pose = np.zeros((len(params), 3))

        for i, operation in enumerate(operations):
            if i in indices:
                pose = operation(pose)
            if not only_end:
                coords[:, i, :] = pose
        if only_end:
            return pose
        else:
            return coords

    def sample_prior(self, n_samples):
        """Sample from the prior distributions of the components

        Parameters
        ----------
        n_samples       : int
            number of samples to draw

        Returns
        -------
        prior_samples: list of length (n_components) with an
            array of shape (n_samples, n_params) for each component
        """
        return self.distribution.rvs((n_samples, 4))

    def sample(self, n_samples, only_end=True):
        """Sample from the forward model

        Parameters
        ----------
        n_samples       : int
            number of samples to draw
        only_end        : bool, optional, default=True
            include only the endpoint, not intermediate positions

        Returns
        -------
        prior_samples: list of length (n_components-n_exclude_last) with an
            array of shape (n_samples, n_params) for each component
        coords: array of shape (n_samples, n_dim)
            End points of the forward process.
        """
        prior_samples = self.sample_prior(n_samples)
        coords = self.forward(prior_samples, only_end=only_end)
        return prior_samples, coords

    def sample_posterior(self, y, n_samples, max_n_batches=1000, weights=None):
        """Sample from the posterior distribution

        Parameters
        ----------
        y               : array of dimension (n_dim,)
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
        samples = np.zeros((n_samples, 4))
        total_accepted = 0
        indices = [[0, 1], [1, 2], [2, 3]]
        complement_indices = [[2, 3], [0, 3], [0, 1]]
        if weights is None:
            index_weights = [1, 1, 1]
        else:
            index_weights = weights
        for i in range(max_n_batches):
            # Sample params_1, params_2 from the prior
            params = self._sample_proposal_distribution(
                y,
                np.maximum(100 * n_samples, 1_000_000),
                n_samples_per_batch=100 * n_samples,
                max_n_batches=max_n_batches,
                index_weights=index_weights,
            )

            weights = self._get_weights(params, indices, complement_indices, index_weights)

            num_missing = n_samples - total_accepted
            p = weights / np.max(weights)
            accepted_mask = (np.random.binomial(1, p) == 1).flatten()
            print(f"Acceptance Rate: {p.sum()/len(params)}")
            n_accepted = min(accepted_mask.sum(), num_missing)
            accepted_params = params[accepted_mask]
            samples[total_accepted : min(total_accepted + n_accepted, n_samples), :] = accepted_params[
                : min(n_accepted, n_samples - total_accepted)
            ]
            total_accepted += n_accepted
            if total_accepted == n_samples:
                return samples
            print(total_accepted)
        raise RuntimeError(
            f"""Only {total_accepted} samples could be produced. Please increase n_samples_per_batch or max_n_batches
            """
        )

    def _sample_proposal_distribution(
        self, y, n_samples, n_samples_per_batch=100_000, max_n_batches=100, index_weights=None
    ):
        if index_weights is None:
            index_weights = [1, 1, 1]
        index_ps = index_weights / np.sum(index_weights)
        samples = np.zeros((n_samples, 4))
        dim_indices = np.zeros((n_samples, 2), dtype=int)
        total_accepted = 0
        possible_indices_map = np.array([[0, 1], [1, 2], [2, 3]])
        for i in range(max_n_batches):
            red_prior_samples = self.sample_prior(n_samples_per_batch)
            red_dim_indices = possible_indices_map[np.random.choice(3, size=n_samples_per_batch, p=index_ps)]
            is_reachable = self._check_reachable(red_prior_samples, red_dim_indices, y)
            n_accepted = np.sum(is_reachable)

            full_samples = self._get_full_params(red_prior_samples[is_reachable], red_dim_indices[is_reachable], y)
            full_dim_indices = red_dim_indices[is_reachable]
            samples[total_accepted : min(total_accepted + n_accepted, n_samples)] = full_samples[
                : min(n_accepted, n_samples - total_accepted)
            ]
            dim_indices[total_accepted : min(total_accepted + n_accepted, n_samples)] = full_dim_indices[
                : min(n_accepted, n_samples - total_accepted)
            ]
            total_accepted += n_accepted

            # Reject samples for which the end effector is not in range
            if total_accepted > n_samples:
                return samples
        raise RuntimeError(
            f"""Only {total_accepted} samples could be produced. If zero samples were produced,
                    please ensure that the observation point is reachable for the given lengths.
                    If too few samples were produced, please increase n_samples_per_batch or max_n_batches
                    """
        )

    def _check_reachable(self, params, indices, y):
        "Check if the end effector is in range for a batch of paramss and xs."
        is_reachable = np.full(len(params), False, dtype=bool)
        if np.any(np.all(indices == np.array([[0, 1]]), axis=1)):
            is_this_index = np.all(indices == np.array([[0, 1]]), axis=1)
            c = np.linalg.norm(self.forward_reduced(params[is_this_index], np.array([0, 1]))[:, :2] - y, axis=1)
            a = self.l[1]
            b = self.l[2]
            is_reachable[is_this_index] = check_triangle(a, b, c)
        if np.any(np.all(indices == np.array([[1, 2]]), axis=1)):
            is_this_index = np.all(indices == np.array([[1, 2]]), axis=1)
            pose = self.forward_reduced(params[is_this_index], np.array([1, 2]))
            is_reachable[is_this_index] = np.abs(pose[:, 0] - y[0]) < self.l[2]
        if np.any(np.all(indices == np.array([[2, 3]]), axis=1)):
            is_this_index = np.all(indices == np.array([[2, 3]]), axis=1)
            pose = self.forward_reduced(params[is_this_index], np.array([2, 3]))
            add_shift = np.zeros_like(pose[:, :2])
            add_shift[:, 0] = self.l[0]
            pose_length = np.linalg.norm(add_shift + pose[:, :2], axis=1)
            is_reachable[is_this_index] = np.abs(y[0]) < pose_length
        return is_reachable

    def _get_full_params(self, params_wrong, indices, y):
        full_params = np.zeros((len(params_wrong), 4))
        if np.any(np.all(indices == np.array([[0, 1]]), axis=1)):
            is_this_index = np.all(indices == np.array([[0, 1]]), axis=1)
            full_params[is_this_index] = self._get_full_params01(params_wrong[is_this_index], y)
        if np.any(np.all(indices == np.array([[1, 2]]), axis=1)):
            is_this_index = np.all(indices == np.array([[1, 2]]), axis=1)
            full_params[is_this_index] = self._get_full_params12(params_wrong[is_this_index], y)
        if np.any(np.all(indices == np.array([[2, 3]]), axis=1)):
            is_this_index = np.all(indices == np.array([[2, 3]]), axis=1)
            full_params[is_this_index] = self._get_full_params23(params_wrong[is_this_index], y)
        return full_params

    def _get_full_params01(self, params_wrong, y):
        "Get the full params from the reduced params and end-effector position x."
        z = self.forward_reduced(params_wrong, np.array([0, 1]))
        a, b = self.l[1], self.l[2]
        gap = y - z[:, :2]
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

        params_wrong[:, 2] = param_a
        params_wrong[:, 3] = param_b

        return params_wrong

    def _get_full_params12(self, params_wrong, y):
        z = self.forward_reduced(params_wrong, np.array([1, 2]))
        d_0 = y[0] - z[:, 0]
        d_1 = np.random.choice([-1, 1], size=(z.shape[0])) * np.sqrt(self.l[2] ** 2 - d_0**2)
        param_0 = y[1] - z[:, 1] - d_1
        param_3 = np.arctan2(d_1, d_0) - z[:, 2]

        params_wrong[:, 0] = param_0
        params_wrong[:, 3] = param_3

        return params_wrong

    def _get_full_params23(self, params_wrong, y):
        z = self.forward_reduced(params_wrong, np.array([2, 3]))
        add_shift = np.zeros_like(z[:, :2])
        add_shift[:, 0] = self.l[0]
        z_shifted = z[:, :2] + add_shift
        l_z = np.linalg.norm(z_shifted, axis=1)
        angle_desired = np.random.choice([-1, 1], size=(z.shape[0])) * np.arccos(y[0] / l_z)
        angle_delta = angle_desired - np.arctan2(z_shifted[:, 1], z_shifted[:, 0])
        param_1 = angle_delta
        param_0 = y[1] - np.sin(angle_desired) * l_z

        params_wrong[:, 0] = param_0
        params_wrong[:, 1] = param_1

        return params_wrong

    def _get_jacobian(self, params):
        l1s2 = self.l[0] * np.sin(params[:, 1])
        l2s23 = self.l[1] * np.sin(np.sum(params[:, 1:3], axis=1))
        l3s234 = self.l[2] * np.sin(np.sum(params[:, 1:4], axis=1))
        l1c2 = self.l[0] * np.cos(params[:, 1])
        l2c23 = self.l[1] * np.cos(np.sum(params[:, 1:3], axis=1))
        l3c234 = self.l[2] * np.cos(np.sum(params[:, 1:4], axis=1))
        jacobian = np.zeros((len(params), 2, 4))
        jacobian[:, 0, 0] = np.ones(len(params))
        jacobian[:, 1, 0] = np.zeros(len(params))
        jacobian[:, 0, 1] = l1c2 + l2c23 + l3c234
        jacobian[:, 1, 1] = -(l1s2 + l2s23 + l3s234)
        jacobian[:, 0, 2] = l2c23 + l3c234
        jacobian[:, 1, 2] = -(l2s23 + l3s234)
        jacobian[:, 0, 3] = l3c234
        jacobian[:, 1, 3] = -(l3s234)
        return jacobian

    def _get_jacobian_minor(self, params, indices):
        sub_jacobians = self._get_jacobian(params)[:, :, indices]
        return np.linalg.det(sub_jacobians)

    def _complement_prior_pdf(self, params, indices=None):
        pdfs = self.distribution.pdf(params)
        if indices is None:
            return np.prod(pdfs, axis=1)
        else:
            return np.prod(pdfs[:, indices], axis=1)

    def _get_weights(self, params, indices, complement_indices, index_weights):
        divisor = np.zeros(len(params))
        for index, complement_index, index_weight in zip(indices, complement_indices, index_weights):
            divisor += (
                index_weight
                * self._complement_prior_pdf(params, index)
                * np.abs(self._get_jacobian_minor(params, complement_index))
            )

        dividend = self._complement_prior_pdf(params)
        weights = dividend / divisor
        return weights


def check_triangle(a, b, c):
    return (a < b + c) & (b < a + c) & (c < a + b)
