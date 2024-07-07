from abc import ABC
from itertools import combinations

import numpy as np

from src.inverse_kinematics_sbi.trigonometry import se2_action, check_reachable_joint_joint, \
    get_missing_params_joint_joint, check_reachable_joint_rail, check_reachable_rail_joint, check_reachable_rail_rail, \
    get_missing_params_joint_rail, get_missing_params_rail_joint, get_missing_params_rail_rail, se2_inverse


class ProbaRobot(ABC):

    def __init__(self, robot, prior, proposal=None):
        self.robot = robot
        self.prior = prior
        self.proposal = proposal if proposal is not None else prior

    def sample_prior(self, n_sample):
        return self.prior.rvs((n_sample, self.robot.get_n_params()))

    def sample_evidence(self, n_sample):
        return self.sample_likelihood(self.sample_prior(n_sample))

    def sample_joint(self, n_sample):
        prior_sample = self.sample_prior(n_sample)
        return self.sample_likelihood(prior_sample), prior_sample

    def sample_likelihood(self, params, n_sample=1, keep_dims=False):
        # if keep_dims=True and n_sample=1 returns shape: (len(params), n_sample, n_params)
        # if params of shape (n_params) returns (n_sample, n_params)
        if keep_dims:
            return self.robot.forward(np.tile(params[:, None, :], reps=(1, n_sample, 1)))
        elif not keep_dims and n_sample == 1:
            return self.robot.forward(params)

    def sample_posterior(self, observations, n_sample=1, keep_dims=False):
        # if keep_dims=True and n_sample=1 returns shape: (len(params), 1, n_params)
        # if keep_dims=True and params of shape (n_params) returns (1, n_sample, n_params)
        self.robot.check_components()
        samples = np.zeros((len(observations), n_sample, 4))
        indices = combinations(range(self.robot.get_n_params()), r=2)
        max_n_batches = 1000
        n_propose_total = 1_000_000
        min_propose = 10
        count_accepted = np.zeros(len(observations))
        for i in range(max_n_batches):
            required_mask = count_accepted < n_sample
            if not np.any(required_mask):
                break
            n_required = np.sum(required_mask)
            n_propose = min(max(int(n_propose_total/n_required), min_propose), 10 * n_sample)
            reach_proposed_params = self._sample_reachable_proposal_distribution(
                observations=observations[required_mask],
                n_sample=n_propose
            )

            weights = self._get_weights(reach_proposed_params)

            p = weights / np.max(weights, axis=-1, keepdims=True)
            accepted_mask = np.random.binomial(1, p) == 1

            update_samples(samples, reach_proposed_params, count_accepted, accepted_mask, required_mask)
        return samples

    def _sample_reachable_proposal_distribution(self, observations, n_sample=1000, n_propose_total=1_000_000, indices=None):
        min_propose = 10
        n_params = self.robot.get_n_params()
        parameters = np.zeros((len(observations), n_sample, n_params))
        if indices is None:
            indices = np.array(list(combinations(range(self.robot.get_n_params()), r=2)))
        max_n_batches = 1000
        counts_accepted = np.zeros(len(observations))
        for i in range(max_n_batches):
            required_mask = counts_accepted < n_sample
            if not np.any(required_mask):
                break
            n_required = np.sum(required_mask)
            n_propose = min(max(int(n_propose_total/n_required), min_propose), 10*n_sample)
            proposed_parameters = self.proposal.rvs((n_required, n_propose, n_params))
            indices_numbers = np.random.choice(len(indices), size=(n_required, n_propose))
            proposed_indices = indices[indices_numbers]
            np.put_along_axis(proposed_parameters, proposed_indices, 0, axis=-1)
            proposed_parameters, is_reachable = self._get_reachable_params(proposed_parameters, proposed_indices, observations[required_mask])
            update_samples(parameters, proposed_parameters, counts_accepted, is_reachable, required_mask)

        return parameters

    def _get_reachable_params(self, params, indices, target):
        action_start_to_first = self.robot._forward_kinematics_reduced(params, np.full(params.shape[:-1], -1), indices[:, :, 0])
        action_first_to_second = self.robot._forward_kinematics_reduced(params, indices[:, :, 0], indices[:, :, 1])
        action_second_to_end = self.robot._forward_kinematics_reduced(params, indices[:, :, 1], np.full(params.shape[:-1], self.robot.get_n_params()))
        indices_is_joint = self.robot._get_index_is_joint()[indices]
        indices_is_rail = self.robot._get_index_is_rail()[indices]
        target_first = se2_action(target[:, None, :], se2_inverse(action_start_to_first))
        is_reachable = np.full(params.shape[:-1], False)
        checks = [check_reachable_joint_joint, check_reachable_joint_rail, check_reachable_rail_joint, check_reachable_rail_rail]
        get_missings = [get_missing_params_joint_joint, get_missing_params_joint_rail, get_missing_params_rail_joint, get_missing_params_rail_rail]
        first_conds = [indices_is_joint, indices_is_joint, indices_is_rail, indices_is_rail]
        second_conds = [indices_is_joint, indices_is_rail, indices_is_joint, indices_is_rail]
        params_new = params.copy().astype(np.float64)
        for check, get_missing, first_cond, second_cond in zip(checks, get_missings, first_conds, second_conds):
            fulfills_cond = np.nonzero(first_cond[:, :, 0] & second_cond[:, :, 1])
            if np.any(fulfills_cond):
                is_reachable_red = check(
                    action_first_to_second[fulfills_cond],
                    action_second_to_end[fulfills_cond],
                    target_first[fulfills_cond]
                )
                decider = np.random.choice(2, size=np.sum(is_reachable_red))
                param_first_reachable, param_second_reachable = get_missing(
                    action_first_to_second[fulfills_cond][is_reachable_red],
                    action_second_to_end[fulfills_cond][is_reachable_red],
                    target_first[fulfills_cond][is_reachable_red],
                    decider
                )
                params_new[
                    fulfills_cond[0][is_reachable_red],
                    fulfills_cond[1][is_reachable_red],
                    indices[fulfills_cond][:, 0][is_reachable_red]
                ] = param_first_reachable
                params_new[
                    fulfills_cond[0][is_reachable_red],
                    fulfills_cond[1][is_reachable_red],
                    indices[fulfills_cond][:, 1][is_reachable_red]
                ] = param_second_reachable
                is_reachable[fulfills_cond] = is_reachable_red

        return params_new, is_reachable


    def _get_jacobian_minor(self, params, indices):
        # shape: (..., 2, n_params)
        jacobian = self.robot.forward_jacobian(params)

        # shape: (..., 2, n_indices, 2) -> (..., n_indices, 2, 2)
        sub_jacobian = np.swapaxes(jacobian[..., indices], -3, -2)

        minors = np.linalg.det(sub_jacobian)
        return minors

    def _get_map_count_indices(self, indices):
        indices_is_rail = self.robot._get_index_is_rail()[indices]
        map_count = np.full(indices.shape[:-1], 2)
        map_count[np.all(indices_is_rail, axis=-1)] = 1
        return map_count

    def _proposal_pdf_marginal(self, params, indices):
        pdfs = np.tile(self.proposal.pdf(params)[..., None, :], reps=(params.ndim - 1) * (1,) + (len(indices),) + (1,))
        not_index = np.full(indices.shape[:-1] + (self.robot.get_n_params(),), False)
        np.put_along_axis(not_index, indices, True, axis=-1)
        pdfs[..., not_index] = 1

        return np.prod(pdfs, axis=-1)

    def _prior_pdf(self, params):
        pdfs = self.prior.pdf(params)

        return np.prod(pdfs, axis=-1)

    def _get_weights(self, reach_proposed_params, indices=None):
        if indices is None:
            indices = np.array(list(combinations(range(self.robot.get_n_params()), r=2)))
        # shape (n_params, n_propose, n_indices)
        minors = np.abs(self._get_jacobian_minor(reach_proposed_params, indices))
        # shape (n_indices)
        map_count = self._get_map_count_indices(indices)
        # shape (n_params, n_propose, n_indices)
        proposal_pdf = self._proposal_pdf_marginal(reach_proposed_params, indices)

        divisor = np.sum(minors*proposal_pdf/map_count[None, None, :], axis=-1)
        divided = self._prior_pdf(reach_proposed_params)
        weights = divided/divisor
        return weights

def update_samples(samples, new_samples, counts, accepted_mask, required_mask=None):
    n_sample = samples.shape[1]
    n_obs = samples.shape[0]
    count_new = np.sum(accepted_mask, axis=1)
    index_mask_samples = np.tile(np.arange(n_sample).reshape(1, -1), reps=(n_obs, 1))
    index_mask_samples = (counts[:, None] <= index_mask_samples) & (index_mask_samples < (counts + count_new)[:, None])
    if required_mask is not None:
        index_mask_samples = index_mask_samples & np.tile(required_mask[:, None], (1, n_sample))
    if required_mask is not None:
        index_mask_new_samples = accepted_mask & (np.cumsum(accepted_mask, axis=1) <= n_sample - counts[required_mask, None])
    else:
        index_mask_new_samples = accepted_mask & (np.cumsum(accepted_mask, axis=1) <= n_sample - counts[:, None])
    samples[index_mask_samples, :] = new_samples[index_mask_new_samples, :]
    counts[:] = np.minimum(counts + count_new, n_sample)