from inverse_kinematics.helpers import mask_sample_dict, sample_dist_dict, concat_sample_dicts 

import numpy as np


def distance_end_effector_position(coord, obs):
    return np.linalg.norm(coord[:, :2] - obs, axis=1)

def abc_reject_mask(forward_fun, prior_samples, distance_fun, target, tolerance=0.1):
    forward_out = forward_fun(prior_samples)
    close_enough = distance_fun(forward_out, target)<tolerance
    return close_enough
    #return mask_sample_dict(prior_samples, close_enough)

def draw_abc_samples(n_samples, forward_fun, prior, distance_fun, target, tolerance, verbose=False):

    # estimate acceptance rate
    n_init = n_samples*100
    init_proposals = sample_dist_dict(prior, n_init) 
    init_acc_mask = abc_reject_mask(forward_fun, init_proposals, distance_fun, target, tolerance)
    acc_rate = float(init_acc_mask.mean())

    print(acc_rate)
    # propose a little extra to reach `n_samples` accepted ABC posterior samples
    eps = 0.000000001
    n_propose = int(n_samples / (acc_rate+eps) * 2)
    if verbose:
        print(f"Need {n_propose} proposals to get probably {n_samples} ABC samples for tolerance {tolerance}")

    # batched collection of samples
    batch_size = 1_000_000
    acc_list = []
    for i in range(n_propose//batch_size+1):
        proposal_batch = sample_dist_dict(prior, min(n_propose, batch_size))
        acc_mask = abc_reject_mask(forward_fun, proposal_batch, distance_fun, target, tolerance)
        acc_list.append(mask_sample_dict(proposal_batch, acc_mask))
        if verbose:
            print(f'{acc_mask.shape[0]} trials -> {acc_mask.sum()} accepted' ) 

    abc_samples = concat_sample_dicts(acc_list)
    n_total_accepted = list(abc_samples.values()).pop().shape[0]

    abc_samples = mask_sample_dict(abc_samples, slice(0,n_samples))
    if verbose:
        print(f'{n_total_accepted} total accepted proposals are cut down to the target of {n_samples}')

    assert n_total_accepted > n_samples, "Not enough proposals were accepted to reach the target of n_samples"
    return abc_samples

