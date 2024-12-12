import numpy as np
from tqdm import tqdm


def distance_end_effector_position(coord, obs):
    return np.linalg.norm(coord[:, :] - obs, axis=1)


def abc_reject_mask(forward_fun, prior_samples, distance_fun, target, tolerance=0.1):
    forward_out = forward_fun(prior_samples)
    close_enough = distance_fun(forward_out, target) < tolerance
    return close_enough


def draw_abc_samples(n_samples, forward_fun, prior, distance_fun, target, tolerance, verbose=False):
    # estimate acceptance rate
    n_init = n_samples * 100
    init_proposals = prior(n_init)
    init_acc_mask = abc_reject_mask(forward_fun, init_proposals, distance_fun, target, tolerance)
    acc_rate = float(init_acc_mask.mean())

    print(acc_rate)
    # propose a little extra to reach `n_samples` accepted ABC posterior samples
    eps = 0.000000001
    n_propose = int(n_samples / (acc_rate + eps) * 2)
    if verbose:
        print(f"Need {n_propose} proposals to get probably {n_samples} ABC samples for tolerance {tolerance}")

    # batched collection of samples
    batch_size = 1_000_000
    acc_list = []
    p = 0
    n_sampled = 0
    if verbose:
        pbar = tqdm(total=n_samples)
    for i in range(n_propose // batch_size + 1):
        proposal_batch = prior(min(n_propose, batch_size))
        acc_mask = abc_reject_mask(forward_fun, proposal_batch, distance_fun, target, tolerance)
        acc_batch = proposal_batch[acc_mask]
        acc_list.append(acc_batch)
        if verbose:
            p = n_sampled / (len(acc_mask) + n_sampled) * p + np.sum(acc_mask) / (len(acc_mask) + n_sampled)
            n_sampled += len(acc_mask)
            pbar.update(len(acc_batch))
            pbar.set_postfix_str(f"{acc_mask.shape[0]} trials -> {len(acc_batch)} accepted, acc rate {p}")
        if sum((len(acc_batch) for acc_batch in acc_list)) > n_samples:
            break

    abc_samples = np.concatenate(acc_list, axis=0)
    n_total_accepted = len(abc_samples)

    abc_samples = abc_samples[: min(n_total_accepted, n_samples)]
    if verbose:
        print(f"{n_total_accepted} total accepted proposals are cut down to the target of {n_samples}")

    assert n_total_accepted > n_samples, "Not enough proposals were accepted to reach the target of n_samples"
    return abc_samples
