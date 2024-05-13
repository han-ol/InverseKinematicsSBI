
import numpy as np

def mask_sample_dict(sample_dict, mask):
    return {name: samples[mask] for name, samples in sample_dict.items()}

def concat_sample_dicts(sample_dict_list):
    names = sample_dict_list[0].keys()
    n_dicts = len(sample_dict_list)
    return {name: np.concatenate([sample_dict_list[i][name] for i in range(n_dicts)]) for name in names}

def sample_dist_dict(dist_dict, n_samples):
    return {name: distribution.rvs(n_samples) for name, distribution in dist_dict.items()}


def sample_matrix_from_distdict(distdict, n_samples=1000):
    "Returns a matrix of samples from the distributions in distdict."
    return np.array([th.rvs(n_samples) for th in distdict.values()]).T

def _slice_distdict(distdict, keys):
    return OrderedDict((key, value) for key, value in self.prior_reference.items() if key in keys)

def _pdf_from_distdict(distdict):
    "Returns a function that computes the pdfs for rows of a matrix in parallel."
    return lambda x: np.prod([distdict[k].pdf(x[:, i]) for i, k in enumerate(distdict.keys())], axis=0)



