from abc import ABC


class ProbaRobot(ABC):

    def __init__(self, robot, prior, proposal=None):
        self.robot = robot
        self.prior = prior
        self.proposal = proposal if proposal is not None else prior

    def sample_prior(self, n_sample):
        return # prior samples

    def sample_evidence(self, n_sample):
        return # evidence

    def sample_joint(self, n_sample):
        return # joint

    def sample_likelihood(self, params, n_sample=1, keep_dims=False):
        # if keep_dims=True and n_sample=1 returns shape: (len(params), 1, n_params)
        # if keep_dims=True and params of shape (n_params) returns (1, n_sample, n_params)
        return # forward

    def sample_posterior(self, observations, n_sample=1, keep_dims=False):
        # if keep_dims=True and n_sample=1 returns shape: (len(params), 1, n_params)
        return # posterior samples
