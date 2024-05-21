def sample_parameters_poses(n_random, precision=None, return_indermediates=False, rotation_form=None,
                            uniform_across_radius=False):
    """Samples random parameters from a truncated normal distribution and the end effector poses.

    Parameters
    ----------
    n_random : int
        The number of random positions to sample
    precision : str
        The precision matrix of the truncated normal distribution. Where for a small precision matrix a
        uniform distribution is used for approximation. Default is
    return_indermediates : bool
        If true the poses for the intermediate stages are also returned.
    rotation_form :
        The rotation formalism to use for the forward kinematics,
        where no means just the position is returned. If rotation_form is not None.
        The pose is the position with the rotation formalism appended as (3 + d_rot) vector.
        "quaternion" uses the quaternion representation as a 4D vector.
        "vector" uses the rotation vector representation as a 3D vector.
        "matrix" uses the flatted matrix itself as a 9D vector.
    uniform_across_radius : bool
        If true the distribution of position radii is spread more uniformly via rejection sampling.

    Returns
    -------
    parameters : torch.Tensor of shape (n_samples, n_params)
        the randomly sampled parameters.
    poses : torch.Tensor of shape (n_samples, 3 + d_rot) or (n_joints, n_samples, 3 + d_rot)
        the pose of the end effector from the base frame for each parameter.
        If return_indermediates is true also returns the pose of every joint.
    """

    if rotation_form is None:
        rotation_form = self.rotation_form

    if precision is None:
        precision = self.precision

    eps = 1E-5

    if precision is None:
        precision = torch.zeros(len(self.parameter_indices))

    is_small = precision < eps
    n_is_small = is_small.sum()

    random_parameters = torch.zeros(n_random, len(self.parameter_indices))

    if n_is_small < len(self.parameter_indices):
        distribution = TruncatedNormal(
            loc=torch.zeros(len(self.parameter_indices) - n_is_small),
            scale=1 / precision[~is_small],
            min=self.parameter_ranges[~is_small, 0],
            max=self.parameter_ranges[~is_small, 1],
            tanh_loc=False,
        )
        random_parameters[:, ~is_small] = distribution.sample((n_random,))
    if n_is_small > 0:
        distribution = Uniform(
            low=self.parameter_ranges[is_small, 0],
            high=self.parameter_ranges[is_small, 1]
        )
        random_parameters[:, is_small] = distribution.sample((n_random,))

    poses = self.forward_kinematics(random_parameters, return_indermediates=return_indermediates,
                                    rotation_form=rotation_form)
    return random_parameters, poses