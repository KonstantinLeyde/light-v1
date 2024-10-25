import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam


def get_Uniform_samples(name, prior_dict_info):
    dist_min = prior_dict_info["min"]
    dist_max = prior_dict_info["max"]

    samples = numpyro.sample(
        name,
        dist.TransformedDistribution(
            dist.Uniform(
                0,
                1,
            ),
            dist.transforms.AffineTransform(dist_min, dist_max - dist_min),
        ),
    )

    return samples


def get_Dirichlet_samples(name, prior_dict_info):
    concentration = prior_dict_info["concentration"]
    samples = numpyro.sample(name, dist.Dirichlet(concentration))

    return samples


def get_LogUniform_samples(name, prior_dict_info):
    dist_min = prior_dict_info["min"]
    dist_max = prior_dict_info["max"]

    samples = numpyro.sample(
        name,
        dist.LogUniform(
            dist_min,
            dist_max,
        ),
    )
    return samples


def get_Delta_samples(name, prior_dict_info):
    return prior_dict_info["value"]


def get_prior_dict_samples(param, prior_dict_info):
    if "dist_type" not in prior_dict_info.keys():
        dist_type = "Uniform"
    else:
        dist_type = prior_dict_info["dist_type"]

    if dist_type == "Uniform":
        with numpyro.handlers.reparam(config={param: TransformReparam()}):
            samples = get_Uniform_samples(param, prior_dict_info)
    elif dist_type == "Dirichlet":
        samples = get_Dirichlet_samples(param, prior_dict_info)
    elif dist_type == "LogUniform":
        with numpyro.handlers.reparam(config={param: TransformReparam()}):
            samples = get_LogUniform_samples(param, prior_dict_info)
    elif dist_type == "Delta":
        samples = get_Delta_samples(param, prior_dict_info)
    else:
        raise NotImplementedError

    return samples
