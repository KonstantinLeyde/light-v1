import numpyro
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_uniform


class AutoGuideList(AutoGuide):
    """
    Container class to combine multiple automatic guides.

    Example usage::

        rng_key_init = random.PRNGKey(0)
        guide = AutoGuideList(my_model)
        guide.append(
            AutoNormal(
                numpyro.handlers.block(numpyro.handlers.seed(model, rng_seed=0), hide=["coefs"])
            )
        )
        guide.append(
            AutoDelta(
                numpyro.handlers.block(numpyro.handlers.seed(model, rng_seed=1), expose=["coefs"])
            )
        )
        svi = SVI(model, guide, optim, Trace_ELBO())
        svi_state = svi.init(rng_key_init, data, labels)
        params = svi.get_params(svi_state)

    :param callable model: a NumPyro model
    """

    def __init__(
        self, model, *, prefix="auto", init_loc_fn=init_to_uniform, create_plates=None
    ):
        self._guides = []
        super().__init__(
            model, prefix=prefix, init_loc_fn=init_loc_fn, create_plates=create_plates
        )

    def append(self, part):
        """
        Add an automatic or custom guide for part of the model. The guide should
        have been created by blocking the model to restrict to a subset of
        sample sites. No two parts should operate on any one sample site.

        :param part: a partial guide to add
        :type part: AutoGuide
        """
        if (
            isinstance(part, numpyro.infer.autoguide.AutoDAIS)
            or isinstance(part, numpyro.infer.autoguide.AutoSemiDAIS)
            or isinstance(part, numpyro.infer.autoguide.AutoSurrogateLikelihoodDAIS)
        ):
            raise ValueError(
                "AutoDAIS, AutoSemiDAIS, and AutoSurrogateLikelihoodDAIS are not supported."
            )
        self._guides.append(part)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        # create all plates
        self._create_plates(*args, **kwargs)

        # run slave guides
        result = {}
        for part in self._guides:
            result.update(part(*args, **kwargs))
        return result

    def __getitem__(self, key):
        return self._guides[key]

    def __len__(self):
        return len(self._guides)

    def __iter__(self):
        yield from self._guides

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), **kwargs):
        result = {}
        for part in self._guides:
            # TODO: remove this when sample_posterior() signatures are consistent
            # we know part is not AutoDAIS, AutoSemiDAIS, or AutoSurrogateLikelihoodDAIS
            if isinstance(part, numpyro.infer.autoguide.AutoDelta):
                result.update(
                    part.sample_posterior(
                        rng_key, params, *args, sample_shape=sample_shape, **kwargs
                    )
                )
            else:
                result.update(
                    part.sample_posterior(rng_key, params, sample_shape=sample_shape)
                )
        return result

    def median(self, params):
        result = {}
        for part in self._guides:
            result.update(part.median(params))
        return result

    def quantiles(self, params, quantiles):
        result = {}
        for part in self._guides:
            result.update(part.quantiles(params, quantiles))
        return result
