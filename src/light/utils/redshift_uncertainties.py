import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def add_shifted(counts_not_jump, counts_jump_left, counts_jump_right):
    """
    Add counts with shifts along the third axis, properly handling boundaries.

    Args:
        counts_not_jump: Tensor of counts with no shift, shape (n, n, n)
        counts_jump_left: Tensor of counts shifting left, shape (n, n, n)
        counts_jump_right: Tensor of counts shifting right, shape (n, n, n)

    Returns:
        Updated tensor where shifts and boundary conditions are properly handled.
    """
    # Shape of the tensors
    n = counts_not_jump.shape[-1]

    # Shift the counts_jump_left by slicing up to the second-to-last element along the third axis
    shifted_left = jnp.zeros_like(counts_not_jump)
    shifted_left = shifted_left.at[:, :, :-1].set(counts_jump_left[:, :, 1:])

    # Shift the counts_jump_right by slicing starting from the second element along the third axis
    shifted_right = jnp.zeros_like(counts_not_jump)
    shifted_right = shifted_right.at[:, :, 1:].set(
        counts_jump_right[:, :, :-1]
    )  # Shift right by one

    # Add the boundary conditions
    shifted_left = shifted_left.at[:, :, 0].add(
        counts_jump_left[:, :, 0]
    )  # Add the left boundary
    shifted_right = shifted_right.at[:, :, -1].add(
        counts_jump_right[:, :, -1]
    )  # Add the right boundary

    # Sum the counts (not shifted) and the shifted counts
    x = counts_not_jump + shifted_left + shifted_right

    return x


def get_probs_jump_outside(z_low, z_high, sigma_z):
    # Compute the exponent part
    exponent_part = -((z_high - z_low) ** 2) / (2 * sigma_z**2)

    # Compute the first term
    first_term = (
        (2 - 2 * jnp.exp(exponent_part))
        * sigma_z
        / (2 * jnp.sqrt(2 * jnp.pi) * (z_high - z_low))
    )

    # Compute the second term
    second_term = 0.5 * jax.scipy.special.erfc(
        (z_high - z_low) / (jnp.sqrt(2) * sigma_z)
    )

    # Combine both terms
    result = first_term + second_term

    # return twice result, since above probability is for one side only
    return 2 * result


def get_new_binomial_random_variable(
    counts, probs, name_new_binomial_random_variable, upper_limit
):
    n = counts.shape[-1]

    binomial = dist.Binomial(
        total_count=jnp.expand_dims(counts, axis=-1),
        probs=jnp.expand_dims(probs, axis=-1),
    )

    values = jnp.arange(upper_limit + 1)
    values = jnp.broadcast_to(
        jnp.expand_dims(values, axis=(-4, -3, -2)), (n, n, n, values.shape[0])
    )
    upper_bounds = jnp.expand_dims(counts, axis=-1)

    cat_logits = binomial.log_prob(jnp.minimum(values, upper_bounds))
    cat_logits = cat_logits.at[values > upper_bounds].set(-jnp.inf)

    jax.debug.print("{}", cat_logits.shape)
    jax.debug.print("{}", values.shape)

    counts_galaxies_true = numpyro.sample(
        name_new_binomial_random_variable,
        dist.Categorical(logits=cat_logits),
        infer={"enumerate": "parallel"},
    ).astype(jnp.int64)

    return counts_galaxies_true
