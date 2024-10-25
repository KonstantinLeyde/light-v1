import jax.numpy as jnp


def power_spectrum_analytical_form(
    k, A_s=3e1, n_s=-0.96, alpha_s=-0.37, k0=0.05, cut_off=1e-6, debug=False
):
    """
    Generate the power spectrum using the tilted power-law form.

    Parameters:
    - k: array-like, Wavenumbers.
    - A_s: float, Amplitude of the primordial power spectrum.
    - n_s: float, Scalar spectral index.
    - alpha_s: float, Running of the spectral index.
    - k0: float, Pivot scale.

    Returns:
    - P: array-like, Power spectrum values corresponding to input wavenumbers.
    """
    k_eff = k + cut_off

    if debug:
        return 1 / (1e-3 + k * k) ** (1 + 0.1 * jnp.sin(k))

    P = A_s * (k_eff / k0) ** (n_s - 1 + 0.5 * alpha_s * jnp.log(k_eff / k0))

    return P


def smooth_power_law_turnover(x, A_s, alpha_1_s, alpha_2_s, k_turn):
    """
    Generate a power law function with a smooth transition from slope alpha to slope beta.

    """
    # Calculate the smooth power law function with turnover
    result = A_s * (x**alpha_1_s) / (1 + (x / k_turn)) ** (alpha_1_s - alpha_2_s)

    return result
