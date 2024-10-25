import jax
import jax.scipy as jsp


@jax.jit
def batch_convolve(x, kernel):
    return jsp.signal.convolve(x, kernel, mode="same")
