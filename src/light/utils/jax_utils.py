from functools import partial

import jax.numpy as jnp
from jax import vmap


def compute_centers_and_delta_from_array(x):
    deltas = x[1:] - x[:-1]
    centers = x[:-1] + deltas / 2
    return centers, deltas


@partial(vmap, in_axes=(None, None, 0))
@partial(vmap, in_axes=(None, None, 0))
def interp1d_with_first_two_dims_as_batch(new_coordinates, x_coordinates, x):
    return jnp.interp(new_coordinates, x_coordinates, x)


class Integrator:
    """
    This class allows to interpolate from a density that
    lives on a 3d grid.

    The original coordinates are assumed to be along the third
    axis and should describe the boundaries of the pixels.

    E.g. if x has shape (10,10,10), the coordinates should have
    shape (11,).


    """

    def __init__(self, x, original_coord_boundaries):
        self.x = x
        self.original_coord_boundaries = original_coord_boundaries

        self.check_conventions()

        self.compute_original_coord_centers()

    def check_conventions(self):
        if len(self.x.shape) != 3:
            raise NotImplementedError

        if len(self.original_coord_boundaries.shape) != 1:
            raise NotImplementedError

        length_coord = self.original_coord_boundaries.shape[0]
        if length_coord != (self.x.shape[2] + 1):
            raise NotImplementedError

    def compute_original_coord_centers(self):
        self.original_coord_centers, self.original_coord_deltas = (
            compute_centers_and_delta_from_array(self.original_coord_boundaries)
        )

    def get_x_from_new_coord_centers_interpolated(self, new_coord_centers):
        return interp1d_with_first_two_dims_as_batch(
            new_coord_centers, self.original_coord_centers, self.x
        )

    def get_x_from_integral_approximation(self, new_coord_boundaries):
        coords_difference = new_coord_boundaries - self.original_coord_boundaries

        # if(coords_difference[0] != 0 or coords_difference[-1] != 0):
        #     raise 'The overall coordinate boundaries have to coincide. '

        if new_coord_boundaries.shape != self.original_coord_boundaries.shape:
            raise "This integral method requires equal shape of both (original\
                and new coordinates. )"

        coords_diff_low = coords_difference[:-1]
        coords_diff_high = coords_difference[1:]

        contribution_low = -coords_diff_low * jnp.where(
            coords_diff_low < 0, jnp.roll(self.x, shift=1, axis=2), self.x
        )
        contribution_high = coords_diff_high * jnp.where(
            coords_diff_high > 0, jnp.roll(self.x, shift=-1, axis=2), self.x
        )

        return (
            self.x * self.original_coord_deltas + contribution_low + contribution_high
        )
