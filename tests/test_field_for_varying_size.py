import jax
import jax.numpy as jnp
import numpy as np

from light.field import field


def test_field_with_varying_size():
    real_fields = {}

    # first instantiate the fields to compute kmin kmax
    power_func_analytic = lambda x: np.ones_like(x)
    real_fields["large"] = field.RealField(
        box_size_d=[3, 3],
        box_shape_d=(2000, 2000),
        power_spectrum_of_k=power_func_analytic,
    )

    real_fields["small"] = field.RealField(
        box_size_d=[1, 1],
        box_shape_d=(500, 500),
        power_spectrum_of_k=power_func_analytic,
    )

    kmin = max(
        [
            max(abs(real_fields["large"].k_components[0][1:])),
            max(abs(real_fields["small"].k_components[0][1:])),
        ]
    )
    kmax = min(
        [
            max(real_fields["small"].k_components[0]),
            max(real_fields["large"].k_components[0]),
        ]
    )

    power_func_analytic = (
        lambda x: (x + 1e-7) ** (-2.7)
        * (1 + 2 * jnp.sin(x) ** 2)
        * np.heaviside(x - kmin, 0)
        * np.heaviside(kmax - x, 0)
    )
    real_fields["large"] = field.RealField(
        box_size_d=real_fields["large"].box_size_d,
        box_shape_d=real_fields["large"].box_shape_d,
        power_spectrum_of_k=power_func_analytic,
    )

    real_fields["small"] = field.RealField(
        box_size_d=real_fields["small"].box_size_d,
        box_shape_d=real_fields["small"].box_shape_d,
        power_spectrum_of_k=power_func_analytic,
    )

    for key in real_fields.keys():
        real_fields[key].set_batch_shape((10,))
        real_fields[key].sample_gaussian_F_whitened_fourier(jax.random.PRNGKey(219))
        real_fields[key].compute_gaussian_F_fourier(
            power_spectrum_kwargs={},
        )
        real_fields[key].compute_gaussian_F_spatial()

    # fig, ax = plt.subplots(1,2, figsize=(10,10))

    # im1 = ax[0].imshow(real_fields['small'].gaussian_F_spatial)
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes('right', size='5%', pad=0.25)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    # ax[0].set_title('small box')

    # im2 = ax[1].imshow(real_fields['large'].gaussian_F_spatial)
    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes('right', size='5%', pad=0.25)
    # fig.colorbar(im2, cax=cax, orientation='vertical')
    # ax[1].set_title('large box');

    # for key in real_fields.keys():
    #     plt.hist(np.array(real_fields[key].gaussian_F_spatial.flatten()), bins='auto', histtype='step', density=True, label=key)
    # plt.legend()

    means, stds = {}, {}
    for key in real_fields.keys():
        means[key] = jnp.mean(real_fields[key].gaussian_F_spatial)
        stds[key] = jnp.std(real_fields[key].gaussian_F_spatial)

    print(stds["large"], stds["small"])
    assert jnp.allclose(means["large"], means["small"], rtol=1e-5), "Means are close"
    assert jnp.allclose(stds["large"], stds["small"], rtol=1e-2), "Stds are close"
