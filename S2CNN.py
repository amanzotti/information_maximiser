import tensorflow as tf
import numpy as np
import healpy as hp
import scipy.special as ss
import sys

class S2CNN:
    def __init__(self, ℓ_max, n_side_in, n_side_out, kernel_size, kernel_index,
        input_filters, output_filters, ω, name, kernel_max = None, μ = 0., σ = 1.,
        input_sphere = None, input_indices = None):

        self._COMPLEXX = tf.complex64
        self._FLOATX = tf.float32
        self.ℓ_max = ℓ_max
        if kernel_max is None:
            self.kernel_max = ℓ_max
        else:
            if kernel_max > ℓ_max:
                print("Kernel bandwidth (kernel_max) must be smaller than ℓ_max but is currently " + str(kernel_max))
                sys.exit()
            if kernel_max <= 0:
                print("Kernel bandwidth (kernel_max) must be bigger than 0 but is currently " + str(kernel_max))
                sys.exit()
            self.kernel_max = kernel_max
        self.θ = np.pi / 2.
        self.n_pix_in = hp.nside2npix(n_side_in)

        if input_sphere is None:
            self.s = tf.placeholder(dtype = self._COMPLEXX, shape = (None,
                self.n_pix_in, input_filters))
        else:
            if input_sphere.get_shape().as_list()[1:] != [self.n_pix_in, input_filters]:
                print("Input sphere must have shape [None, hp.nside2npix(n_side_in), input_filters], but has shape " + str(input_sphere.get_shape().as_list()))
                sys.exit()
            if input_sphere.dtype != self._COMPLEXX:
                print("Input sphere must have type " + str(self._COMPLEXX) + ", but has type " + str(input_sphere.dtype))
                sys.exit()
            self.s = input_sphere
        if input_indices is None:
            self.indices = tf.constant(self.get_weight_indices(n_side_in,
                kernel_size, kernel_index), dtype = tf.int32)
        else:
            if input_indices.get_shape().as_list() != [kernel_size**2, 1]:
                print("Input indices must have shape [kernel_size**2, 1], but has shape " + str(input_indices.get_shape().as_list()))
                sys.exit()
            if input_indices.dtype != tf.int32:
                print("Input indices must have type " + str(tf.int32) + ", but has type " + str(input_indices.dtype))
                sys.exit()
            self.indices = input_indices
        self.k = self.spherical_weight_kernel(name, input_filters,
            output_filters, μ, σ)

        self.Yℓm, self.Yℓm_conj = self.get_Yℓm(n_side_in)

        self.d = self.get_d()

        self.i = tf.cast(tf.complex(0., 1.), dtype = self._COMPLEXX)
        self.m = tf.cast(np.arange(-self.ℓ_max, self.ℓ_max + 1),
            dtype = self._COMPLEXX)
        self.iωmpp = tf.exp(self.i * tf.cast(np.arange(-self.kernel_max, self.kernel_max + 1),
            dtype = self._COMPLEXX) \
            * tf.constant(ω, dtype = self._COMPLEXX))
        self.iϕmp, self.iϕEm = self.get_output_angles(n_side_in, kernel_index,
            n_side_out)

        self.s_hat = self.spherical_transform(self.s, self.Yℓm)
        self.k_hat = self.spherical_transform(self.k, self.Yℓm, kernel = True)
        self.T = self.spherical_convolution(self.s_hat, self.k_hat, self.d,
            self.iωmpp, self.iϕmp, self.iϕEm)

        self.weight_indices = self.get_weight_indices(n_side_in, kernel_size,
            kernel_index)

    def spherical_harmonics(self, ϕE, ϕ):
        Yℓm_arr = np.zeros((self.n_pix_in, self.ℓ_max + 1, 2 * self.ℓ_max + 1))
        for ℓ in range(self.ℓ_max + 1):
            if ℓ == 0:
                Yℓm_arr[:, 0, self.ℓ_max] = ss.sph_harm(0, 0, ϕE, ϕ)
            if ℓ > 0:
                for m in range(ℓ, -ℓ - 1, -1):
                    Yℓm_arr[:, ℓ, m + self.ℓ_max] = ss.sph_harm(m, ℓ, ϕE, ϕ)
        Yℓm = tf.constant(Yℓm_arr, dtype = self._COMPLEXX)
        return Yℓm, tf.conj(Yℓm)

    def get_Yℓm(self, n_side_in):
        ϕ_in, ϕE_in = hp.pix2ang(n_side_in, np.arange(self.n_pix_in))
        return self.spherical_harmonics(ϕ_in, ϕE_in)

    def g(self, ℓ, m):
        if ℓ == 0 and m == 0:
            return 1
        elif ℓ > 0 and m == 0:
            return np.sqrt((2 * ℓ - 1) / (2 * ℓ)) * self.g(ℓ - 1, m)
        elif ℓ > 0 and m > 0 and m <= ℓ:
            return np.sqrt((ℓ - m + 1) / (ℓ + m)) * self.g(ℓ, m - 1)
        else:
            print("g -> ℓ = " + str(ℓ) + ", m = " + str(m))
            sys.exit()

    def d_ℓmℓ(self, ℓ, m):
        return (-1.)**(ℓ + m) * self.g(ℓ, m) * (1. + np.cos(self.θ))**m \
            * np.sin(self.θ)**(ℓ - m)

    def d_ℓmmpm1(self, ℓ, m, mp, d_arr):
        return np.sqrt((ℓ * (ℓ + 1.) - m * (m + 1)) / (ℓ * (ℓ + 1.) - mp \
            * (mp - 1))) * d_arr[ℓ, m + 1 + self.ℓ_max, mp + self.ℓ_max] \
            + (m + mp) / np.sqrt(ℓ * (ℓ + 1.) - mp * (mp - 1.)) \
            * np.sin(self.θ) / (1. + np.cos(self.θ)) \
            * d_arr[ℓ, m + self.ℓ_max, mp + self.ℓ_max]

    def d_ℓℓmpm1(self, ℓ, mp, d_arr):
        return (ℓ + mp) / np.sqrt(ℓ * (ℓ + 1.) - mp * (mp - 1.)) \
            * np.sin(self.θ) / (1. + np.cos(self.θ)) \
            * d_arr[ℓ, ℓ + self.ℓ_max, mp + self.ℓ_max]

    def d_ℓmmmmp(self, ℓ, m, mp, d_arr):
        return (-1.)**(m + mp) * d_arr[ℓ, - m + self.ℓ_max, - mp + self.ℓ_max]

    def get_d(self):
        d_arr = np.zeros((self.ℓ_max + 1, 2 * self.ℓ_max + 1, 2 * self.ℓ_max \
            + 1))
        for ℓ in range(self.ℓ_max, -1, -1):
            for m in range(ℓ, -ℓ - 1, -1):
                for mp in range(ℓ, -ℓ - 1, -1):
                    if m >= 0:
                        if mp == ℓ :
                            d_arr[ℓ, m + self.ℓ_max, mp + self.ℓ_max] = \
                                self.d_ℓmℓ(ℓ, m)
                        if m == ℓ and mp > -ℓ:
                            d_arr[ℓ, m + self.ℓ_max, mp - 1 + self.ℓ_max] = \
                                self.d_ℓℓmpm1(ℓ, mp, d_arr)
                        if m < ℓ and mp > -ℓ:
                            d_arr[ℓ, m + self.ℓ_max, mp - 1 + self.ℓ_max] = \
                                self.d_ℓmmpm1(ℓ, m, mp, d_arr)
                    if m < 0:
                        d_arr[ℓ, m + self.ℓ_max, mp + self.ℓ_max] = \
                            self.d_ℓmmmmp(ℓ, m, mp, d_arr)
        return tf.constant(d_arr, dtype = self._COMPLEXX)

    def spherical_weight_kernel(self, name, input_filters, output_filters, μ,
        σ):
        with tf.variable_scope(name):
            w = tf.get_variable(name,
                shape = [self.indices.get_shape().as_list()[0],
                input_filters, output_filters],
                initializer = tf.random_normal_initializer(μ, σ))
        k = tf.scatter_nd(self.indices, w, [self.n_pix_in, input_filters,
            output_filters])
        return tf.cast(k, dtype = self._COMPLEXX)

    def get_output_angles(self, n_side_in, kernel_index, n_side_out):
        ϕ_kernel, ϕE_kernel = hp.pix2ang(n_side_in, kernel_index)
        ϕ_kernel = tf.constant(ϕ_kernel, dtype = self._COMPLEXX)
        ϕE_kernel = tf.constant(ϕE_kernel, dtype = self._COMPLEXX)
        ϕ_out, ϕE_out = hp.pix2ang(n_side_out,
            np.arange(hp.nside2npix(n_side_out)))
        ϕ_out = tf.constant(ϕ_out, dtype = self._COMPLEXX) + ϕ_kernel
        ϕE_out = tf.constant(ϕE_out, dtype = self._COMPLEXX) + ϕE_kernel
        return tf.exp(self.i * self.m * ϕ_out[:, None]), tf.exp(self.i \
            * self.m * ϕE_out[:, None])

    def spherical_transform(self, γ, Yℓm, kernel = False):
        if kernel:
            γ = tf.transpose(γ, [1, 2, 0])
            Yℓm = Yℓm[:, :, self.ℓ_max - self.kernel_max: self.ℓ_max + self.kernel_max + 1]
        else:
            γ = tf.transpose(γ, [0, 2, 1])
        transform = tf.reduce_sum(4. * np.pi / self.n_pix_in \
            * γ[:, :, :, None, None] * Yℓm, axis = 2)
        if kernel:
            return tf.transpose(transform, [2, 3, 0, 1])
        else:
            return tf.transpose(transform, [0, 2, 3, 1])

    def spherical_convolution(self, s_, k_, d, iωmpp, iϕmp, iϕEm):
        k_ωmpp = tf.einsum('ijkl,j->ijkl', k_, iωmpp)
        dk_ωmp = tf.einsum('ijk,iklm->ijlm', d[:, :, self.ℓ_max - self.kernel_max: self.ℓ_max + self.kernel_max + 1], k_ωmpp)
        ddk_ωmmp = tf.einsum('ijk,iklm->ijklm', d, dk_ωmp)
        ϕddk_ωm = tf.einsum('ij,kljmn->iklmn', iϕmp, ddk_ωmmp)
        ϕEϕddk_ωm = tf.einsum('ij,ikjlm->ikjlm', iϕEm, ϕddk_ωm)
        T = tf.einsum('ijkl,mjkln->imn',s_, ϕEϕddk_ωm)
        return tf.cast(T, dtype = self._FLOATX)
        #Tmmpmpp = tf.einsum('ijklmn,jmno->iklmo', tf.einsum('ijklm,jln->ijklnm',
        #    tf.einsum('ijkl,jkm->ijkml', s_, d), d), k_)
        #Tmmp = tf.einsum('ijklm,l->ijkm', Tmmpmpp, iωmpp)
        #Tm = tf.einsum('ijkl,mk->imjl', Tmmp, iϕmp)
        #return tf.cast(tf.einsum('ijkl,jk->ijl', Tm, iϕEm),
        #    dtype = self._FLOATX)

    def get_weight_indices(self, n_side_in, kernel_size, kernel_index):
        weight_indices = np.array([kernel_index])
        if (kernel_size % 2 == 0):
            print('Pixels per side of the weight kernel needs to be odd')
            sys.exit()
        else:
            if kernel_size > 1:
                initial_index = 0
                for pixel_count in range(3, kernel_size + 1, 2):
                    final_index = weight_indices.shape[0]
                    for pixel in range(initial_index, final_index):
                        outer_indices = hp.get_all_neighbours(n_side_in,
                            weight_indices[pixel])
                        weight_indices = np.append(weight_indices,
                            outer_indices[np.where(np.in1d(outer_indices,
                            weight_indices, invert = True))[0]])
                    initial_index = final_index
        return np.reshape(weight_indices, (kernel_size**2, 1))

    def get_aℓm_Cℓ(self, γ, kernel_max = None):
        aℓm = np.zeros((self.ℓ_max, self.ℓ_max, 2))
        Cℓ = np.sum(γ * np.conj(γ), axis = 1) / (2 * np.arange(self.ℓ_max + 1) \
            + 1)
        if kernel_max is None:
            kernel_max = self.ℓ_max
        for ℓ in range(self.ℓ_max):
            if kernel_max is None:
                up_to = ℓ
            if ℓ > 0:
                aℓm[ℓ, max(0, ℓ - kernel_max):ℓ, 0] = np.real(γ[ℓ, max(-ℓ, -kernel_max) + kernel_max: kernel_max])
                aℓm[max(0, ℓ - kernel_max):ℓ, ℓ, 0] = np.real(γ[ℓ, kernel_max + 1: kernel_max + min(ℓ, kernel_max) + 1])
                aℓm[ℓ, max(0, ℓ - kernel_max):ℓ, 1] = np.imag(γ[ℓ, max(-ℓ, -kernel_max) + kernel_max: kernel_max])
                aℓm[max(0, ℓ - kernel_max):ℓ, ℓ, 1] = np.imag(γ[ℓ, kernel_max + 1: kernel_max + min(ℓ, kernel_max) + 1])
            aℓm[ℓ, ℓ, 0] = np.real(γ[ℓ, kernel_max])
            aℓm[ℓ, ℓ, 1] = np.imag(γ[ℓ, kernel_max])
        return aℓm, Cℓ
