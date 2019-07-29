'''
This module defines the interfaces to RTE implementations
'''
import os
import importlib
import abc
import numpy as np
from cffi import FFI
from .rte_api import RTEAPIDefinition


class RTEInterface(object):
    '''
    Base class for implementations of the RTE interface

    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 precision='double',
                 direction='bottom_to_top'):
        '''
        Create an interface to an RTE implementation

        Args:
            precision (string, optional):
                whether the library has been compiled with single or double
                precision. Should be one of :code:`single` or :code:`double`.
                Defaults to :code:`double`.

        direction (string, optional):
            indicates the orientation of vertical dimension.

            * 'bottom_to_top': index 0 of the :code:`num_layers` dimension
               is the bottom of the model.
            * 'top_to_bottom': index -1 of the :code:`num_layers` dimension
               is the bottom of the model.

            Defaults to 'bottom_to_top'.
        '''
        self.precision = precision
        self.direction = direction

    @abc.abstractmethod
    def apply_zero_bc(self, downward_flux):
        '''
        Apply a boundary condition of zero incident flux.

        Args:
            downward_flux (np.ndarray):
                The downward flux array. Must have dimensions
                :code:`[num_g_points, num_layers, num_columns]`.
        '''

    @abc.abstractmethod
    def apply_gpoint_bc(self,
                        downward_flux, incident_flux):
        '''
        Apply the provided boundary condition for incident flux.

        Args:
            downward_flux (np.ndarray):
                The downward flux array. Must have dimensions
                :code:`[num_g_points, num_layers, num_columns]`.

            incident_flux (np.ndarray, optional):
                The boundary condition to be applied at the top of the model.
                Must have dimensions :code:`[num_g_points, num_columns]`.
        '''

    @abc.abstractmethod
    def apply_scaled_gpoint_bc(self,
                               downward_flux, incident_flux,
                               scale_factor):
        '''
        Apply a scaled version of the provided boundary condition for incident flux.

        Args:
            downward_flux (np.ndarray):
                The downward flux array. Must have dimensions
                :code:`[num_g_points, num_layers, num_columns]`.

            incident_flux (np.ndarray, optional):
                The boundary condition to be applied at the top of the model.
                Must have dimensions :code:`[num_g_points, num_columns]`.

            scale_factor (np.ndarray, optional):
                The scale factor to be applied to :code:`incident_flux`. Must
                have dimensions :code:`[num_columns]`.

        '''

    @abc.abstractmethod
    def longwave_no_scattering(self,
                               secant_propagation_angle,
                               planck_src_mid,
                               planck_src_iface_decreasing,
                               planck_src_iface_increasing,
                               surface_emissivity,
                               surface_source,
                               downward_flux,
                               quadrature_weight,
                               optical_tau):

        '''
        has to be implemented by an inheriting class
        '''

    @abc.abstractmethod
    def gaussian_longwave_no_scattering(self,
                                        secant_propagation_angle,
                                        planck_src_mid,
                                        planck_src_iface_decreasing,
                                        planck_src_iface_increasing,
                                        surface_emissivity,
                                        surface_source,
                                        downward_flux,
                                        quadrature_weight,
                                        num_angles,
                                        optical_tau):
        '''
        has to be implemented by an inheriting class
        '''

    @abc.abstractmethod
    def two_stream_longwave(self,
                            planck_src_mid,
                            planck_src_iface_decreasing,
                            planck_src_iface_increasing,
                            surface_emissivity,
                            surface_source,
                            downward_flux,
                            single_scattering_albedo,
                            asymmetry_parameter,
                            optical_tau):
        '''
        has to be implemented by an inheriting class
        '''

    @abc.abstractmethod
    def shortwave_no_scattering(self,
                                cos_zenith_angle,
                                direct_beam_flux,
                                optical_tau):
        '''
        has to be implemented by an inheriting class
        '''

    @abc.abstractmethod
    def two_stream_shortwave(self,
                             cos_zenith_angle,
                             direct_beam_flux,
                             single_scattering_albedo,
                             asymmetry_parameter,
                             surface_albedo_direct,
                             surface_albedo_diffuse,
                             downward_flux,
                             optical_tau):
        '''
        has to be implemented by an inheriting class
        '''


class CompiledRTEInterface(RTEInterface):
    '''
    Wrap a compiled library using FFI
    '''

    def __init__(self, rte_extension=None, **kwargs):
        '''
        Create an object that wraps a compiled RTE implementation
        '''

        super().__init__(**kwargs)

        self._rte_extension = rte_extension
        self._ffi = FFI()
        self._api = RTEAPIDefinition(self.precision)

        if self.precision == 'single':
            self._numpy_type = np.float32
            self._ffi_type = 'float'
        elif self.precision == 'double':
            self._numpy_type = np.float64
            self._ffi_type = 'double'
        else:
            raise ValueError(
                'Uknown precision type {}'.format(self.precision))

    def apply_zero_bc(self, downward_flux):
        '''
        Implementation of the abstract method.
        '''

        validate_bc_inputs(downward_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)

        self._rte_extension.lib.apply_BC_0(
            num_cols_ffi, num_layers_ffi,
            num_gpts_ffi, top_at_1_ffi,
            downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)

        return down_flux_out.reshape(downward_flux.shape)

    def apply_gpoint_bc(self,
                        downward_flux, incident_flux):
        '''
        Implementation of abstract method
        '''
        validate_bc_inputs(downward_flux,
                           incident_flux=incident_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        inc_flux_ffi = self.convert_to_from_cffi(incident_flux)

        self._rte_extension.lib.apply_BC_gpt(
            num_cols_ffi, num_layers_ffi,
            num_gpts_ffi, top_at_1_ffi,
            inc_flux_ffi,
            downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)

        return down_flux_out.reshape(downward_flux.shape)

    def apply_scaled_gpoint_bc(self,
                               downward_flux, incident_flux,
                               scale_factor):
        '''
        Implementation of abstract method.
        '''
        validate_bc_inputs(downward_flux,
                           incident_flux=incident_flux,
                           scale_factor=scale_factor)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        inc_flux_ffi = self.convert_to_from_cffi(incident_flux)
        scale_factor_ffi = self.convert_to_from_cffi(scale_factor)

        self._rte_extension.lib.apply_BC_factor(
            num_cols_ffi, num_layers_ffi,
            num_gpts_ffi, top_at_1_ffi,
            inc_flux_ffi,
            scale_factor_ffi,
            downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)

        return down_flux_out.reshape(downward_flux.shape)

    def longwave_no_scattering(self,
                               secant_propagation_angle,
                               planck_src_mid,
                               planck_src_iface_decreasing,
                               planck_src_iface_increasing,
                               surface_emissivity,
                               surface_source,
                               downward_flux,
                               quadrature_weight,
                               optical_tau):
        '''
        Implementation of abstract method
        '''
        validate_lw_solver_inputs(
            planck_src_mid,
            planck_src_iface_decreasing,
            planck_src_iface_increasing,
            surface_emissivity,
            surface_source,
            downward_flux,
            optical_tau,
            secant_propagation_angle,
            quadrature_weight)

        upward_flux = np.empty_like(downward_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        upward_flux_ffi = self.convert_to_from_cffi(upward_flux)
        sec_angle_ffi = self.convert_to_from_cffi(secant_propagation_angle)
        planck_mid_ffi = self.convert_to_from_cffi(planck_src_mid)
        planck_dec_ffi = self.convert_to_from_cffi(planck_src_iface_decreasing)
        planck_inc_ffi = self.convert_to_from_cffi(planck_src_iface_increasing)
        sfc_emiss_ffi = self.convert_to_from_cffi(surface_emissivity)
        sfc_src_ffi = self.convert_to_from_cffi(surface_source)
        tau_ffi = self.convert_to_from_cffi(optical_tau)
        weight_ffi = self._ffi.cast('{} *'.format(self._ffi_type), quadrature_weight)

        self._rte_extension.lib.lw_solver_noscat(
            num_cols_ffi, num_layers_ffi, num_gpts_ffi, top_at_1_ffi,
            sec_angle_ffi, weight_ffi, tau_ffi, planck_mid_ffi,
            planck_inc_ffi, planck_dec_ffi, sfc_emiss_ffi,
            sfc_src_ffi, upward_flux_ffi, downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)
        up_flux_out = self.convert_to_from_cffi(
            upward_flux_ffi, num_bytes=downward_flux.nbytes)

        return up_flux_out.reshape(downward_flux.shape), down_flux_out.reshape(downward_flux.shape)

    def gaussian_longwave_no_scattering(self,
                                        secant_propagation_angle,
                                        planck_src_mid,
                                        planck_src_iface_decreasing,
                                        planck_src_iface_increasing,
                                        surface_emissivity,
                                        surface_source,
                                        downward_flux,
                                        quadrature_weight,
                                        num_angles,
                                        optical_tau):
        '''
        Implementation of abstract method
        '''
        validate_lw_solver_inputs(
            planck_src_mid,
            planck_src_iface_decreasing,
            planck_src_iface_increasing,
            surface_emissivity,
            surface_source,
            downward_flux,
            optical_tau,
            secant_propagation_angle,
            quadrature_weight,
            num_angles)

        upward_flux = np.empty_like(downward_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        upward_flux_ffi = self.convert_to_from_cffi(upward_flux)
        sec_angle_ffi = self.convert_to_from_cffi(secant_propagation_angle)
        planck_mid_ffi = self.convert_to_from_cffi(planck_src_mid)
        planck_dec_ffi = self.convert_to_from_cffi(planck_src_iface_decreasing)
        planck_inc_ffi = self.convert_to_from_cffi(planck_src_iface_increasing)
        sfc_emiss_ffi = self.convert_to_from_cffi(surface_emissivity)
        sfc_src_ffi = self.convert_to_from_cffi(surface_source)
        tau_ffi = self.convert_to_from_cffi(optical_tau)
        weight_ffi = self.convert_to_from_cffi(quadrature_weight)
        num_angles_ffi = self._ffi.cast('int *', num_angles)

        self._rte_extension.lib.lw_solver_noscat_GaussQuad(
            num_cols_ffi, num_layers_ffi, num_gpts_ffi, top_at_1_ffi,
            num_angles_ffi, sec_angle_ffi, weight_ffi, tau_ffi, planck_mid_ffi,
            planck_inc_ffi, planck_dec_ffi, sfc_emiss_ffi,
            sfc_src_ffi, upward_flux_ffi, downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)
        up_flux_out = self.convert_to_from_cffi(
            upward_flux_ffi, num_bytes=downward_flux.nbytes)

        return up_flux_out.reshape(downward_flux.shape), down_flux_out.reshape(downward_flux.shape)

    def two_stream_longwave(self,
                            planck_src_mid,
                            planck_src_iface_decreasing,
                            planck_src_iface_increasing,
                            surface_emissivity,
                            surface_source,
                            downward_flux,
                            single_scattering_albedo,
                            asymmetry_parameter,
                            optical_tau):
        '''
        Implementation of abstract interface
        '''
        validate_lw_solver_inputs(
            planck_src_mid,
            planck_src_iface_decreasing,
            planck_src_iface_increasing,
            surface_emissivity,
            surface_source,
            downward_flux,
            optical_tau,
            secant_propagation_angle=None,
            quadrature_weight=None,
            num_angles=None,
            ssa=single_scattering_albedo,
            asym_param=asymmetry_parameter,
        )

        upward_flux = np.empty_like(downward_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             downward_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        upward_flux_ffi = self.convert_to_from_cffi(upward_flux)
        planck_mid_ffi = self.convert_to_from_cffi(planck_src_mid)
        planck_dec_ffi = self.convert_to_from_cffi(planck_src_iface_decreasing)
        planck_inc_ffi = self.convert_to_from_cffi(planck_src_iface_increasing)
        sfc_emiss_ffi = self.convert_to_from_cffi(surface_emissivity)
        sfc_src_ffi = self.convert_to_from_cffi(surface_source)
        tau_ffi = self.convert_to_from_cffi(optical_tau)
        ssa_ffi = self.convert_to_from_cffi(single_scattering_albedo)
        asym_ffi = self.convert_to_from_cffi(asymmetry_parameter)

        self._rte_extension.lib.lw_solver_2stream(
            num_cols_ffi, num_layers_ffi, num_gpts_ffi, top_at_1_ffi,
            tau_ffi, ssa_ffi, asym_ffi, planck_mid_ffi,
            planck_inc_ffi, planck_dec_ffi, sfc_emiss_ffi,
            sfc_src_ffi, upward_flux_ffi, downward_flux_ffi)

        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)
        up_flux_out = self.convert_to_from_cffi(
            upward_flux_ffi, num_bytes=downward_flux.nbytes)

        return up_flux_out.reshape(downward_flux.shape), down_flux_out.reshape(downward_flux.shape)

    def shortwave_no_scattering(self,
                                cos_zenith_angle,
                                direct_beam_flux,
                                optical_tau):
        '''
        Implementation of abstract method
        '''
        validate_sw_solver_inputs(
            cos_zenith_angle,
            direct_beam_flux,
            optical_tau
        )

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             direct_beam_flux)

        direct_flux_ffi = self.convert_to_from_cffi(direct_beam_flux)
        zen_angle_ffi = self.convert_to_from_cffi(cos_zenith_angle)
        tau_ffi = self.convert_to_from_cffi(optical_tau)

        self._rte_extension.lib.sw_solver_noscat(
            num_cols_ffi, num_layers_ffi, num_gpts_ffi, top_at_1_ffi,
            tau_ffi, zen_angle_ffi, direct_flux_ffi)

        direct_flux_out = self.convert_to_from_cffi(
            direct_flux_ffi, num_bytes=direct_beam_flux.nbytes)

        return direct_flux_out.reshape(direct_beam_flux.shape)

    def two_stream_shortwave(self,
                             cos_zenith_angle,
                             direct_beam_flux,
                             single_scattering_albedo,
                             asymmetry_parameter,
                             surface_albedo_direct,
                             surface_albedo_diffuse,
                             downward_flux,
                             optical_tau):
        '''
        Implementation of abstract method
        '''

        validate_sw_solver_inputs(
            cos_zenith_angle,
            direct_beam_flux,
            optical_tau,
            single_scattering_albedo,
            asymmetry_parameter,
            surface_albedo_direct,
            surface_albedo_diffuse,
            downward_flux,
        )

        upward_flux = np.empty_like(downward_flux)

        (num_gpts_ffi, num_cols_ffi,
         num_layers_ffi, top_at_1_ffi) = self.get_cffi_dims_and_dir(
             direct_beam_flux)

        downward_flux_ffi = self.convert_to_from_cffi(downward_flux)
        upward_flux_ffi = self.convert_to_from_cffi(upward_flux)
        direct_flux_ffi = self.convert_to_from_cffi(direct_beam_flux)
        zen_angle_ffi = self.convert_to_from_cffi(cos_zenith_angle)
        tau_ffi = self.convert_to_from_cffi(optical_tau)
        ssa_ffi = self.convert_to_from_cffi(single_scattering_albedo)
        asym_ffi = self.convert_to_from_cffi(asymmetry_parameter)
        sfc_alb_dir_ffi = self.convert_to_from_cffi(surface_albedo_direct)
        sfc_alb_diff_ffi = self.convert_to_from_cffi(surface_albedo_diffuse)

        self._rte_extension.lib.sw_solver_2stream(
            num_cols_ffi, num_layers_ffi, num_gpts_ffi, top_at_1_ffi,
            tau_ffi, ssa_ffi, asym_ffi,
            zen_angle_ffi, sfc_alb_dir_ffi, sfc_alb_diff_ffi,
            upward_flux_ffi, downward_flux_ffi,
            direct_flux_ffi)

        direct_flux_out = self.convert_to_from_cffi(
            direct_flux_ffi, num_bytes=direct_beam_flux.nbytes)
        down_flux_out = self.convert_to_from_cffi(
            downward_flux_ffi, num_bytes=downward_flux.nbytes)
        up_flux_out = self.convert_to_from_cffi(
            upward_flux_ffi, num_bytes=downward_flux.nbytes)

        return (
            direct_flux_out.reshape(direct_beam_flux.shape),
            up_flux_out.reshape(downward_flux.shape),
            down_flux_out.reshape(downward_flux.shape))

    def get_cffi_dims_and_dir(self, downward_flux):
        '''
        Utility function to get pointers to required
        quantities.
        '''
        top_at_1 = None

        if self.direction == 'bottom_to_top':
            top_at_1 = False
        elif self.direction == 'top_to_bottom':
            top_at_1 = True

        num_gpts_ffi = self._ffi.new('int *',
                                     downward_flux.shape[0])
        num_cols_ffi = self._ffi.new('int *',
                                     downward_flux.shape[2])
        num_layers_ffi = self._ffi.new('int *',
                                       downward_flux.shape[1] - 1)

        top_at_1_ffi = self._ffi.new('bool *',
                                     top_at_1)

        return num_gpts_ffi, num_cols_ffi, num_layers_ffi, top_at_1_ffi

    def convert_to_from_cffi(self, c_or_numpy_array, num_bytes=None):

        '''
        Utility function to convert to and from numpy and ffi buffer interfaces

        Args:
            c_or_numpy_array (buffer):
                an object that is either a numpy array or a FFI cdata pointer

            num_bytes (integer, optional):
                size of buffer when converting from CFFI to ndarray
        '''

        if isinstance(c_or_numpy_array, np.ndarray):
            out_buffer = self._ffi.cast('{} *'.format(self._ffi_type),
                                        c_or_numpy_array.ctypes.data)
        elif isinstance(c_or_numpy_array, self._ffi.CData):
            out_buffer = np.frombuffer(self._ffi.buffer(
                c_or_numpy_array, num_bytes), dtype=self._numpy_type)
        else:
            raise ValueError(
                'Buffer must be either numpy.ndarray or ffi CData,\
                not {}'.format(type(c_or_numpy_array)))

        return out_buffer


def validate_bc_inputs(downward_flux,
                       incident_flux=None,
                       scale_factor=None):
    '''
    Validate the inputs provided to the boundary condition
    functions.

    Args:
        downward_flux (np.ndarray):
            The downward flux array. Must have dimensions
            :code:`[num_g_points, num_layers, num_columns]`.

        incident_flux (np.ndarray, optional):
            The boundary condition to be applied at the top of the model.
            Must have dimensions :code:`[num_g_points, num_columns]`.
            Ignored if :code:`zero_incident_flux = True`.

        scale_factor (np.ndarray, optional):
            The scale factor to be applied to :code:`incident_flux`. Must
            have dimensions :code:`[num_columns]`.
            Ignored if :code:`zero_incident_flux = True`.

    '''
    try:
        assert len(downward_flux.shape) == 3
    except AssertionError:
        raise ValueError(
            'Incorrect shape for downward flux {}'.format(
                downward_flux.shape))

    if incident_flux is not None:
        try:
            assert len(incident_flux.shape) == 2
        except AssertionError:
            raise ValueError(
                'Incorrect shape for incident flux {}'.format(
                    incident_flux.shape))

        try:
            assert incident_flux.shape[0] == downward_flux.shape[0]
        except AssertionError:
            raise ValueError(
                'Incompatible gpts dimension for downward ({}) and incident flux ({})'.format(
                    downward_flux.shape[0], incident_flux.shape[0]))
        try:
            assert incident_flux.shape[1] == downward_flux.shape[2]
        except AssertionError:
            raise ValueError(
                'Incompatible columns dimension for downward ({}) and incident flux ({})'.format(
                    downward_flux.shape[2], incident_flux.shape[1]))

    if scale_factor is not None:
        try:
            assert len(scale_factor.shape) == 1
        except AssertionError:
            raise ValueError(
                'Incorrect shape for scale factor {}'.format(
                    scale_factor.shape))

        try:
            assert scale_factor.shape[0] == downward_flux.shape[2]
        except AssertionError:
            raise ValueError(
                'Incompatible columns dimension for down flux ({}) and scale factor ({})'.format(
                    downward_flux.shape[2], scale_factor.shape[0]))


def validate_lw_solver_inputs(planck_src_mid,
                              planck_src_iface_decreasing,
                              planck_src_iface_increasing,
                              surface_emissivity,
                              surface_source,
                              downward_flux,
                              optical_tau,
                              secant_propagation_angle=None,
                              quadrature_weight=None,
                              num_angles=None,
                              ssa=None,
                              asym_param=None,):
    '''
    Validate longwave solver inputs.
    '''

    try:
        assert len(downward_flux.shape) == 3
    except AssertionError:
        raise ValueError(
            'Incorrect shape for downward flux {}'.format(
                downward_flux.shape))

    ngpts, nlayers, ncols = downward_flux.shape
    nlayers -= 1
    array_shape = (ngpts, nlayers, ncols)

    assert_shapes(planck_src_mid, array_shape)
    assert_shapes(planck_src_iface_decreasing, array_shape)
    assert_shapes(planck_src_iface_increasing, array_shape)
    assert_shapes(surface_emissivity, (ngpts, ncols))
    assert_shapes(surface_source, (ngpts, ncols))
    assert_shapes(optical_tau, array_shape)

    if num_angles:
        assert_shapes(secant_propagation_angle, (num_angles,))
        assert_shapes(quadrature_weight, (num_angles,))
    else:
        if secant_propagation_angle:
            assert_shapes(secant_propagation_angle, (ngpts, ncols))
        if quadrature_weight:
            assert isinstance(quadrature_weight, (np.float64, np.float32))

    if ssa:
        assert_shapes(ssa, array_shape)
    if asym_param:
        assert_shapes(asym_param, array_shape)


def validate_sw_solver_inputs(cos_zenith_angle,
                              direct_beam_flux,
                              optical_tau,
                              single_scattering_albedo=None,
                              asymmetry_parameter=None,
                              surface_albedo_direct=None,
                              surface_albedo_diffuse=None,
                              downward_flux=None):

    '''
    Validate longwave solver inputs.
    '''

    assert len(direct_beam_flux.shape) == 3
    ngpts, nlayers, ncols = direct_beam_flux.shape
    nlayers -= 1
    array_shape = (ngpts, nlayers, ncols)

    assert_shapes(optical_tau, array_shape)
    assert_shapes(cos_zenith_angle, (ncols,))
    if single_scattering_albedo:
        assert_shapes(single_scattering_albedo, array_shape)
    if asymmetry_parameter:
        assert_shapes(asymmetry_parameter, array_shape)
    if surface_albedo_direct:
        assert_shapes(surface_albedo_direct, (ngpts, ncols))
    if surface_albedo_diffuse:
        assert_shapes(surface_albedo_diffuse, (ngpts, ncols))
    if downward_flux:
        assert_shapes(downward_flux, direct_beam_flux.shape)


def assert_shapes(array, target):
    '''
    Match shapes or throw exception
    '''
    try:
        assert array.shape == target
    except AssertionError:
        raise ValueError(
            'Incompatible input array dimensions')


class UserCompiledRTEInterface(CompiledRTEInterface):
    '''toa_
    Class that loads a Compiled wrapper from a user defined library.
    '''
    def __init__(self,
                 compiler='gcc',
                 python_module_name='_rte_wrapper',
                 compiled_library_path=None, **kwargs):
        '''
        Initialise the object with information needed for Compiled to build wrapper.

        Args:
            compiler (string, optional):
                what command must be used to invoke the compiler. Defaults to
                :code:`gcc`. Only used when :code:`use_external_library` is
                :code:`True`.

            python_module_name (string, optional):
                The name of the compiled module in Python, to use when invoking
                :code:`import`. Defaults to :code:`_rte_wrapper`.

            compiled_library_path (string, optional):
                the relative (or absolute) path to the user compiled RTE
                library. Is expanded to absolute path internally. Only used
                when :code:`use_external_library` is :code:`True`.
        '''

        self._compiler = compiler
        self._module_name = python_module_name
        absolute_library_path = os.path.abspath(compiled_library_path)
        if not os.path.exists(absolute_library_path):
            raise OSError('{} does not exist!'.format(
                absolute_library_path))
        self._compiled_library_path = absolute_library_path

        super().__init__(**kwargs)

        self._rte_extension = self.compile_wrapper()

    def compile_wrapper(self):
        '''
        utility function to compile the wrapper and return the imported module.
        '''

        ####################
        # Setup part
        ####################

        os.environ['CC'] = self._compiler

        ####################
        # Compilation part
        ####################

        # cdef() expects a single string declaring the C types, functions and
        # globals needed to use the shared object. It must be in valid C syntax.
        self._ffi.cdef(self._api.bc_gpt_api)
        self._ffi.cdef(self._api.bc_factor_api)
        self._ffi.cdef(self._api.bc_0_api)
        self._ffi.cdef(self._api.lw_solver_noscat_api)
        self._ffi.cdef(self._api.lw_solver_noscat_gauss_quad_api)
        self._ffi.cdef(self._api.lw_solver_2stream_api)
        self._ffi.cdef(self._api.sw_solver_noscat_api)
        self._ffi.cdef(self._api.sw_solver_2stream_api)

        # set_source() gives the name of the python extension module to
        # produce, and some C source code as a string.  This C code needs
        # to make the declarated functions, types and globals available,
        # so it is often just the "#include".
        self._ffi.set_source(
            self._module_name,
            """
            #include <stdbool.h>   // the C header of the library
            #include "rte_wrapper.h"
            """,
            library_dirs=[os.getcwd()],
            extra_compile_args=[self._api.compiler_def],
            extra_link_args=[
                os.path.abspath(self._compiled_library_path),
                '-lgfortran'],
        )

        self._ffi.compile(verbose=True)

        return importlib.import_module(self._module_name)
