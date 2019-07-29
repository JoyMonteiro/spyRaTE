'''
Test suite for the RTE related code
'''
import pytest
import numpy as np
import spyrate._rte_wrapper_double_precision as _rte_wrapper_double_precision
from .rte_test_classes import BaseTestRTE
from spyrate.rte.rte_interface import (
    validate_bc_inputs, CompiledRTEInterface)


class TestDoublePrecisionTopToBottom(BaseTestRTE):
    '''
    Test internal double precision library
    '''

    @property
    def _interface(self):
        return CompiledRTEInterface(
            direction='top_to_bottom',
            rte_extension=_rte_wrapper_double_precision)


class TestDoublePrecisionBottomToTop(BaseTestRTE):
    '''
    Test internal double precision library
    '''

    @property
    def _interface(self):
        return CompiledRTEInterface(
            direction='bottom_to_top',
            rte_extension=_rte_wrapper_double_precision)


class TestBCValidation(object):
    '''
    Test the validation of boundary conditions
    '''

    def test_wrong_downflux_shape(self):
        '''
        Test bc validation
        '''
        downward_flux = np.zeros((10, 10))

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux)

        assert 'Incorrect shape' in str(ex.value)

    def test_wrong_incflux_shape(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros(5)

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux, incident_flux)

        assert 'Incorrect shape' in str(ex.value)

    def test_dim_1_incompatible_down_incflux(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros((1, 5))

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux, incident_flux)

        assert 'Incompatible' in str(ex.value)

    def test_dim_2_incompatible_down_incflux(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros((2, 4))

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux, incident_flux)

        assert 'Incompatible' in str(ex.value)

    def test_dim_both_incompatible_down_incflux(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros((1, 4))

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux, incident_flux)

        assert 'Incompatible' in str(ex.value)

    def test_incompatible_scale_downflux(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros((2, 5))
        scale_factor = np.zeros(4)

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux,
                               incident_flux, scale_factor)

        assert 'Incompatible' in str(ex.value)

    def test_wrong_scale_shape(self):
        '''
        Test bc validation
        '''

        downward_flux = np.zeros((2, 3, 5))
        incident_flux = np.zeros((2, 5))
        scale_factor = np.zeros((4, 5))

        with pytest.raises(ValueError) as ex:
            validate_bc_inputs(downward_flux,
                               incident_flux, scale_factor)

        assert 'Incorrect shape' in str(ex.value)
