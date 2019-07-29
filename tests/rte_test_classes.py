'''
Base class for RTE test suite
'''
import abc
import numpy as np

class BaseTestRTE(object):
    '''
    base class to test all interfaces
    '''
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def _interface(self):
        return None

    def test_apply_bc_0(self):
        '''
        apply zero boundary condition
        '''
        downward_flux = np.random.randn(2, 20, 3)

        out_array = self._interface.apply_zero_bc(downward_flux)

        self.validate_bcs(out_array, 0)

    def test_apply_inc_flux(self):
        '''
        apply incident flux
        '''
        downward_flux = np.random.randn(2, 20, 3)
        incident_flux = 10*np.ones((2, 3), dtype=np.double)

        out_array = self._interface.apply_gpoint_bc(
            downward_flux, incident_flux)

        self.validate_bcs(out_array, incident_flux)


    def test_apply_scaled_inc_flux(self):
        '''
        apply incident flux
        '''
        downward_flux = np.random.randn(2, 20, 3)
        incident_flux = 10*np.ones((2, 3), dtype=np.double)
        scale_factor = np.arange(3, dtype=np.double)

        out_array = self._interface.apply_scaled_gpoint_bc(
            downward_flux, incident_flux, scale_factor)

        self.validate_bcs(out_array, scale_factor*incident_flux)


    def validate_bcs(self, array, target):
        '''
        check if bcs are applied correctly.

        Args:
            array (ndarray):
                output from RTE.

            target (ndarray or float):
                what to check against.
        '''

        if self._interface.direction == 'top_to_bottom':
            assert np.all(array[:, 0, :] == target)
        else:
            assert np.all(array[:, -1, :] == target)
