'''
This module implements the Aspyrate class, an interface to the RTE library
'''
import importlib
from .rte.rte_interface import CompiledRTEInterface


class Spyrate(object):
    '''
    The Aspyrate class, which provides a generic wrapper to computational
    kernels implementing the RTE interface.
    '''

    def __init__(self, use_external_library=False,
                 external_rte_interface=None,
                 precision='double',
                 direction='bottom_to_top'):
        '''
        Initialise the RTE wrapper.

         Args:
             use_external_library (bool, optional):
                 wrap a user-compiled RTE library instead of the internal
                 libraries. use internal RTE library by default.

            external_rte_interface (RTEInterface):
                A class that implements RTEInterface.

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

        self._precision = precision
        self._direction = direction

        if precision == 'single':
            raise NotImplementedError(
                'Single precision version not available yet.')

        if use_external_library:
            self._rte_implementation = external_rte_interface
        else:
            module_name = 'spyrate._rte_wrapper_{}_precision'.format(self._precision)
            rte_extension = importlib.import_module(module_name)
            self._rte_implementation = CompiledRTEInterface(
                rte_extension=rte_extension,
                precision=self._precision, direction=self._direction)

    def calculate_radiative_transfer(self):
        '''
        main external interface
        '''

    def reduce_calculated_fluxes(self):
        '''
        hook to reduce radiative fluxes to desired format
        '''
