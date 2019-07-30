import os
from cffi import FFI
from .rte.rte_api import RTEAPIDefinition

desired_compiler = 'gcc'
rte_library_path = 'spyrate/_lib/librte.a'
precision = 'double'
extension_name = "spyrate._rte_wrapper_double_precision"
rte_api = RTEAPIDefinition()

absolute_library_path = os.path.abspath(rte_library_path)

####################
# Setup part
####################

if not os.environ.get('CC'):
    os.environ['CC'] = desired_compiler

precision_dict = {}
precision_def_dict = {}

precision_dict['single'] = 'float'
precision_dict['double'] = 'double'

precision_def_dict['single'] = '-DRTE_SINGLE_PRECISION'
precision_def_dict['double'] = '-DRTE_DOUBLE_PRECISION'

library_precision = precision_dict[precision]

####################
# Compilation part
####################

ffibuilder = FFI()
# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef(rte_api.bc_gpt_api)
ffibuilder.cdef(rte_api.bc_factor_api)
ffibuilder.cdef(rte_api.bc_0_api)
ffibuilder.cdef(rte_api.lw_solver_noscat_api)
ffibuilder.cdef(rte_api.lw_solver_noscat_gauss_quad_api)
ffibuilder.cdef(rte_api.lw_solver_2stream_api)
ffibuilder.cdef(rte_api.sw_solver_noscat_api)
ffibuilder.cdef(rte_api.sw_solver_2stream_api)


# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source(extension_name,
                      """
     #include <stdbool.h>   // the C header of the library
     #include "_lib/rte_wrapper.h"
                      """,
                      library_dirs=[os.getcwd()],
                      include_dirs=[os.getcwd()+'/spyrate/'],
                      extra_compile_args=[precision_def_dict[precision]],
                      extra_link_args=[os.path.abspath(rte_library_path), '-lgfortran'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
