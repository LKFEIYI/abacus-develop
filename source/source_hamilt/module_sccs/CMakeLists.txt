add_library(sccs OBJECT
  sccs_diagnostics.cpp
  sccs_cavity.cpp
  sccs_cavity_derivatives.cpp
  sccs_parameters.cpp
  sccs_functional.cpp
  sccs_ionic_charge.cpp
  sccs_lowpass.cpp
  sccs_ionic_force.cpp
  sccs_pcc_0d_coulomb.cpp
  sccs_pcc_2d_coulomb.cpp
  sccs_pw_coulomb.cpp
  sccs_response.cpp)

if(ENABLE_COVERAGE)
  add_coverage(sccs)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
