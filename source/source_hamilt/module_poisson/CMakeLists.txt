# source_hamilt/module_poisson/CMakeLists.txt


set( sources
    mt_correction.cpp
    parabolic_correction.cpp
    mt_poisson.cpp
)

add_library(hamilt_poisson OBJECT ${sources})


if(ENABLE_COVERAGE)
    add_coverage(hamilt_poisson)
endif()