# Copyright 2011,2012,2016,2018,2019 Free Software Foundation, Inc.
#
# This file was generated by gr_modtool, a tool from the GNU Radio framework
# This file is a part of gr-wavelearner
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

cmake_minimum_required(VERSION 3.17)

########################################################################
# Find CUDA and TensorRT libraries
########################################################################

# CUDA is not packaged for conda on aarch64. So we need some special
# configuration to allow cmake to find host system libraries. TensorRT
# is not packaged for conda on any architecture yet.
set(_old_path_mode_inc ${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
set(_old_path_mode_lib ${CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
if(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${_old_path_mode_inc})
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${_old_path_mode_lib})
else()
    find_package(TensorRT REQUIRED)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${_old_path_mode_inc})
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${_old_path_mode_lib})
    find_package(CUDAToolkit REQUIRED)
endif()

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

list(APPEND wavelearner_sources
    cuda_utils.cc
    fft_impl.cc
    inference_impl.cc
)

set(wavelearner_sources "${wavelearner_sources}" PARENT_SCOPE)
if(NOT wavelearner_sources)
    MESSAGE(STATUS "No C++ sources... skipping lib/")
    return()
endif(NOT wavelearner_sources)

add_library(gnuradio-wavelearner SHARED ${wavelearner_sources})
target_link_libraries(gnuradio-wavelearner
                      gnuradio::gnuradio-runtime
                      CUDA::cuda_driver
                      CUDA::cudart
                      CUDA::cufft
                      ${TensorRT_LIBRARY}
  )
target_include_directories(gnuradio-wavelearner
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
  )
target_include_directories(gnuradio-wavelearner
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${TensorRT_INCLUDE_DIRS}>
  )
set_target_properties(gnuradio-wavelearner PROPERTIES DEFINE_SYMBOL "gnuradio_wavelearner_EXPORTS")

if(APPLE)
    set_target_properties(gnuradio-wavelearner PROPERTIES
        INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    )
endif(APPLE)

########################################################################
# Install built library files
########################################################################
include(GrMiscUtils)
GR_LIBRARY_FOO(gnuradio-wavelearner)

########################################################################
# Print summary
########################################################################
message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Building for version: ${VERSION} / ${LIBVER}")

########################################################################
# Build and register unit test
########################################################################
include(GrTest)

# If your unit tests require special include paths, add them here
#include_directories()
# List all files that contain Boost.UTF unit tests here
list(APPEND test_wavelearner_sources
)
# Anything we need to link to for the unit tests go here
list(APPEND GR_TEST_TARGET_DEPS gnuradio-wavelearner)

if(NOT test_wavelearner_sources)
    MESSAGE(STATUS "No C++ unit tests... skipping")
    return()
endif(NOT test_wavelearner_sources)

foreach(qa_file ${test_wavelearner_sources})
    GR_ADD_CPP_TEST("wavelearner_${qa_file}"
        ${CMAKE_CURRENT_SOURCE_DIR}/${qa_file}
    )
endforeach(qa_file)
