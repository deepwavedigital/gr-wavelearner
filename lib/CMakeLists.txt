# Copyright 2011,2012 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

########################################################################
# Find CUDA and TensorRT libraries
########################################################################
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

include_directories(${Boost_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

list(APPEND wavelearner_sources
    inference_impl.cc
)

set(wavelearner_sources "${wavelearner_sources}" PARENT_SCOPE)
if(NOT wavelearner_sources)
	MESSAGE(STATUS "No C++ sources... skipping lib/")
	return()
endif(NOT wavelearner_sources)

add_library(gnuradio-wavelearner SHARED ${wavelearner_sources})
target_link_libraries(gnuradio-wavelearner
                      ${Boost_LIBRARIES}
                      ${GNURADIO_ALL_LIBRARIES}
                      ${TensorRT_LIBRARY}
                      ${CUDA_CUDA_LIBRARY}   # CUDA Driver API
                      ${CUDA_CUDART_LIBRARY} # CUDA Runtime API
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
install(TARGETS gnuradio-wavelearner
    LIBRARY DESTINATION lib${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)

########################################################################
# Build and register unit test, if CppUnit library is available
########################################################################
if(CPPUNIT_FOUND)
    include(GrTest)

    include_directories(${CPPUNIT_INCLUDE_DIRS})
    link_directories(${CPPUNIT_LIBRARY_DIRS})

    list(APPEND test_wavelearner_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wavelearner.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/qa_wavelearner.cc
    )

    add_executable(test-wavelearner ${test_wavelearner_sources})

    target_link_libraries(
        test-wavelearner
        ${GNURADIO_RUNTIME_LIBRARIES}
        ${Boost_LIBRARIES}
        ${CPPUNIT_LIBRARIES}
        gnuradio-wavelearner
    )
endif()
