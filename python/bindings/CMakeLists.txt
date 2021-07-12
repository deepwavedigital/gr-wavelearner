# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT wavelearner_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT wavelearner_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND wavelearner_python_files
    inference_python.cc
    fft_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(wavelearner 
   ../..
   gr::wavelearner
   "${wavelearner_python_files}")

install(TARGETS wavelearner_python DESTINATION ${GR_PYTHON_DIR}/wavelearner COMPONENT pythonapi)
