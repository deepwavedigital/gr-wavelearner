/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(fft.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(afd0b1a2bce86f9bdfd3194416c5417f)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <wavelearner/fft.h>
// pydoc.h is automatically generated in the build directory
#include <fft_pydoc.h>

void bind_fft(py::module& m)
{

    using fft    = gr::wavelearner::fft;


    py::class_<fft,
        gr::sync_block,
        gr::block,
        gr::basic_block,
        std::shared_ptr<fft>>(m, "fft", D(fft))

        .def(py::init(&fft::make),
           py::arg("vlen"),
           py::arg("fft_size"),
           py::arg("forward_fft"),
           D(fft,make)
        )
        



        ;




}








