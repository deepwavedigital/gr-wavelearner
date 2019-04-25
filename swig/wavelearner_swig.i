/* -*- c++ -*- */

#define WAVELEARNER_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "wavelearner_swig_doc.i"

%{
#include "wavelearner/fft.h"
#include "wavelearner/inference.h"
%}

%include "wavelearner/fft.h"
%include "wavelearner/inference.h"

GR_SWIG_BLOCK_MAGIC2(wavelearner, fft);
GR_SWIG_BLOCK_MAGIC2(wavelearner, inference);
