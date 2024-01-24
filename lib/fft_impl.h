/* -*- c++ -*- */
/*
 * Copyright 2019 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WAVELEARNER_FFT_IMPL_H
#define INCLUDED_WAVELEARNER_FFT_IMPL_H

#include <wavelearner/fft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuda_utils.h"

namespace gr {
namespace wavelearner {

class fft_impl : public fft {
 public:
  fft_impl(const size_t vlen, const size_t fft_size, const bool forward_fft);
  ~fft_impl();

  int work(int noutput_items, gr_vector_const_void_star& input_items,
           gr_vector_void_star& output_items);

 private:
  static constexpr auto kBlockName = "fft";
  size_t samples_per_buffer_;
  size_t buffer_size_;
  size_t batch_size_;
  CUcontext context_;
  cudaStream_t stream_;
  cufftComplex* fft_data_;
  cufftHandle fft_plan_;
  int fft_direction_;
  CudaErrorHandler err_handler_;
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_FFT_IMPL_H
