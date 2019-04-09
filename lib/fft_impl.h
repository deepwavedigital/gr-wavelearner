/* -*- c++ -*- */
/* 
 * Copyright 2019 Deepwave Digital Inc.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_WAVELEARNER_FFT_IMPL_H
#define INCLUDED_WAVELEARNER_FFT_IMPL_H

#include <wavelearner/fft.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace wavelearner {

class fft_impl : public fft {
 public:
  fft_impl(const size_t vlen, const size_t fft_size, const bool forward_fft);
  ~fft_impl();

  int work(int noutput_items, gr_vector_const_void_star& input_items,
           gr_vector_void_star& output_items);

 private:
  static const std::string kBlockName;
  size_t samples_per_buffer_;
  size_t buffer_size_;
  CUcontext context_;
  cudaStream_t stream_;
  cufftComplex* fft_data_;
  cufftHandle fft_plan_;
  int fft_direction_;
  void throw_on_cuda_drv_err(const CUresult error_code,
                             const std::string& description);
  void throw_on_cuda_rt_err(const cudaError error_code,
                            const std::string& description);
  void throw_on_cufft_err(const cufftResult error_code,
                          const std::string& description);
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_FFT_IMPL_H
