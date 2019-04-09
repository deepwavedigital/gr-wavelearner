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

#include "fft_impl.h"
#include <cstring>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <boost/atomic.hpp>
#include <gnuradio/io_signature.h>

namespace gr {
namespace wavelearner {

const std::string fft_impl::kBlockName = "fft";

fft::sptr fft::make(const size_t vlen, const size_t fft_size,
                    const bool forward_fft) {
  return gnuradio::get_initial_sptr(new fft_impl(vlen, fft_size, forward_fft));
}

fft_impl::fft_impl(const size_t vlen, const size_t fft_size,
                   const bool forward_fft)
    : samples_per_buffer_(vlen),
      buffer_size_(sizeof(cufftComplex) * vlen),
      context_(),
      stream_(),
      fft_data_(NULL),
      fft_plan_(),
      fft_direction_(forward_fft ? CUFFT_FORWARD : CUFFT_INVERSE),
      gr::sync_block(
          kBlockName,
          gr::io_signature::make(1, 1, (sizeof(cufftComplex) * vlen)),
          gr::io_signature::make(1, 1, (sizeof(cufftComplex) * vlen))) {
  static boost::atomic<bool> init_done(false);
  if (!init_done) {
    throw_on_cuda_drv_err(cuInit(0), "driver API initialization");
    init_done = true;
  }

  CUdevice dev;
  throw_on_cuda_drv_err(cuDeviceGet(&dev, 0), "getting CUDA device");
  static const unsigned int kContextFlags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST;
  throw_on_cuda_drv_err(cuCtxCreate(&context_, kContextFlags, dev),
                        "creating device context");
  throw_on_cuda_rt_err(cudaStreamCreate(&stream_), "creating stream");

  throw_on_cuda_rt_err(cudaHostAlloc(reinterpret_cast<void**>(&fft_data_),
                                     buffer_size_, cudaHostAllocMapped),
                       "allocating GPU buffer");

  const size_t batch_size = samples_per_buffer_ / fft_size;
  throw_on_cufft_err(cufftPlan1d(&fft_plan_, fft_size, CUFFT_C2C, batch_size),
                     "creating FFT plan");
  throw_on_cufft_err(cufftSetStream(fft_plan_, stream_), "setting FFT stream");

  throw_on_cuda_drv_err(cuCtxPopCurrent(&context_), "popping context");
}

fft_impl::~fft_impl() {
  cudaFreeHost(fft_data_);
  fft_data_ = NULL;
  cufftDestroy(fft_plan_);
  cudaStreamDestroy(stream_);
  cuCtxDestroy(context_);
}

int fft_impl::work(int noutput_items, gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items) {
  const gr_complex* const in =
    reinterpret_cast<const gr_complex* const>(input_items[0]);
  gr_complex* const out = reinterpret_cast<gr_complex* const>(output_items[0]);

  throw_on_cuda_drv_err(cuCtxPushCurrent(context_), "pushing context");
  for (int i = 0; i < noutput_items; ++i) {
    const int buffer_index = i * samples_per_buffer_;
    std::memcpy(&fft_data_[0], &in[buffer_index], buffer_size_);
    throw_on_cufft_err(cufftExecC2C(fft_plan_, fft_data_, fft_data_,
                                    fft_direction_),
                       "executing FFT");
    throw_on_cuda_rt_err(cudaStreamSynchronize(stream_),
                         "synchronizing stream");
    std::memcpy(&out[buffer_index], &fft_data_[0], buffer_size_);
  }

  throw_on_cuda_drv_err(cuCtxPopCurrent(&context_), "popping context");
  return noutput_items;
}

void fft_impl::throw_on_cuda_drv_err(const CUresult error_code,
                                     const std::string& description) {
  if (error_code != CUDA_SUCCESS) {
    std::stringstream error_stream;
    error_stream << kBlockName << ": error " << description << "." << std::endl;
    error_stream << "-> CUDA Driver Error Code: " << error_code << std::endl;
    const char* error_str = NULL;
    if (cuGetErrorName(error_code, &error_str) == CUDA_SUCCESS) {
      error_stream << "-> CUDA Driver Error Name: " << error_str << std::endl;
    }
    if (cuGetErrorString(error_code, &error_str) == CUDA_SUCCESS) {
      error_stream << "-> CUDA Driver Error Details: " << error_str
        << std::endl;
    }
    throw std::runtime_error(error_stream.str());
  }
}

void fft_impl::throw_on_cuda_rt_err(const cudaError error_code,
                                    const std::string& description) {
  if (error_code != cudaSuccess) {
    std::stringstream error_stream;
    error_stream << kBlockName << ": error " << description << "." << std::endl;
    error_stream << "-> CUDA Runtime Error Code: " << error_code << std::endl;
    error_stream << "-> CUDA Runtime Error Details: "
      << cudaGetErrorString(error_code) << std::endl;
    throw std::runtime_error(error_stream.str());
  }
}

void fft_impl::throw_on_cufft_err(const cufftResult error_code,
                                  const std::string& description) {
  if (error_code != CUFFT_SUCCESS) {
    std::stringstream error_stream;
    error_stream << kBlockName << ": error " << description << "." << std::endl;
    error_stream << "-> cuFFT Error Code: " << error_code << std::endl;
    throw std::runtime_error(error_stream.str());    
  }
}

}  // namespace wavelearner
}  // namespace gr
