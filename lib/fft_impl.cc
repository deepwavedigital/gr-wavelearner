/* -*- c++ -*- */
/*
 * Copyright 2019, 2021 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_impl.h"
#include <cstring>
#include <gnuradio/io_signature.h>

namespace gr {
namespace wavelearner {

fft::sptr fft::make(const size_t vlen, const size_t fft_size,
                    const bool forward_fft) {
  return gnuradio::make_block_sptr<fft_impl>(vlen, fft_size, forward_fft);
}

fft_impl::fft_impl(const size_t vlen, const size_t fft_size,
                   const bool forward_fft)
    : samples_per_buffer_(vlen),
      buffer_size_(sizeof(cufftComplex) * vlen),
      context_(),
      stream_(),
      fft_data_(nullptr),
      fft_plan_(),
      fft_direction_(forward_fft ? CUFFT_FORWARD : CUFFT_INVERSE),
      err_handler_(kBlockName),
      gr::sync_block(
          kBlockName,
          gr::io_signature::make(1, 1, (sizeof(cufftComplex) * vlen)),
          gr::io_signature::make(1, 1, (sizeof(cufftComplex) * vlen))) {
  err_handler_.throw_on_cuda_drv_err(cuInit(0), "initialize CUDA driver API");
  CUdevice dev;
  err_handler_.throw_on_cuda_drv_err(cuDeviceGet(&dev, 0), "get CUDA device");
  err_handler_.throw_on_cuda_drv_err(
      cuCtxCreate(&context_, kDefaultCtxFlags, dev),
      "create device context");
  err_handler_.throw_on_cuda_rt_err(cudaStreamCreate(&stream_),
                                    "create stream");

  err_handler_.throw_on_cuda_rt_err(
      cudaHostAlloc(reinterpret_cast<void**>(&fft_data_), buffer_size_,
                    cudaHostAllocMapped),
      "allocate GPU buffer");

  const size_t batch_size = samples_per_buffer_ / fft_size;
  err_handler_.throw_on_cufft_err(
      cufftPlan1d(&fft_plan_, fft_size, CUFFT_C2C, batch_size),
      "create FFT plan");
  err_handler_.throw_on_cufft_err(cufftSetStream(fft_plan_, stream_),
                                  "set FFT stream");

  err_handler_.throw_on_cuda_drv_err(cuCtxPopCurrent(&context_),
                                     "pop context during init");
}

fft_impl::~fft_impl() {
  cudaFreeHost(fft_data_);
  fft_data_ = nullptr;
  cufftDestroy(fft_plan_);
  cudaStreamDestroy(stream_);
  cuCtxDestroy(context_);
}

int fft_impl::work(int noutput_items, gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items) {
  const gr_complex* const in =
    reinterpret_cast<const gr_complex* const>(input_items[0]);
  gr_complex* const out = reinterpret_cast<gr_complex* const>(output_items[0]);

  err_handler_.throw_on_cuda_drv_err(cuCtxPushCurrent(context_),
                                     "push context");
  for (int i = 0; i < noutput_items; ++i) {
    const int buffer_index = i * samples_per_buffer_;
    std::memcpy(&fft_data_[0], &in[buffer_index], buffer_size_);
    err_handler_.throw_on_cufft_err(
        cufftExecC2C(fft_plan_, fft_data_, fft_data_, fft_direction_),
        "execute FFT");
    err_handler_.throw_on_cuda_rt_err(cudaStreamSynchronize(stream_),
                                      "synchronize stream");
    std::memcpy(&out[buffer_index], &fft_data_[0], buffer_size_);
  }

  err_handler_.throw_on_cuda_drv_err(cuCtxPopCurrent(&context_),
                                     "pop context during work()");
  return noutput_items;
}

}  // namespace wavelearner
}  // namespace gr
