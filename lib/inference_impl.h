/* -*- c++ -*- */
/* 
 * Copyright 2018 Deepwave Digital Inc.
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

#ifndef INCLUDED_WAVELEARNER_INFERENCE_IMPL_H
#define INCLUDED_WAVELEARNER_INFERENCE_IMPL_H

#include <wavelearner/inference.h>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "wavelearner_logger.h"

namespace gr {
namespace wavelearner {

class inference_impl : public inference {
 public:
  inference_impl(const std::string& plan_filepath, const size_t input_vlen,
                 const size_t output_vlen, const size_t batch_size);
  ~inference_impl();

  int work(int noutput_items, gr_vector_const_void_star& input_items,
           gr_vector_void_star& output_items);

 private:
  static constexpr int kNumIOPorts = 2;
  static constexpr int kInvalidBindingIndex = -1;
  CUcontext context_;
  nvinfer1::IRuntime* infer_runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* infer_context_;
  float* buffers_[kNumIOPorts];
  int input_binding_index_;
  int output_binding_index_;
  size_t input_buffer_size_;
  size_t output_buffer_size_;
  size_t batch_size_;
  // A note on terminology: a "signal segment" is a single set of digital
  // samples that are processed together to give some inference output. In image
  // processing, this is akin to an image, where you may have multiple images in
  // a batch.
  int total_signal_segments_processed_;
  std::chrono::duration<double> total_work_time_;
  WavelearnerLogger wavelearner_logger_;
  
  // Helper functions to load and validate the engine
  cudaError load_engine(const std::string& plan_filepath);
  cudaError validate_engine();

  // Helper function to combine all the I/O dimensions (i.e., NCHW dimensions)
  // into a single number, which can then be compared against the vlen
  // parameters.
  size_t get_samples_per_batch(const nvinfer1::Dims& dims) const noexcept;
  
  void print_performance_metrics() const noexcept;

  // Error handling function
  void throw_due_to_cuda_err(const int error_code, const std::string& operation)
      const {
    std::stringstream error_stream;
    error_stream << "Inference Block failed to " << operation
        << " (error_code = " << error_code << ").";
    throw std::runtime_error(error_stream.str());
  }
  // Wrapper error handling functions for the different CUDA APIs
  void throw_on_cuda_drv_err(const CUresult error_code,
                             const std::string& operation) const {
    if (error_code != CUDA_SUCCESS) {
      throw_due_to_cuda_err(static_cast<int>(error_code), operation);
    }
  }
  void throw_on_cuda_rt_err(const cudaError error_code,
                            const std::string& operation) const {
    if (error_code != cudaSuccess) {
      throw_due_to_cuda_err(static_cast<int>(error_code), operation);
    }
  }
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_INFERENCE_IMPL_H

