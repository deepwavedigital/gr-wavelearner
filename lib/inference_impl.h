/* -*- c++ -*- */
/*
 * Copyright 2018-2021 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WAVELEARNER_INFERENCE_IMPL_H
#define INCLUDED_WAVELEARNER_INFERENCE_IMPL_H

#include <wavelearner/inference.h>
#include <chrono>
#include <memory>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "cuda_utils.h"

namespace gr {
namespace wavelearner {

class inference_impl : public inference {
 public:
  inference_impl(const std::string& plan_filepath, const bool complex_input,
                 const size_t input_vlen, const size_t output_vlen,
                 const size_t batch_size);
  ~inference_impl();

  int work(int noutput_items, gr_vector_const_void_star& input_items,
           gr_vector_void_star& output_items);

 private:
  static constexpr auto kBlockName = "inference";
  static constexpr int kNumIOPorts = 2;
  static constexpr int kInvalidBindingIndex = -1;
  CUcontext context_;
  std::unique_ptr<nvinfer1::IRuntime> infer_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> infer_context_;
  float* buffers_[kNumIOPorts];
  int input_binding_index_;
  int output_binding_index_;
  size_t input_buffer_size_;
  size_t output_buffer_size_;
  size_t batch_size_;
  bool explicit_batch_size_;
  // A note on terminology: a "signal segment" is a single set of digital
  // samples that are processed together to give some inference output. In image
  // processing, this is akin to an image, where you may have multiple images in
  // a batch.
  int total_signal_segments_processed_;
  std::chrono::duration<double> total_work_time_;
  TrtLogger trt_logger_;
  CudaErrorHandler err_handler_;
  
  // Helper functions to load and validate the engine
  cudaError load_engine(const std::string& plan_filepath);
  cudaError validate_engine();

  // Helper function that converts the vector length set in the GNU Radio
  // flowgraph to the number of bytes an I/O port expects. Used only in the
  // constructor, since from that point forward, we only keep track of the
  // total number of bytes an input or output buffer requires.
  size_t get_gr_buffer_size(const size_t vlen,
                            const bool is_complex = false) const noexcept;
  // Helper function to combine all the I/O dimensions of a TensorRT engine's
  // input or output binding into a number of bytes (aka. a buffer size).
  // Used to determine if the engine's parameters and the vector lengths
  // set in the GNU Radio flowgraph match up.
  size_t get_trt_binding_size(const nvinfer1::Dims& dims) const noexcept;
  
  void print_performance_metrics() const noexcept;
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_INFERENCE_IMPL_H

