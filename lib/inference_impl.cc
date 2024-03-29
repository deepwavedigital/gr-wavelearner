/* -*- c++ -*- */
/*
 * Copyright 2018-2021 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "inference_impl.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <gnuradio/io_signature.h>

namespace gr {
namespace wavelearner {

inference::sptr inference::make(const std::string& plan_filepath,
                                const bool complex_input,
                                const size_t input_vlen,
                                const size_t output_vlen,
                                const size_t batch_size) {
  return gnuradio::make_block_sptr<inference_impl>(
      plan_filepath, complex_input, input_vlen, output_vlen, batch_size);
}

inference_impl::inference_impl(const std::string& plan_filepath,
                               const bool complex_input,
                               const size_t input_vlen,
                               const size_t output_vlen,
                               const size_t batch_size)
    : context_(),
      infer_runtime_(nullptr),
      engine_(nullptr),
      infer_context_(nullptr),
      buffers_(),
      input_binding_index_(kInvalidBindingIndex),
      output_binding_index_(kInvalidBindingIndex),
      input_buffer_size_(get_gr_buffer_size(input_vlen, complex_input)),
      output_buffer_size_(get_gr_buffer_size(output_vlen)),
      batch_size_(batch_size),
      explicit_batch_size_(false),
      total_signal_segments_processed_(0),
      total_work_time_(std::chrono::seconds::zero()),
      trt_logger_(),
      err_handler_(kBlockName),
      gr::sync_block(
           kBlockName,
           gr::io_signature::make(1, 1, get_gr_buffer_size(input_vlen, complex_input)),
           gr::io_signature::make(1, 1, get_gr_buffer_size(output_vlen))) {
  err_handler_.throw_on_cuda_drv_err(cuInit(0), "initialize CUDA driver API");
  CUdevice dev;
  err_handler_.throw_on_cuda_drv_err(cuDeviceGet(&dev, 0), "get CUDA device");
  err_handler_.throw_on_cuda_drv_err(
      cuCtxCreate(&context_, kDefaultCtxFlags, dev),
      "create device context");

  err_handler_.throw_on_cuda_rt_err(load_engine(plan_filepath),
                                    "load TensorRT engine");
  err_handler_.throw_on_cuda_rt_err(validate_engine(),
                                    "validate TensorRT engine");

  err_handler_.throw_on_cuda_rt_err(
      cudaHostAlloc(reinterpret_cast<void**>(&buffers_[input_binding_index_]),
                    input_buffer_size_, cudaHostAllocMapped),
      "allocate input buffer");
  err_handler_.throw_on_cuda_rt_err(
      cudaHostAlloc(reinterpret_cast<void**>(&buffers_[output_binding_index_]),
                    output_buffer_size_, cudaHostAllocMapped),
      "allocate output buffer");
  
  err_handler_.throw_on_cuda_drv_err(cuCtxPopCurrent(&context_),
                                     "pop context during init");
}

inference_impl::~inference_impl() {
  print_performance_metrics();
  for (int i = 0; i < kNumIOPorts; ++i) {
    if (buffers_[i] != nullptr) {
      cudaFreeHost(buffers_[i]);
      buffers_[i] = nullptr;
    }
  }
  infer_context_.reset(nullptr);
  engine_.reset(nullptr);
  infer_runtime_.reset(nullptr);
  cuCtxDestroy(context_);
}

cudaError inference_impl::load_engine(const std::string& plan_filepath) {
  infer_runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  if (!infer_runtime_) {
    trt_logger_.log_error("Failed to create inference runtime.");
    return cudaErrorStartupFailure;
  }

  std::ifstream plan_file(plan_filepath.c_str(), std::ifstream::binary);
  if (!plan_file.is_open()) {
    trt_logger_.log_error("Failed to open PLAN file.");
    return cudaErrorUnknown;
  }

  std::stringstream plan_buffer;
  plan_buffer << plan_file.rdbuf();
  if (plan_file.bad() || plan_file.fail()) {
    plan_file.close();
    trt_logger_.log_error("Failed to read PLAN file.");
    return cudaErrorUnknown;
  }

  const std::string serialized_engine = plan_buffer.str();
  plan_file.close();
  engine_.reset(infer_runtime_->deserializeCudaEngine(serialized_engine.data(),
                                                      serialized_engine.size()));
  if (!engine_) {
    trt_logger_.log_error("Failed to deserialize engine.");
    return cudaErrorInitializationError;
  }
  
  return cudaSuccess;
}

cudaError inference_impl::validate_engine() {
  if (!engine_) {
    trt_logger_.log_error("Attempted to validate NULL engine.");
    return cudaErrorInitializationError;
  }

  if (engine_->getNbBindings() != kNumIOPorts) {
    trt_logger_.log_error("Engine has invalid number of bindings.");
    return cudaErrorInvalidValue;
  }

  const size_t max_batch_size = static_cast<size_t>(engine_->getMaxBatchSize());
  if (batch_size_ > max_batch_size) {
    trt_logger_.log_error("Unsupported batch size detected.");
    return cudaErrorInvalidValue;
  }

  for (int i = 0; i < kNumIOPorts; ++i) {
    if (engine_->bindingIsInput(i)) {
      if (input_binding_index_ != kInvalidBindingIndex) {
        trt_logger_.log_error("Multiple input bindings detected.");
        return cudaErrorUnknown;
      } else {
        input_binding_index_ = i;
      }
    } else {
      if (output_binding_index_ != kInvalidBindingIndex) {
        trt_logger_.log_error("Multiple output bindings detected.");
        return cudaErrorUnknown;
      }
      output_binding_index_ = i;
    }
    // Inputs and outputs into TRT are always FP32. For reduced precision
    // inference, the engine performs a conversion under the hood.
    if (engine_->getBindingDataType(i) != nvinfer1::DataType::kFLOAT) {
      trt_logger_.log_error("Unsupported I/O data type found.");
      return cudaErrorInvalidValue;
    }
  }

  if ((input_binding_index_ == output_binding_index_) ||
      (input_binding_index_ == kInvalidBindingIndex) ||
      (output_binding_index_ == kInvalidBindingIndex)) {
    trt_logger_.log_error("Invalid I/O bindings detected.");
    return cudaErrorUnknown;
  }

  const nvinfer1::Dims input_dims =
      engine_->getBindingDimensions(input_binding_index_);
  if (get_trt_binding_size(input_dims) != input_buffer_size_) {
    trt_logger_.log_error("Input size mismatch detected.");
    return cudaErrorInvalidValue;
  }

  const nvinfer1::Dims output_dims =
    engine_->getBindingDimensions(output_binding_index_);
  if (get_trt_binding_size(output_dims) != output_buffer_size_) {
    trt_logger_.log_error("Output size mismatch detected.");
    return cudaErrorInvalidValue;
  }

  // If everything checks out, we build the execution context.
  infer_context_.reset(engine_->createExecutionContext());
  if (!infer_context_) {
    trt_logger_.log_error("Unable to create TensorRT execution context.");
    return cudaErrorInitializationError;
  }

  // Handle the case of a PLAN that is expecting an explicit batch size (common
  // case for PLAN files generated from ONNX files). If we are using an explicit
  // batch size, we need to set the input binding dimensions accordingly.
  if (input_dims.d[0] == -1) {
    explicit_batch_size_ = true;
    nvinfer1::Dims new_input_dims(input_dims);
    new_input_dims.d[0] = batch_size_;
    const bool resize_success =
      infer_context_->setBindingDimensions(input_binding_index_, new_input_dims);
    if (!resize_success) {
      trt_logger_.log_error("Failed to resize input binding.");
      return cudaErrorInitializationError;
    }
  }

  return cudaSuccess;
}

size_t inference_impl::get_gr_buffer_size(const size_t vlen, const bool is_complex)
    const noexcept {
  const size_t float_vector_size = vlen * sizeof(float);
  // A complex vector is made up of two floats, where one float is the real
  // component and the other float is the imaginary component.
  return is_complex ? (2 * float_vector_size) : float_vector_size;
}

size_t inference_impl::get_trt_binding_size(const nvinfer1::Dims& dims)
    const noexcept {
  size_t count = batch_size_;  // total # of elements in the N-dim tensor
  // Skip over the first index (batch size) since it's already accounted for.
  for (int i = 1; i < dims.nbDims; ++i) count *= dims.d[i];
  return count * sizeof(float);  // convert to bytes
}

void inference_impl::print_performance_metrics() const noexcept {
  if (total_signal_segments_processed_ > 0) {
    const double sig_segments_processed =
        static_cast<double>(total_signal_segments_processed_);
    const auto total_work_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(total_work_time_);
    const double throughput = sig_segments_processed / total_work_time_.count();
    const double time_per_segment_us =
        total_work_time_us.count() / sig_segments_processed;
    std::cout << "Processed " << total_signal_segments_processed_
        << " Signal Segments in " << total_work_time_.count() << "s"
        << std::endl;
    std::cout << "Throughput: " << throughput << " Segments/s" << std::endl;
    std::cout << "Average Time per Segment: " << time_per_segment_us << "us"
        << std::endl;
  } else {
    std::cout << "0 Signal Segments Processed." << std::endl;
  }
}

int inference_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items) {
  static std::chrono::time_point<std::chrono::steady_clock> start, end;
  // Even if we have a complex vector, it is safe to cast to a float pointer,
  // since a complex number in GNU Radio is just two floats in adjacent
  // memory. Earlier in the constructor, we already accounted for the fact
  // that (given an equal number of samples) a complex buffer will be twice
  // the size of a float buffer.
  const float* const in = reinterpret_cast<const float* const>(input_items[0]);
  float* const out = reinterpret_cast<float* const>(output_items[0]);

  start = std::chrono::steady_clock::now();
  err_handler_.throw_on_cuda_drv_err(cuCtxPushCurrent(context_),
                                     "push context");
  std::memcpy(buffers_[input_binding_index_], in, input_buffer_size_);

  bool infer_success = false;
  void** buffs = reinterpret_cast<void**>(buffers_);
  if (explicit_batch_size_)
    infer_success = infer_context_->executeV2(buffs);
  else
    infer_success = infer_context_->execute(batch_size_, buffs);
  if (!infer_success)
    err_handler_.throw_on_cuda_rt_err(cudaErrorLaunchFailure, "run inference");

  std::memcpy(out, buffers_[output_binding_index_], output_buffer_size_);
  err_handler_.throw_on_cuda_drv_err(cuCtxPopCurrent(&context_),
                                     "pop context during work()");
  end = std::chrono::steady_clock::now();

  std::chrono::duration<double> time_elapsed = end - start;
  total_work_time_ += time_elapsed;
  total_signal_segments_processed_ += batch_size_;
  return 1;  // only process one vector at a time
}

}  // namespace wavelearner
}  // namespace gr

