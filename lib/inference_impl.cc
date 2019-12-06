/* -*- c++ -*- */
/* 
 * Copyright 2018-2019 Deepwave Digital Inc.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

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
  return gnuradio::get_initial_sptr(
      new inference_impl(plan_filepath, complex_input, input_vlen,
                         output_vlen, batch_size));
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
  if (infer_context_ != nullptr) {
    infer_context_->destroy();
    infer_context_ = nullptr;
  }
  if (engine_ != nullptr) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (infer_runtime_ != nullptr) {
    infer_runtime_->destroy();
    infer_runtime_ = nullptr;
  }
  cuCtxDestroy(context_);
}

cudaError inference_impl::load_engine(const std::string& plan_filepath) {
  infer_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
  if (infer_runtime_ == nullptr) {
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
  engine_ = infer_runtime_->deserializeCudaEngine(serialized_engine.data(),
                                                  serialized_engine.size(),
                                                  nullptr);
  if (engine_ == nullptr) {
    trt_logger_.log_error("Failed to deserialize engine.");
    return cudaErrorInitializationError;
  }
  
  return cudaSuccess;
}

cudaError inference_impl::validate_engine() {
  if (engine_ == nullptr) {
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
  } else if (batch_size_ != max_batch_size) {
    trt_logger_.log_warn("Unoptimized batch size detected.");
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
  infer_context_ = engine_->createExecutionContext();
  if (infer_context_ == nullptr) {
    trt_logger_.log_error("Unable to create TensorRT execution context.");
    return cudaErrorInitializationError;
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
  // Count total number of elements in the N-dimensional tensor
  size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    count *= dims.d[i];
  }
  // Account for batch size and convert to number of bytes based on
  // all inputs and outputs to TRT being FP32.
  return count * batch_size_ * sizeof(float);
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

  if (!infer_context_->execute(batch_size_,
                               reinterpret_cast<void**>(buffers_))) {
    err_handler_.throw_on_cuda_rt_err(cudaErrorLaunchFailure,
                                      "execute inference");
  }

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

