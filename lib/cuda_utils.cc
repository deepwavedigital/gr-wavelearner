/* -*- c++ -*- */
/*
 * Copyright 2019 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_utils.h"
#include <iostream>
#include <stdexcept>

namespace gr {
namespace wavelearner {

void CudaErrorHandler::throw_on_cuda_drv_err(
    const CUresult error_code, const std::string& description) const {
  if (error_code == CUDA_SUCCESS) return;
  std::stringstream error_stream;
  add_error_header(&error_stream, description);
  error_stream << "-> CUDA Driver Error Code: " << error_code << std::endl;
  const char* error_str = NULL;
  if (cuGetErrorName(error_code, &error_str) == CUDA_SUCCESS) {
    error_stream << "-> CUDA Driver Error Name: " << error_str << std::endl;
  }
  if (cuGetErrorString(error_code, &error_str) == CUDA_SUCCESS) {
    error_stream << "-> CUDA Driver Error Details: " << error_str << std::endl;
  }
  throw std::runtime_error(error_stream.str());
}

void CudaErrorHandler::throw_on_cuda_rt_err(
    const cudaError error_code, const std::string& description) const {
  if (error_code == cudaSuccess) return;
  std::stringstream error_stream;
  add_error_header(&error_stream, description);
  error_stream << "-> CUDA Runtime Error Code: " << error_code << std::endl;
  error_stream << "-> CUDA Runtime Error Details: "
    << cudaGetErrorString(error_code) << std::endl;
  throw std::runtime_error(error_stream.str());
}

void CudaErrorHandler::throw_on_cufft_err(
    const cufftResult error_code, const std::string& description) const {
  if (error_code == CUFFT_SUCCESS) return;
  std::stringstream error_stream;
  add_error_header(&error_stream, description);
  error_stream << "-> cuFFT Error Code: " << error_code << std::endl;
  throw std::runtime_error(error_stream.str());    
}

void CudaErrorHandler::add_error_header(
    std::stringstream* const error_stream,
    const std::string& description) const noexcept {
  *error_stream << "gr-wavelearner: " << block_name_ << " block failed to "
    << description << "!" << std::endl;
}

void TrtLogger::log(const Severity severity, const char* const msg) noexcept {
  static constexpr auto kLibName("TensorRT");
  if ((severity == Severity::kERROR) ||
      (severity == Severity::kINTERNAL_ERROR)) {
    std::cerr << kLibName << " ERROR: " << msg << std::endl;
  } else if (severity == Severity::kWARNING) {
    std::cerr << kLibName <<  " WARNING: " << msg << std::endl;
  } else {  // informational message
    std::cout << kLibName <<  " INFO: " << msg << std::endl;    
  }
}

}  // namespace wavelearner
}  // namespace gr
