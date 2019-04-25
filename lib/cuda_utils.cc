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
