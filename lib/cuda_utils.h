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

#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <NvInfer.h>

#ifndef INCLUDED_WAVELEARNER_CUDA_UTILS_H
#define INCLUDED_WAVELEARNER_CUDA_UTILS_H

namespace gr {
namespace wavelearner {

static constexpr auto kDefaultCtxFlags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST;

class CudaErrorHandler {
 public:
  CudaErrorHandler() = delete;
  CudaErrorHandler(const std::string& block_name) : block_name_(block_name) {}
  ~CudaErrorHandler() {}
  void throw_on_cuda_drv_err(const CUresult error_code,
                             const std::string& description) const;
  void throw_on_cuda_rt_err(const cudaError error_code,
                            const std::string& description) const;
  void throw_on_cufft_err(const cufftResult error_code,
                          const std::string& description) const;

 private:
  void add_error_header(std::stringstream* const error_stream,
                        const std::string& description) const noexcept;
  std::string block_name_;
};

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* const msg) noexcept override;
  void log_error(const char* const msg) noexcept { log(Severity::kERROR, msg); }
  void log_warn(const char* const msg) noexcept { log(Severity::kWARNING, msg); }
  void log_info(const char* const msg) noexcept { log(Severity::kINFO, msg); }
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_CUDA_UTILS_H
