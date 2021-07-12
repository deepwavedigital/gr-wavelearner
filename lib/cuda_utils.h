/* -*- c++ -*- */
/*
 * Copyright 2019 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
