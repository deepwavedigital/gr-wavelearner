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
 
#ifndef INCLUDED_WAVELEARNER_LOGGER_H
#define INCLUDED_WAVELEARNER_LOGGER_H

#include <iostream>
#include <NvInfer.h>

namespace gr {
namespace wavelearner {

class WavelearnerLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) override {
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
  
  void log_error(const char* msg) { log(Severity::kERROR, msg); }
  void log_warn(const char* msg) { log(Severity::kWARNING, msg); }
  void log_info(const char* msg) { log(Severity::kINFO, msg); }
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_LOGGER_H
