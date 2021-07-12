/* -*- c++ -*- */
/*
 * Copyright 2018-2019, 2021 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WAVELEARNER_INFERENCE_H
#define INCLUDED_WAVELEARNER_INFERENCE_H

#include <wavelearner/api.h>
#include <string>
#include <gnuradio/sync_block.h>

namespace gr {
namespace wavelearner {

class WAVELEARNER_API inference : virtual public gr::sync_block {
 public:
  typedef std::shared_ptr<inference> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wavelearner::inference.
   *
   * To avoid accidental use of raw pointers, wavelearner::inference's
   * constructor is in a private implementation
   * class. wavelearner::inference::make is the public interface for
   * creating new instances.
   */
  static sptr make(const std::string& plan_filepath,
                   const bool complex_input,
                   const size_t input_vlen,
                   const size_t output_vlen,
                   const size_t batch_size);
};

} // namespace wavelearner
} // namespace gr

#endif  // INCLUDED_WAVELEARNER_INFERENCE_H

