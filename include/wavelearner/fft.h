/* -*- c++ -*- */
/*
 * Copyright 2019, 2021 Deepwave Digital Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WAVELEARNER_FFT_H
#define INCLUDED_WAVELEARNER_FFT_H

#include <wavelearner/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace wavelearner {

class WAVELEARNER_API fft : virtual public gr::sync_block {
 public:
  typedef std::shared_ptr<fft> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wavelearner::fft.
   *
   * To avoid accidental use of raw pointers, wavelearner::fft's
   * constructor is in a private implementation
   * class. wavelearner::fft::make is the public interface for
   * creating new instances.
   */
  static sptr make(const size_t vlen, const size_t fft_size,
                   const bool forward_fft);
};

}  // namespace wavelearner
}  // namespace gr

#endif  // INCLUDED_WAVELEARNER_FFT_H
