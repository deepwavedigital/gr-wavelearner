/* -*- c++ -*- */
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

#ifndef INCLUDED_WAVELEARNER_FFT_H
#define INCLUDED_WAVELEARNER_FFT_H

#include <wavelearner/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace wavelearner {

class WAVELEARNER_API fft : virtual public gr::sync_block {
 public:
  typedef boost::shared_ptr<fft> sptr;
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
