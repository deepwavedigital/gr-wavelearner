#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2018 Deepwave Digital Inc.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

# TODO: Add capability for non classifier models
# TODO: Convert to C++ to reduce overhead

import numpy
from gnuradio import gr


class terminal_sink(gr.sync_block):
    # Prints output layer of classifier to terminal for troubleshooting
    def __init__(self, input_vlen, batch_size):
        self.input_vlen = input_vlen
        self.batch_size = batch_size
        gr.sync_block.__init__(self, name="terminal_sink",
                               in_sig=[(numpy.float32, self.input_vlen)], out_sig=None)
        self.batch_ctr = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        batch = in0.reshape(self.batch_size, -1)
        print 'Batch %d' % self.batch_ctr
        for segment in batch:
            vec_str = ' '.join(["{0:0.2f}".format(i) for i in segment])
            print '   ArgMax at %d: [ ' % numpy.argmax(segment) + vec_str + ' ]'
        self.batch_ctr += 1
        return len(input_items[0])
