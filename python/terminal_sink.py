#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2018, 2021 Deepwave Digital Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


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
        print('Batch %d' % self.batch_ctr)
        for segment in batch:
            vec_str = ' '.join(["{0:0.2f}".format(i) for i in segment])
            print('   ArgMax at %d: [ ' % numpy.argmax(segment) + vec_str + ' ]')
        self.batch_ctr += 1
        return len(input_items[0])
