#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Classifier Test
# Author: Deepwave Digital, Inc
# Description: Test for Wavelearn Inference Block
# Generated: Wed Sep 18 16:30:58 2019
##################################################


from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import wavelearner


class classifier_test(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Classifier Test")

        ##################################################
        # Variables
        ##################################################
        self.output_len = output_len = 16
        self.input_len = input_len = 4096
        self.fs = fs = 100e6
        self.batch_size = batch_size = 4

        ##################################################
        # Blocks
        ##################################################
        self.wavelearner_terminal_sink_0 = wavelearner.terminal_sink(output_len*batch_size, batch_size)
        self.type_confert = blocks.short_to_float(1, 32768)
        self.throttle = blocks.throttle(gr.sizeof_short*1, fs,True)
        self.source = blocks.file_source(gr.sizeof_short*1, 'classifier_test.dat', False)
        self.inference = wavelearner.inference('classifier_test_gtx950M_trt4.plan', False, input_len*batch_size, output_len*batch_size, batch_size)
        self.buffer0 = blocks.stream_to_vector(gr.sizeof_float*1, batch_size*input_len)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.buffer0, 0), (self.inference, 0))
        self.connect((self.inference, 0), (self.wavelearner_terminal_sink_0, 0))
        self.connect((self.source, 0), (self.throttle, 0))
        self.connect((self.throttle, 0), (self.type_confert, 0))
        self.connect((self.type_confert, 0), (self.buffer0, 0))

    def get_output_len(self):
        return self.output_len

    def set_output_len(self, output_len):
        self.output_len = output_len

    def get_input_len(self):
        return self.input_len

    def set_input_len(self, input_len):
        self.input_len = input_len

    def get_fs(self):
        return self.fs

    def set_fs(self, fs):
        self.fs = fs
        self.throttle.set_sample_rate(self.fs)

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def main(top_block_cls=classifier_test, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print "Error: failed to enable real-time scheduling."

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == '__main__':
    main()
