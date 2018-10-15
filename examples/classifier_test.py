#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Classifier Test
# Author: Deepwave Digital, Inc
# Description: Test for Wavelearn Inference Block
# Generated: Mon Oct 15 10:09:26 2018
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
        self.samp_rate = samp_rate = 25e6
        self.output_length = output_length = 16
        self.input_length = input_length = 4096
        self.batch_size = batch_size = 4

        ##################################################
        # Blocks
        ##################################################
        self.wavelearner_terminal_sink_0 = wavelearner.terminal_sink(output_length * batch_size, batch_size)
        self.wavelearner_inference_0 = wavelearner.inference("/home/john/googledrive/software/gr-wavelearner/test/classifier_test.plan", input_length*batch_size, output_length*batch_size, batch_size)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_short*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_float*1, batch_size*input_length)
        self.blocks_short_to_float_0 = blocks.short_to_float(1, 32768)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_short*1, "/home/john/googledrive/software/gr-wavelearner/test/classifier_test.dat", False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.blocks_short_to_float_0, 0), (self.blocks_stream_to_vector_0, 0))    
        self.connect((self.blocks_stream_to_vector_0, 0), (self.wavelearner_inference_0, 0))    
        self.connect((self.blocks_throttle_0, 0), (self.blocks_short_to_float_0, 0))    
        self.connect((self.wavelearner_inference_0, 0), (self.wavelearner_terminal_sink_0, 0))    

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_output_length(self):
        return self.output_length

    def set_output_length(self, output_length):
        self.output_length = output_length

    def get_input_length(self):
        return self.input_length

    def set_input_length(self, input_length):
        self.input_length = input_length

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
