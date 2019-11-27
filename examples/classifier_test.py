#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Classifier Test
# Author: Deepwave Digital, Inc
# Description: Example for Wavelearner Inference Block
# Generated: Tue Nov 26 18:57:38 2019
##################################################


from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import soapy
import wavelearner


class classifier_test(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Classifier Test")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 31.25e6
        self.output_len = output_len = 12
        self.input_len = input_len = 2048
        self.frequency = frequency = 2.4e9
        self.batch_size = batch_size = 128

        ##################################################
        # Blocks
        ##################################################
        self.wavelearner_terminal_sink_0 = wavelearner.terminal_sink(output_len*batch_size, batch_size)
        self.soapy_source_0 = \
         soapy.source(1, "device=SoapyAIRT", '', samp_rate, "fc32")



        self.soapy_source_0.set_frequency(0, frequency)
        self.soapy_source_0.set_gain(0, 0)
        self.soapy_source_0.set_gain_mode(0, True)
        self.soapy_source_0.set_dc_offset_mode(0, True)

        self.inference = wavelearner.inference('saved_model.plan', True, input_len*batch_size, output_len*batch_size, batch_size)
        self.buffer0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, batch_size*input_len)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.buffer0, 0), (self.inference, 0))
        self.connect((self.inference, 0), (self.wavelearner_terminal_sink_0, 0))
        self.connect((self.soapy_source_0, 0), (self.buffer0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_output_len(self):
        return self.output_len

    def set_output_len(self, output_len):
        self.output_len = output_len

    def get_input_len(self):
        return self.input_len

    def set_input_len(self, input_len):
        self.input_len = input_len

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.soapy_source_0.set_frequency(0, self.frequency)

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def main(top_block_cls=classifier_test, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print "Error: failed to enable real-time scheduling."

    tb = top_block_cls()
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
