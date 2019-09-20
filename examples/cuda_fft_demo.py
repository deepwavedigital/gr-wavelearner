#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: AIR-T Example
# Author: Deepwave Digital
# Generated: Fri Sep 20 20:31:32 2019
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import sip
import soapy
import sys
import wavelearner


class cuda_fft_demo(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "AIR-T Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("AIR-T Example")
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "cuda_fft_demo")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 31.25e6
        self.nfft = nfft = 1024
        self.frequency = frequency = 3.1e9
        self.fft_batch_size = fft_batch_size = 128

        ##################################################
        # Blocks
        ##################################################
        self.wavelearner_fft_1 = wavelearner.fft(fft_batch_size*nfft, nfft, True)
        self.soapy_source_0 = \
          soapy.source(1, "device=SoapyAIRT", samp_rate, "fc32")
        
        
        
        
        self.soapy_source_0.set_frequency(0, frequency)
        self.soapy_source_0.set_gain(0, 0)
        self.soapy_source_0.set_gain_mode(0, True)
        self.soapy_source_0.set_dc_offset_mode(0, True)
         
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            nfft,
            -0.5,
            1.0/float(nfft),
            "Normalized Frequency",
            "PSD (dB)",
            "cuFFT",
            1 # Number of inputs
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.05)
        self.qtgui_vector_sink_f_0.set_y_axis(-60, 10)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)
        
        labels = ["", "", "", "", "",
                  "", "", "", "", ""]
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])
        
        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_gr_complex*nfft, fft_batch_size)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, fft_batch_size*nfft)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, nfft, 0)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(nfft)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))    
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))    
        self.connect((self.blocks_stream_to_vector_0, 0), (self.wavelearner_fft_1, 0))    
        self.connect((self.blocks_vector_to_stream_0, 0), (self.blocks_complex_to_mag_squared_0, 0))    
        self.connect((self.soapy_source_0, 0), (self.blocks_stream_to_vector_0, 0))    
        self.connect((self.wavelearner_fft_1, 0), (self.blocks_vector_to_stream_0, 0))    

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "cuda_fft_demo")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_nfft(self):
        return self.nfft

    def set_nfft(self, nfft):
        self.nfft = nfft
        self.qtgui_vector_sink_f_0.set_x_axis(-0.5, 1.0/float(self.nfft))

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.soapy_source_0.set_frequency(0, self.frequency)

    def get_fft_batch_size(self):
        return self.fft_batch_size

    def set_fft_batch_size(self, fft_batch_size):
        self.fft_batch_size = fft_batch_size


def main(top_block_cls=cuda_fft_demo, options=None):

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
