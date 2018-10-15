# GR-WAVELEARNER
Incorporate Deep Learning into GNU Radio

### Author
<p align="center">
<img src="http://www.deepwavedigital.net/logos/deepwave-logo-2-white.png" Width="50%" />
</p>

This software is written by **Deepwave Digital, Inc.** [www.deepwavedigital.com]().

Should you find it useful, please consider a donation to help
[support the effort](https://www.crowdsupply.com/deepwave-digital/air-t).


&nbsp;
### Inquiries
  - General company contact: [info@deepwavedigital.com](mailto:info@deepwavedigital.com)
  - Bugs/issues/requests contact: [support@deepwavedigital.com](mailto:support@deepwavedigital.com)

&nbsp;
### Description
This out of tree (OOT) module for GNU Radio contains code to provide an interface to call
[NVIDIA's TensorRT](https://developer.nvidia.com/tensorrt) deep learning binaries from a GNU Radio
flowgraph. TensorRT allows for deep learning networks to be optimized for inference operations on an
NVIDIA graphics processing units (GPU).


&nbsp;
### Dependencies:
  - **SWIG** (v3.0.8 or newer)
    - SWIG does not come with GNU Radio and may not be included with your Linux distribution.
    - Building this package will fail if SWIG is not  installed and you will get an import error
      for the inference block. The easiest way to install SWIG is:

         `$ sudo apt install swig`           
  - **TensorRT** (v3.0 GA or newer) [1]
    - Both TensorRT and gr-wavelearner use C++14. As a result, the OOT modules are
            compiled with the C++14 standard flag set in CMake.
    - This package has only been tested with TensorRT being installed via the .deb packages. Any
      other method of installation may work but will require you make sure the software sees all
      the dependencies. 
  - **GNU Radio** (v3.7.9 or newer) [2]

&nbsp;
### Requirments:
  - NVIDIA GPU that supports TensorRT v3.0 or newer or the NVIDA Tegra TX2


&nbsp;
### Current Blocks
  - __**Inference**__ - This C++ block uses TensorRT to perform inference. Currently, it assumes one input
                        and one output node for the network. It requires a TensorRT PLAN file be
                        provided that contains information on what operations should take place.
                        The block currently does not handle complex data types. This is primarily
                        because the vast majority of deep learning software tools (if not all)
                        do not support complex operations yet. This does not mean that you cannot
                        perform deep learning on complex data, it just means that the neural
                        network will treat it as real. For example, one method is to associate
                        the in-phase and quadrature samples within the network's convolution
                        kernels [3].
  
  - __**Terminal Sink**__ - This Python block prints the output of a deep learning classifier
                            to the terminal for debugging and to help the user get up and running
                            quickly.

  - More to come ...

&nbsp;
## How to Build and Install gr-wavelearner
1. Install Dependencies listed above (no seriously, make sure they are installed)
   - Make sure you can import gnuradio and trt from the same python environment in which you are
     installing gr-wavelearner

2. Clone the gr-wavelearner repo
   ```
   $ git clone https://github.com/deepwavedigital/gr-wavelearner.git
   ```

3. Install the OOT Module
   ```
   $ cd gr-wavelearner
   $ mkdir build
   $ cd build
   $ cmake ../
   $ make
   $ sudo make install
   $ sudo ldconfig
   ```
  
4. To uninstall gr-wavelearner Blocks from GNU Radio Companion
   ```
   cd gr-wavelearner/build
   sudo make uninstall
   ```

&nbsp;

&nbsp;
## Screen Shots
#### Example Flow Graph of Deep Learning Classifier
<p align="center">
<img src="http://www.deepwavedigital.net/images/classifier_test.png" Width="50%" />
</p>

&nbsp;
#### Inference Block
<p align="center">
<img src="http://www.deepwavedigital.net/images/inference_block.png" Width="25%" />
</p>

&nbsp;
#### Terminal Sink Block
<p align="center">
<img src="http://www.deepwavedigital.net/images/terminal_sink_block.png" Width="25%" />
</p>



&nbsp;
### Known Issues / Future Enhancements
- Currently the inference block's work() function only processes a single
   vector each time it is called. However, there may be more data available. It
   may improve performance to process all available data by either performing
   inference in a loop or (assuming the TensorRT engine has a large enough
   maximum batch size) combine the vectors into a larger batch.
- As of now, the input and output node dimensions are combined into a vector
   length. This makes the block easier to program; however, it may be more
   useful for the user to instead provide NCHW dimensions (i.e., to specify each
   dimension individually). Additionally, this would allow the engine validation
   step to check each dimension instead of combining everything into one vector
   length. For example, in the current code, if a 1x2x2048 node is expected,
   then the code will accept any shape (e.g., 4x64x16, 4096x1x1, 8x8x64, etc.)
   as long as the resulting dimensions combine into a vector length of 4096.
- Currently the inference block assumes one input node and one output node. As
   a result, networks with more than one input or output are not currently
   supported.
- The current block assumes that the GPU to be used for inference is device #0,
   which may not be correct for multiple GPU systems.
- As of now, only the device mapped memory API (aka. "zero copy" or UVA) is
   used since (in previous testing) this memory API provided the best
   performance on the Jetson TX2. Unified memory support should be considered
   since it would be fairly easy to add (very little code would change) and also
   improve performance for PCIe based cards.
- Performance metrics are always printed to the console. The user may want to
   log these instead or alternatively just ignore them completely.
- First data that comes out of inference block is garbage. Need to initialize
   as zeros.

&nbsp;
### Tags
  - Deep Learning
  - Artificial Intelligence
  - Machine Learning
  - TensorRT
  - GPU
  - Deepwave Digital
  - Jetson
  - GNU Radio
  

&nbsp;
### Credits and License
GR-WAVELEARNER is designed and written by **Deepwave Digital, Inc.** [www.deepwavedigital.com]()
and is licensed under the GNU General Public License. Copyright notices at the top of source files.

Should you find it useful, please consider a donation to help
[support the effort](https://www.crowdsupply.com/deepwave-digital/air-t).


&nbsp;
### References
[1] NVIDIA TensorRT - Programmable Inference Accelerator:
    [https://developer.nvidia.com/tensorrt]()

[2] GNU Radio - The Free & Open Software Radio Ecosystem:
    [https://www.gnuradio.org]()
    
[2] Making Sense of Signals Presentation- NVIDIA GPU Technology Conference:
    [http://on-demand.gputechconf.com/gtc/2018/video/S8375/]()
