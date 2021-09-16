# Examples for GR-WAVELEARNER
This folder contains an example of how to use the gr-wavelearner package.

## Files:

- gpu_fft_demo.grc: Executable GRC file demonstrating how to use the cuFFT block
- classifier_test.grc: Executable GRC file demonstrating how to use gr-wavelearner package for
                       inference of a neural network
                       
## gpu_fft_demo Instructions:

Please see the [tutorial](https://docs.deepwavedigital.com/Tutorials/3_cufft.html) that Deepwave has put together for this example.

## classifier_test Instructions:

1. Generate the PLAN file for your trained network. For examples on how to generate the PLAN file, see
[here](https://github.com/deepwavedigital/airstack-examples/tree/master/python/inference)
2. Open GNU Radio Companion and open `classifier_test.grc`
3. Open the **Inference Block** and select the `.plan` file created in Step 1.
4. Execute the GRC flowgraph

Note that the examples utilize a SoapyAIRT source block. However, these examples can be tailored to any type of SDR by simply swapping the source block with the appropriate block for your hardware.



Copyright 2018-2021 Deepwave Digital Inc.
