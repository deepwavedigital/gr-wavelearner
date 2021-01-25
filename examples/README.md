# Examples for GR-WAVELEARNER
This folder contains an example of how to use the gr-wavelearner package.

## Files:

- classifier_test.grc: Executable GRC file demonstrating how to use gr-wavelearner package for
                       inference of a neural network
- cuda_fft_demo.grc: Executable GRC file demonstrating how to use the cuFFT block

## Instructions:

1. Generate the PLAN file for your trained network. For examples on how to generate the PLAN file, see
[here](https://github.com/deepwavedigital/airstack-examples/tree/master/python/inference)
2. Open GNU Radio Companion and open `classifier_test.grc`
3. Open the **Inference Block** and select the `.plan` file created in Step 1.
4. Execute the GRC flowgraph



Copyright 2018-2021 Deepwave Digital Inc.
