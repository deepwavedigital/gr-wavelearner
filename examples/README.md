# Examples for GR-WAVELEARNER
This folder contains an example of how to use the gr-wavelearner package.

## Files:

- classifier_test.grc: Executable GRC file demonstrating how to use gr-wavelearner package for
                       inference of a neural network
- uff2plan.py: Executable Python script to convert a uff file to a plan file for execution
- cuda_fft_demo.grc: Executable GRC file demonstrating how to use the cuFFT block

## Instructions:

1. Train your neural network with any deep learning framework and layers that are supported by [NVIDIA's TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html#supported-ops)
2. Save the trained model as a `.uff` file
3. Use `uff2plan.py` to convert the .uff file to a .plan file for inference
4. Open GNU Radio Companion and open `classifier_test.grc`
5. Open the **Inference Block** and select the `.plan` file created in Step 2.
6. Execute the GRC flowgraph



Copyright 2018 Deepwave Digital Inc.