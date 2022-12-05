# Examples for GR-WAVELEARNER
This folder contains an example of how to use the gr-wavelearner package.

## Files:

- `gpu_fft_demo.grc`: Executable GRC file demonstrating how to use the cuFFT block
- `classifier_test.grc`: Executable GRC file demonstrating how to use gr-wavelearner
  package for inference of a neural network
                       
## gpu_fft_demo Instructions:
Please see the [tutorial](https://docs.deepwavedigital.com/Tutorials/3_cufft.html) that
Deepwave has put together for this example.

## classifier_test Instructions:
Please see the [tutorial](https://docs.deepwavedigital.com/Tutorials/inference.html)
that Deepwave has put together for this example.

1. Generate the PLAN file for your trained network. As an example, you may generate a plan
   file from the onnx file provided in our [airstack-examples](https://github.com/deepwavedigital/airstack-examples)
   repository. See the **Creating a PLAN File* section below for instructions on this
   example.
2. Open GNU Radio Companion and open `classifier_test.grc`
3. Open the **Inference Block** and select the PLAN file created in Step 1.
4. Ensure that the settings of the **Inference Block** match the PLAN file.
5. Execute the GRC flowgraph

Note that the examples utilize a SoapyAIRT source block. However, these examples can be
tailored to any type of SDR by simply swapping the source block with the appropriate block
for your hardware.

### Creating a PLAN File
The following is an example of how to do this on the AIR-T which runs Ubuntu. By default,
AirStack includes the [airstack-examples](https://github.com/deepwavedigital/airstack-examples)
repository. You will need to have the [airstack conda environment](https://docs.deepwavedigital.com/Tutorials/6_conda.html)
activated. If you are using an older version of AirStack, are not up-to-date with the
latest version of airstack-examples, or not using an AIR-T, you will want to clone this
repository prior to the first step.

* Go to the `airstack-examples/inference` directory.
  * To use the most up-to-date version clone the `airstack-examples` repo:
    ```
    git clone https://github.com/deepwavedigital/airstack-examples.git
    cd airstack-examples/inference/
    ```
  * If using the version included with AirStack on the AIR-T:
    ```
    cd /opt/deepwave/AIR-T_QuickStart_Apps/inference/
    ```

* If on the AIR-T, activate the airstack conda environment:
  ```
  conda activate airstack
  ```

* Create the PLAN from the provided ONNX file. Here we will use the ONNX file
   associated with the PyTorch example model (default):
  ```
  python3 onnx2plan.py -b
  ```
  This process can take a few minutes. When finished, you will see an output as
  follows:
  ```
  Benchmark Result:
  Samples Processed : 33,554,432
  Processing Time   : 37.975 msec
  Throughput        : 883.589 MSPS
  Data Rate         : 56.550 Gbit / sec
  
  ONNX File Name  : /home/deepwave/airstack-examples/inference/pytorch/avg_pow_net.onnx
  ONNX File Size  : 16777
  PLAN File Name : /home/deepwave/airstack-examples/inference/pytorch/avg_pow_net.plan
  PLAN File Size : 114725
  
  Network Parameters for inference on AIR-T:
  CPLX_SAMPLES_PER_INFER = 2048
  BATCH_SIZE <= 128
  ```
  In addition to the above `CPLX_SAMPLES_PER_INFER` and `BATCH_SIZE` variables,
  the `OUTPUT_LEN` will need to be set to 1 for this model.
* You will use the PLAN file in the Classifier Test above.

Copyright 2018-2021 Deepwave Digital Inc.
