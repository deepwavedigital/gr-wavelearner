# Examples for GR-WAVELEARNER
This folder contains a simple test for the gr-wavelearner package.

## Files:
- classifier_test.dat: binary data containing complex data as interleaved int16 data types
- classifier_test.plan: TensorRT PLAN file of CNN for performing inference
- classifier_test.grc: GNU Radio Companion flowgraph to read from classifier_test.day, perform
  inference using classifier_test.plan, and display the results to the terminal
- classifier_test.py: Executable Python file produced when executing the classifier_test.grc
  flowgraph


Copyright 2018 Deepwave Digital Inc.