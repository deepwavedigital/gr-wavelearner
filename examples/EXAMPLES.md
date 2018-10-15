# Examples for GR-WAVELEARNER
This folder contains a simple test for the gr-wavelearner package.

## Files:
- classifier_test.dat: binary data containing complex data as interleaved int16 data types
- classifier_test.plan: TensorRT PLAN file of CNN for performing inference
- classifier_test.grc: GNU Radio Companion flowgraph to read from classifier_test.day, perform
  inference using classifier_test.plan, and display the results to the terminal
- classifier_test.py: Executable Python file produced when executing the classifier_test.grc
  flowgraph
  
## Instructions:
1. Open GNU Radio Companion
2. Choose *File* -> *Open*
3. Open the *classifier_test.grc* file
4. Double click the Point the *File Source Block* and choose the *classifier_test.dat* file
5  Open the *Inference Block* and point the PLAN File to *classifier_test.plan*.
6. Execute the code.

## Expected Output
See the screenshots in README.md

Copyright 2018 Deepwave Digital Inc.