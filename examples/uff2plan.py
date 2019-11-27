#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# 
# Copyright 2018 Deepwave Digital Inc.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
 
# This script must be executed on the platform that will be used for inference, i.e., the AIR-T.

import tensorrt as trt

# Input parameters specific to the trained model
uff_file_name = 'saved_model.uff'  # Name of uff file that defines the trained model
input_node_name = 'input/IteratorGetNext'  # Input node (best to name it with tf.name.scope)
input_node_dims = (1, 1, 4096)  # Input dimensions to trained model

# Input parameter for inference
batch_size = 128  # Batch size to optimize to. This should be used for inference
workspace_size = 1073741824  # 1 GB, for example
use_fp16 = True  # Do you want to use float16 type
output_file_name = 'saved_model.plan'  # Name of output file

# Make the plan file
builder = trt.Builder(trt.Logger(trt.Logger.INFO))
network = builder.create_network()

parser = trt.UffParser()
parser.register_input(input_node_name, input_node_dims)
parser.parse(uff_file_name, network)

builder.max_batch_size = batch_size
builder.maxworkspace_size = workspace_size
builder.fp16_mode = use_fp16

engine = builder.build_cuda_engine(network)

with open(output_file_name, 'wb') as f:
     f.write(engine.serialize())
