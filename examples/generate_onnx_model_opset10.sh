#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

python3 -m tf2onnx.convert --graphdef models/frozen_inference_graph.pb --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0 --output models/model_opset_10.onnx --fold_const --opset 10 --verbose

# Unfortunately, using opset 11, the inputs to the combinedNMS would result in having dynamic shapes during ONNX parsing time
# Currently, the batchedNMS plugin only supports  inputs whose shapes are deterministic during ONNX parsing time, because it is implemented based on the IPluginV2Ext template
# The ultimate solution is upgrade the batchedNMS plugin to use the IPluginV2DynamicExt template
# python -m tf2onnx.convert --graphdef models/frozen_inference_graph.pb --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0 --output models/model_opset_11.onnx --fold_const --opset 11 --verbose