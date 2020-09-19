#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import numpy as np
import argparse

import onnx_graphsurgeon as gs
import onnx
import onnx.shape_inference


def count_uint8(graph):
    count = len(
        [tensor for tensor in graph.tensors().values() if hasattr(tensor, "dtype") and tensor.dtype == np.uint8])

    return count


def fix_uint8_tensors(graph):
    for node in graph.nodes:
        for node_input_tensor in node.inputs:
            if hasattr(node_input_tensor, "dtype") and node_input_tensor.dtype == np.uint8:
                node_input_tensor.dtype = np.int32
        for node_output_tensor in node.outputs:
            if hasattr(node_output_tensor, "dtype") and node_output_tensor.dtype == np.uint8:
                node_output_tensor.dtype = np.int32

    return graph


def replace_combinedNMS(graph, top_k=1284, keep_top_k=100, num_classes=3, plugin_version="1"):
    """
    Although, in principle, the value of top_k, keep_top_k, num_classes should be able to be inferred from the graph.
    Due to the limitation of the ONNX parser, we currently are not able to get these values.
    """

    for node in graph.nodes:
        if node.name == "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/CombinedNonMaxSuppression":
            clip_boxes = node.attrs["clip_boxes"]

    tensor_map = graph.tensors()
    model_input_tensor = tensor_map["image_tensor:0"]

    input_boxes = tensor_map["Postprocessor/ExpandDims_1:0"]
    input_scores = tensor_map["Postprocessor/Slice:0"]
    output_boxes = tensor_map["detection_boxes:0"]
    output_scores = tensor_map["detection_scores:0"]
    output_boxes.name = "detection_boxes"
    output_scores.name = "detection_scores"

    batch_size = model_input_tensor.shape[0]

    iou_threshold = tensor_map[
        "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/iou_threshold:0"].values.item()
    score_threshold = tensor_map[
        "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/score_threshold:0"].values.item()

    output_classes = gs.Variable(name="detection_classes_nms:0", dtype=np.float32, shape=(batch_size, keep_top_k))
    output_num_detections = gs.Variable(name="num_detections_nms:0", dtype=np.int32, shape=(batch_size, 1))

    attributes_ordered_dict = {"shareLocation": True, "backgroundLabelId": -1, "numClasses": num_classes, "topK": top_k,
                               "keepTopK": keep_top_k, "scoreThreshold": score_threshold, "iouThreshold": iou_threshold,
                               "isNormalized": True, "clipBoxes": clip_boxes, "plugin_version": plugin_version}

    for node in graph.nodes:
        if node.name == "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/CombinedNonMaxSuppression":
            node.op = "BatchedNMS_TRT"
            node.attrs = attributes_ordered_dict
            node.inputs = [input_boxes, input_scores]
            node.outputs = [output_num_detections, output_boxes, output_scores, output_classes]

    for node in graph.nodes:
        if node.name == "add":
            for i, input_tensor in enumerate(node.inputs):
                if input_tensor.name == "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/CombinedNonMaxSuppression:2":
                    input_id = i
            node.inputs[input_id] = output_classes

        if node.name == "Postprocessor/Cast_4":
            for i, input_tensor in enumerate(node.inputs):
                if input_tensor.name == "Postprocessor/CombinedNonMaxSuppression/combined_non_max_suppression/CombinedNonMaxSuppression:3":
                    input_id = i
            node.inputs[input_id] = output_num_detections

    graph.nodes[-2].outputs[0].name = "detection_classes"
    graph.nodes[-1].outputs[0].name = "num_detections"
    tensor_map['detection_classes:0'].name = "detection_classes"
    tensor_map["num_detections:0"].name = "num_detections"

    graph.outputs[0].name = "detection_boxes"
    graph.outputs[1].name = "detection_classes"
    graph.outputs[2].name = "detection_scores"
    graph.outputs[3].name = "num_detections"

    return graph


def modify_onnx(onnx_model_filepath="vot_opset_10.onnx", modified_onnx_model_filepath="vot_opset_10_modified.onnx"):
    orig_model = onnx.load(onnx_model_filepath)

    inferred_model = onnx.shape_inference.infer_shapes(orig_model)
    graph = gs.import_onnx(inferred_model)

    if count_uint8(graph=graph) > 0:
        print("Fixing UINT8 issues...")
        graph = fix_uint8_tensors(graph=graph)

        if count_uint8(graph=graph) > 0:
            raise Exception("UINT8 issue has not been fixed!")
        else:
            print("UINT8 issue has been fixed!")

    print("Replacing CombinedNMS to BatchedNMS...")
    graph = replace_combinedNMS(graph=graph)

    onnx.save(gs.export_onnx(graph.cleanup()), modified_onnx_model_filepath)
    print("CombinedNMS has been replaced to BatchedNMS!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help='onnx model path', default="model_opset_10.onnx")
    parser.add_argument('--modified', help='modified onnx model path', default="model_opset_10_modified.onnx")
    args = parser.parse_args()
    modify_onnx(onnx_model_filepath=args.onnx, modified_onnx_model_filepath=args.modified)
