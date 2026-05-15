"""Convert FP16 ONNX model to FP32 for devices without shader-f16 support."""

import os
import onnx
from onnx import TensorProto, numpy_helper

FP16 = TensorProto.FLOAT16
FP32 = TensorProto.FLOAT

def convert(input_path, output_path):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)

    n_inits = 0
    for init in model.graph.initializer:
        if init.data_type == FP16:
            arr = numpy_helper.to_array(init).astype("float32")
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            n_inits += 1

    n_types = 0
    for vi in list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    ):
        if vi.HasField("type") and vi.type.HasField("tensor_type"):
            if vi.type.tensor_type.elem_type == FP16:
                vi.type.tensor_type.elem_type = FP32
                n_types += 1

    n_casts = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FP16:
                    attr.i = FP32
                    n_casts += 1

    print(f"Converted {n_inits} initializers, {n_types} value types, {n_casts} cast nodes")

    print(f"Saving {output_path}...")
    onnx.save_model(model, output_path, save_as_external_data=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! {size_mb:.1f} MB")


if __name__ == "__main__":
    convert(
        "training/output/merged-onnx/model_fp16.onnx",
        "training/output/merged-onnx/model_fp32.onnx",
    )
