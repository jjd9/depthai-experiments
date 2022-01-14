#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter
from kornia.geometry.depth import depth_to_3d
import numpy as np

name = 'depth_to_3d'


class Model(nn.Module):
    def forward(self, image):

        # TODO: Remove hardcoded camera intrinsic matrix
        camera_matrix = torch.Tensor([[[454.2445,  -2.048, 320.5384], [1.9331, 455.2362, 237.1727], [-0.0022,   0.0013,   1.]]])

        # convert the uint8 representation of the image to uint16 (this is needed because the converter only allows
        # U8 and FP16 input types)
        depth = 256.0 * image[:,:,:,1::2] + image[:,:,:,::2]
        return depth_to_3d(depth, camera_matrix, normalize_points=False)


# # Simplest model for debugging
# class Model(nn.Module):
#     def forward(self, image: torch.Tensor):
#         return torch.div(image, 1.0)

# Define the expected input shape (dummy input)
# Note there are twice as many columns as in the actual image because the network will interpret the memory buffer input as as uint8
# even though it is actually uint16.
shape = (1, 1, 480, 2*640)
model = Model()
X = torch.ones(shape, dtype=torch.float16)

path = Path("out/")
path.mkdir(parents=True, exist_ok=True)
onnx_path = str(path / (name + '.onnx'))

print(f"Writing to {onnx_path}")
torch.onnx.export(
    model,
    X,
    onnx_path,
    opset_version=12,
    do_constant_folding=True,
)

onnx_simplified_path = str(path / (name + '_simplified.onnx'))

# Use onnx-simplifier to simplify the onnx model
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, onnx_simplified_path)

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_simplified_path,
    data_type="FP16",
    shaves=5,
    use_cache=False,
    output_dir="../models",
    compile_params=["-ip U8"],
    optimizer_params=[]
)
