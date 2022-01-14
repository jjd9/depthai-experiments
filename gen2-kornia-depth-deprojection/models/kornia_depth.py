#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter
from kornia.utils import create_meshgrid
from kornia.geometry.camera import unproject_points
import numpy as np

name = 'depth_to_3d'


class Model(nn.Module):
    def forward(self, image):
        """Compute a 3d point per pixel given its depth value and the camera intrinsics.
        Args:
            depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
            camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
            normalize_points: whether to normalise the pointcloud. This must be set to `True` when the depth is
            represented as the Euclidean ray length from the camera position.
        Return:
            tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.
        Example:
            >>> depth = torch.rand(1, 1, 4, 4)
            >>> K = torch.eye(3)[None]
            >>> depth_to_3d(depth, K).shape
            torch.Size([1, 3, 4, 4])
        """
        camera_matrix = torch.Tensor([[[454.2445,  -2.048, 320.5384], [1.9331, 455.2362, 237.1727], [-0.0022,   0.0013,   1.]]])
        normalize_points = False

        # convert the uint8 representation of the image to uint16
        depth = 256.0 * image[:,:,:,1::2] + image[:,:,:,::2]

        # create base coordinates grid
        _, _, height, width = depth.shape
        points_2d: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
        points_2d = points_2d.to(depth.device).to(depth.dtype)

        # depth should come in Bx1xHxW
        points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

        # project pixels to camera frame
        camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
        points_3d: torch.Tensor = unproject_points(
            points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points
        )  # BxHxWx3

        return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW

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
