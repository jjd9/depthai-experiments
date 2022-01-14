#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occured when importing PCL projector: {e}")

############################################################################
# USER CONFIGURABLE PARAMETERS (also see configureDepthPostProcessing())

# parameters to speed up visualization
downsample_pcl = True  # downsample the pointcloud before operating on it and visualizing

# StereoDepth config options.
# whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels
LRcheckthresh = 5
confidenceThreshold = 200
min_depth = 400  # mm
max_depth = 20000  # mm
speckle_range = 60
############################################################################

# Select camera resolution (oak-d-lite only supports THE_480_P  and THE_400_P depth)
# res = {"height": 400, "width": 640,
#        "THE_P": dai.MonoCameraProperties.SensorResolution.THE_400_P}
res = {"height": 480, "width": 640,
       "THE_P": dai.MonoCameraProperties.SensorResolution.THE_480_P}
# res = {"height": 720, "width": 1080, "THE_P": dai.MonoCameraProperties.SensorResolution.THE_720_P}


def configureDepthPostProcessing(stereoDepthNode):
    """
    In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps.
    """
    stereoDepthNode.initialConfig.setConfidenceThreshold(confidenceThreshold)
    stereoDepthNode.initialConfig.setLeftRightCheckThreshold(LRcheckthresh)
    # stereoDepthNode.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)
    # stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
    config = stereoDepthNode.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = speckle_range
    # config.postProcessing.temporalFilter.enable = True
    # config.postProcessing.spatialFilter.enable = True
    # config.postProcessing.spatialFilter.holeFillingRadius = 2
    # config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = min_depth  # mm
    config.postProcessing.thresholdFilter.maxRange = max_depth  # mm
    config.postProcessing.decimationFilter.decimationFactor = 1
    config.censusTransform.enableMeanMode = True
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2
    stereoDepthNode.initialConfig.set(config)
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)
    stereoDepthNode.setRectifyEdgeFillColor(
        0)  # Black, to better see the cutout


def create_pcl_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
    
    # Define sources
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath("models/depth_to_3d_simplified_openvino_2021.4_3shave.blob")

    # Define Output
    pointsOut = pipeline.createXLinkOut()
    pointsOut.setStreamName("pcl")

    # Configure Camera Properties
    mono_camera_resolution = res["THE_P"]
    left.setResolution(mono_camera_resolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(mono_camera_resolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # This block is all post-processing to depth image
    configureDepthPostProcessing(stereo)

    # Linking device side outputs to host side
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(nn.input)
    nn.out.link(pointsOut.input)

    # Book-keeping
    stream = pointsOut.getStreamName()

    return pipeline, stream

if __name__ == "__main__":
    print("Initialize pipeline")
    pipeline, stream = create_pcl_pipeline()

    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device(pipeline) as device:
        # silence device warning spam
        device.setLogLevel(dai.LogLevel.ERR)

        pcl_converter = PointCloudVisualizer()

        # setup bookkeeping variables
        queue = device.getOutputQueue(stream, maxSize=8, blocking=False)

        # main stream loop
        while True:
            name = queue.getName()
            raw_pcl = queue.get()
            pcl_data = np.array(raw_pcl.getData()).view(np.float16).reshape(1, 3, 480, 640)
            pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
            pcl_data = pcl_data[pcl_data[:,2] > 0, :]
            if pcl_data is not None:
                pcl_converter.update_pcl(pcl_data, downsample=downsample_pcl)
            pcl_converter.visualize_pcd()
