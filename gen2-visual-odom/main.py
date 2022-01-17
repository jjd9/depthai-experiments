#!/usr/bin/env python3

from fnmatch import translate
import cv2
import depthai as dai
import numpy as np
from pose import Pose

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
    # Set StereoDepth config options.
    # whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
    align_on_host = False
    lrcheck = True  # Better handling for occlusions
    extended = False  # Closer-in minimum depth, disparity range is doubled
    subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels
    LRcheckthresh = 5
    confidenceThreshold = 200
    min_depth = 400  # mm
    max_depth = 20000  # mm
    speckle_range = 60
    # Apply config options to StereoDepth.
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


def create_odom_pipeline():
    pipeline = dai.Pipeline()

    # Define sources
    camRgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    # Define outputs
    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth")
    rectifiedRightOut = pipeline.createXLinkOut()
    rectifiedRightOut.setStreamName("rectified_right")

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
    stereo.depth.link(depthOut.input)
    stereo.rectifiedRight.link(rectifiedRightOut.input)

    # Book-keeping
    streams = [depthOut.getStreamName(), rectifiedRightOut.getStreamName()]

    return pipeline, streams


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def points_to_3d(depth_image, image_points, inverse_depth_intrinsic):
    x_idx = image_points[:,[0]].astype(int)
    y_idx = image_points[:,[1]].astype(int)
    pixel_coords = np.hstack((x_idx, y_idx, np.ones((image_points.shape[0], 1)))).T
    cam_coords = np.dot(inverse_depth_intrinsic, pixel_coords) * \
                        depth_image[y_idx, x_idx].ravel().astype(float)
    return cam_coords.T


def point_3d_tracking(old_image_points, new_image_points, old_3d_points, new_3d_points, camera_intrinsics):
    # Rotation(R) estimation using Nister's Five Points Algorithm
    # recovering the pose and the essential cv::matrix
    # E, mask = cv2.findEssentialMat(old_image_points, new_image_points, camera_intrinsics, cv2.RANSAC, 0.999, 1.0)
    # _, R_mono, _, _ = cv2.recoverPose(E, old_image_points, new_image_points, camera_intrinsics, None, None, mask)
    # Translation (t) estimation by use solvePnPRansac
    iterationsCount=500        # number of Ransac iterations.
    reprojectionError=.5    # maximum allowed distance to consider it an inlier.
    confidence=0.999          # RANSAC successful confidence.
    useExtrinsicGuess=False
    flags=cv2.SOLVEPNP_ITERATIVE

    distCoeffs = np.zeros((4,1), dtype=float)
    _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, distCoeffs, None, None,
                    useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                    None, flags )
    rotation, _ = cv2.Rodrigues(rvec)
    odom_ok = True
    return translation, rotation, odom_ok


if __name__ == "__main__":
    print("Initialize pipeline")
    pipeline, streams = create_odom_pipeline()    

    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device(pipeline) as device:
        # get the camera calibration info
        calibData = device.readCalibration()
        right_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, res["width"], res["height"]))
        # This is techinically needed because the depth image is made from the rectified right camera image, not the right camera image
        # (although in practice, I did not see a big difference)
        right_rotation = np.array(calibData.getStereoRightRectificationRotation())
        right_homography = np.matmul(np.matmul(right_intrinsic, right_rotation), np.linalg.inv(right_intrinsic))
        inverse_rectified_right_intrinsic = np.matmul(np.linalg.inv(right_intrinsic), np.linalg.inv(right_homography))
        rectified_right_intrinsic = np.linalg.inv(inverse_rectified_right_intrinsic)

        # setup cv window(s)
        # cv2.namedWindow("Feature Tracking")
        cv2.namedWindow("Current Features")

        # setup bookkeeping variables
        prev_img_frames = [None, None] # depth frame, recitified right frame
        cur_img_frames = [None, None]  # depth frame, recitified right frame
        queue_list = [device.getOutputQueue(
            stream, maxSize=8, blocking=False) for stream in streams]
        pose = Pose()
        prev_feature_points = None
        prev_points_3d = None
        cur_feature_points = None    
        cur_points_3d = None

        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(res["width"], res["height"]))
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                cur_img_frames[i] = np.array(image.getFrame())

            # We can only estimate odometry with we have both the current and previous frames
            if not (any([frame is None for frame in cur_img_frames]) or any([frame is None for frame in prev_img_frames])):
                # TODO: Visual odometry Algo goes here

                # TODO: How to avoid extinction of points? Need to add points back in

                # find and draw the keypoints
                cur_feature_points = cv2.goodFeaturesToTrack(cur_img_frames[1], 100, 0.01,10)
                termcrit = (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 30, 0.01)
                window_size = (21,21)
                _, status, err = cv2.calcOpticalFlowPyrLK(prev_img_frames[1], cur_img_frames[1], prev_feature_points, cur_feature_points, None, None, window_size, 3, termcrit, 0, 0.001)
                # delete unmatched features
                remove_idx = []
                for i in range(status.shape[0]):
                    pt = cur_feature_points[i].ravel()
                    if status[i] == 0 or pt[0] < 0 or pt[1] < 0 or pt[0] >= res["width"] or pt[1] >= res["height"]:
                        status[i] = 0
                        remove_idx.append(i)
                cur_feature_points = np.delete(cur_feature_points, remove_idx, axis=0)
                n_feature_points = cur_feature_points.shape[0]

                feature_img = cv2.cvtColor(cur_img_frames[1].copy(),cv2.COLOR_GRAY2RGB)
                for pt in cur_feature_points:
                    x,y = pt.ravel().astype(int)
                    cv2.circle(feature_img,(x,y),3,(0,255,0),-1)
                cv2.imshow("Current Features", feature_img)

                # convert points to 3d
                cur_points_3d = points_to_3d(cur_img_frames[0], cur_feature_points.squeeze(axis=1), inverse_rectified_right_intrinsic)
                # convert from camera to world coordinates
                cur_points_3d = cur_points_3d[:, [2,0,1]]
                cur_points_3d[:,2] *= -1.0


                if prev_points_3d is not None and n_feature_points >= 4:
                    # apply 3d point tracking and get change in 3d position and orientation
                    translation, rotation, odom_ok = point_3d_tracking(prev_feature_points, cur_feature_points, prev_points_3d, cur_points_3d, rectified_right_intrinsic)

                    # integration change
                    if odom_ok:
                        pose.translate(translation)
                        pose.rotate(rotation)

                # update these book keeping variables
                prev_feature_points = cur_feature_points.copy()
                prev_points_3d = cur_points_3d.copy()

            elif not any([frame is None for frame in cur_img_frames]):
                prev_feature_points = cv2.goodFeaturesToTrack(cur_img_frames[1], 30, 0.01,10)

            # Update book keeping variables
            prev_img_frames[0] = cur_img_frames[0]
            prev_img_frames[1] = cur_img_frames[1]
            cur_img_frames[0] = None
            cur_img_frames[1] = None
            
            if cv2.waitKey(1) == "q":
                break
