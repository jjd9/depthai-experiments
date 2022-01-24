#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pose import Pose
from bucket import Bucket
from feature import FeaturePoint, FeatureSet
import pickle

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
    config.postProcessing.temporalFilter.enable = True
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
    rectifiedLeftOut = pipeline.createXLinkOut()
    rectifiedLeftOut.setStreamName("rectified_left")

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
    stereo.rectifiedLeft.link(rectifiedLeftOut.input)

    # Book-keeping
    streams = [depthOut.getStreamName(), rectifiedRightOut.getStreamName(),
               rectifiedLeftOut.getStreamName()]

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
    x_idx = image_points[:, [0]].astype(int)
    y_idx = image_points[:, [1]].astype(int)
    pixel_coords = np.hstack(
        (x_idx, y_idx, np.ones((image_points.shape[0], 1)))).T
    cam_coords = np.dot(inverse_depth_intrinsic, pixel_coords) * \
        depth_image[y_idx, x_idx].ravel().astype(float)
    return cam_coords.T


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def point_3d_tracking(old_image_points, new_image_points, old_3d_points, new_3d_points, camera_intrinsics):
    # Rotation(R) estimation using Nister's Five Points Algorithm
    # recovering the pose and the essential cv::matrix
    # E, mask = cv2.findEssentialMat(old_image_points, new_image_points, camera_intrinsics, cv2.RANSAC, 0.999, 1.0)
    # _, R_mono, _, _ = cv2.recoverPose(E, old_image_points, new_image_points, camera_intrinsics, None, None, mask)
    # Translation (t) estimation by use solvePnPRansac
    iterationsCount = 500        # number of Ransac iterations.
    # maximum allowed distance to consider it an inlier.
    reprojectionError = .5
    confidence = 0.999          # RANSAC successful confidence.
    flags = cv2.SOLVEPNP_ITERATIVE

    # Use ICP with known correspondences to get initial guess
    _, rotation_init, translation_init = best_fit_transform(
        old_3d_points, new_3d_points)
    rvec_init, _ = cv2.Rodrigues(rotation_init)

    # Ransac with initial guess for robustness
    distCoeffs = np.zeros((4, 1), dtype=float)
    useExtrinsicGuess = True
    _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, distCoeffs, rvec_init, translation_init,
                                                 useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                                                 None, flags)
    rotation, _ = cv2.Rodrigues(rvec)

    # # Ransac with initial guess for robustness
    # useExtrinsicGuess=False
    # distCoeffs = np.zeros((4,1), dtype=float)
    # _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, distCoeffs, None, None,
    #                 useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
    #                 None, flags )
    # rotation, _ = cv2.Rodrigues(rvec)

    odom_ok = True
    translation[0] = (translation[0] / 10).round() * 10
    return translation, rotation, odom_ok


fast_detector = cv2.FastFeatureDetector_create()
fast_detector.setThreshold(20)
fast_detector.setNonmaxSuppression(True)

def appendFeaturePoints(image, features):
    kpts = fast_detector.detect(image, None)
    features.points += [list(k.pt) for k in kpts]
    features.ages += [0] * len(kpts)


def bucketFeatures(image, current_features, bucket_size, features_per_bucket):
    buckets_nums_height = int(image.shape[0]/bucket_size)
    buckets_nums_width = int(image.shape[1]/bucket_size)
    buckets_number = buckets_nums_height * buckets_nums_width

    bucket_list = [Bucket(features_per_bucket) for _ in range(buckets_number)]
    for i in range(current_features.size()):
        buckets_nums_height_idx = min(
            int(current_features.points[i][1]/bucket_size), buckets_nums_height - 1)
        buckets_nums_width_idx = min(
            int(current_features.points[i][0]/bucket_size), buckets_nums_width - 1)
        buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx
        bucket_list[buckets_idx].add_feature(
            current_features.points[i], current_features.ages[i])
    # clear current set of features
    current_features.clear()
    # repopulate with bucketed features
    for i in range(buckets_number):
        bucket_list[i].get_features(current_features)


def deleteUnmatchedFeatures(points0, points1, points2, points3, new_points, status0, status1, status2, status3, feature_ages):
    matched_cond = (status0 != 0) & (status1 != 0) & (status2 != 0) & (status3 != 0) & np.all(points0 >= 0, axis=2) & np.all(
        points1 >= 0, axis=2) & np.all(points2 >= 0, axis=2) & np.all(points3 >= 0, axis=2) & np.all(new_points >= 0, axis=2)
    points0 = points0[matched_cond.ravel()]
    points1 = points1[matched_cond.ravel()]
    points2 = points2[matched_cond.ravel()]
    points3 = points3[matched_cond.ravel()]
    new_points = new_points[matched_cond.ravel()]
    feature_ages = np.array(feature_ages)[matched_cond.ravel()]
    feature_ages += 1
    feature_ages.tolist()
    return points0, points1, points2, points3, new_points, feature_ages

def circularMatchFeatures(prev_left_frame, prev_right_frame, cur_left_frame, cur_right_frame, odom_points):
    termcrit = (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 30, 0.01)
    window_size = (21, 21)

    prev_right_points = np.array(
        odom_points.points, dtype=np.float32).reshape(-1, 1, 2)

    prev_left_points, status0, _ = cv2.calcOpticalFlowPyrLK(
        prev_right_frame, prev_left_frame, prev_right_points, None, None, None, window_size, 3, termcrit, 0, 0.001)
    cur_left_points, status1, _ = cv2.calcOpticalFlowPyrLK(
        prev_left_frame,  cur_left_frame,  prev_left_points, None, None, None, window_size, 3, termcrit, 0, 0.001)
    cur_right_points, status2, _ = cv2.calcOpticalFlowPyrLK(
        cur_left_frame,   cur_right_frame, cur_left_points,  None, None, None, window_size, 3, termcrit, 0, 0.001)
    matched_right_points, status3, _ = cv2.calcOpticalFlowPyrLK(
        cur_right_frame,  prev_right_frame, cur_right_points, None, None, None, window_size, 3, termcrit, 0, 0.001)
    prev_right_points, prev_left_points, cur_right_points, cur_left_points, matched_right_points, odom_points.ages = deleteUnmatchedFeatures(prev_right_points, prev_left_points, cur_right_points, cur_left_points,
                                 matched_right_points, status0, status1, status2, status3, odom_points.ages)

    return prev_left_points, prev_right_points, cur_left_points, cur_right_points, matched_right_points, odom_points


def validateMatches(prev_points, new_points, threshold):
    # norms = np.linalg.norm(prev_points - new_points.reshape(1,-1, 2), axis=-1, ord=1)
    norms = np.max(np.abs(prev_points - new_points), axis=2)
    return (norms <= threshold)

def matchFeatures(prev_img_frames, cur_img_frames, odom_points):
    _, prev_right_frame, prev_left_frame = prev_img_frames
    _, cur_right_frame, cur_left_frame = cur_img_frames

    if odom_points.size() < 2000:
        appendFeaturePoints(prev_right_frame, odom_points)

    img_height = cur_right_frame.shape[0]
    bucket_size = img_height / 10
    features_per_bucket = 1
    bucketFeatures(prev_right_frame, odom_points,
                   bucket_size, features_per_bucket)

    prev_left_points, prev_right_points, cur_left_points, cur_right_points, matched_right_points, odom_points = circularMatchFeatures(
        prev_left_frame, prev_right_frame, cur_left_frame, cur_right_frame, odom_points)

    status = validateMatches(prev_right_points, matched_right_points, threshold=5)

    num_points = status.sum()
    print("Matched {} feature points".format(num_points))
    print("Oldest point {}".format(odom_points.ages.max()))
    prev_left_points = prev_left_points[status]
    cur_left_points = cur_left_points[status]
    prev_right_points = prev_right_points[status]
    cur_right_points = cur_right_points[status]
    odom_points.points = cur_right_points.tolist()
    odom_points.ages = odom_points.ages.tolist()
    return num_points, prev_left_points, prev_right_points, cur_left_points, cur_right_points, odom_points

# Adapted from:
# https://github.com/ZhenghaoFei/visual_odom/blob/master/src/visualOdometry.cpp


if __name__ == "__main__":
    print("Initialize pipeline")
    pipeline, streams = create_odom_pipeline()

    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device(pipeline) as device:
        # get the camera calibration info
        calibData = device.readCalibration()
        right_intrinsic = np.array(calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT, res["width"], res["height"]))
        left_intrinsic = np.array(calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.LEFT, res["width"], res["height"]))

        left_rotation = np.array(
            calibData.getStereoLeftRectificationRotation())
        right_rotation = np.array(
            calibData.getStereoRightRectificationRotation())
        right_homography = np.matmul(
            np.matmul(right_intrinsic, right_rotation), np.linalg.inv(right_intrinsic))
        left_homography = np.matmul(
            np.matmul(right_intrinsic, left_rotation), np.linalg.inv(left_intrinsic))
        inverse_rectified_right_intrinsic = np.matmul(
            np.linalg.inv(right_intrinsic), np.linalg.inv(right_homography))
        rectified_right_intrinsic = np.linalg.inv(
            inverse_rectified_right_intrinsic)
        inverse_rectified_left_intrinsic = np.matmul(
            np.linalg.inv(left_intrinsic), np.linalg.inv(left_homography))
        rectified_left_intrinsic = np.linalg.inv(
            inverse_rectified_left_intrinsic)

        rl_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))[:3,:]
        R = rl_extrinsics[:3, :3]
        t = rl_extrinsics[:3, 3] * 10.0 # times 10 to go from cm to mm
        proj_mat_right = rectified_right_intrinsic @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
        proj_mat_left = rectified_left_intrinsic @ cv2.hconcat([R, t]) # R, T from left to right
        proj_mat_right = proj_mat_right.astype(float)
        proj_mat_left = proj_mat_left.astype(float)

        # setup bookkeeping variables
        odom_points = FeatureSet()
        # depth frame, recitified right frame, recitified left frame
        prev_img_frames = [None, None, None]
        # depth frame, recitified right frame, recitified left frame
        cur_img_frames = [None, None, None]

        queue_list = [device.getOutputQueue(
            stream, maxSize=8, blocking=False) for stream in streams]

        pose = Pose()

        max_num_features = 2000
        max_feature_depth = 4000  # mm

        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(
            res["width"], res["height"]))
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                cur_img_frames[i] = np.array(image.getFrame())

            # We can only estimate odometry with we have both the current and previous frames
            if all([frame is not None for frame in cur_img_frames + prev_img_frames]):
                # depth, right, left
                prev_depth_frame, prev_right_frame, prev_left_frame = prev_img_frames
                cur_depth_frame, cur_right_frame, cur_left_frame = cur_img_frames
                num_points, prev_left_points, prev_right_points, cur_left_points, cur_right_points, odom_points = matchFeatures(
                    prev_img_frames, cur_img_frames, odom_points)

                if num_points > 0:
                    # convert points to 3d using left and right images
                    prev_right_points = prev_right_points.reshape(-1,2).T
                    prev_left_points = prev_left_points.reshape(-1,2).T
                    cur_right_points = cur_right_points.reshape(-1,2).T
                    cur_left_points = cur_left_points.reshape(-1,2).T
                    prev_points_4d = cv2.triangulatePoints(proj_mat_right, proj_mat_left, prev_right_points, prev_left_points)
                    prev_points_4d_normalized = prev_points_4d/prev_points_4d[3,:]
                    prev_points_3d = prev_points_4d_normalized[:3,:].T.astype(np.float64)
                    cur_points_4d = cv2.triangulatePoints(proj_mat_right, proj_mat_left, cur_right_points, cur_left_points)
                    cur_points_4d_normalized = cur_points_4d/cur_points_4d[3,:]
                    cur_points_3d = cur_points_4d_normalized[:3,:].T.astype(np.float64)

                    prev_right_points = prev_right_points.T.reshape(-1,1,2)
                    cur_right_points = cur_right_points.T.reshape(-1,1,2)

                    cond = (cur_points_3d[:, 2] < max_feature_depth) & (prev_points_3d[:, 2] < max_feature_depth)
                    prev_points_3d = prev_points_3d[cond]
                    cur_points_3d = cur_points_3d[cond]
                    prev_right_points = prev_right_points[cond]
                    cur_right_points = cur_right_points[cond]
                    num_3d_points = cond.sum()
                    if num_3d_points >= 20:
                        print("Num 3d points: ", num_3d_points)
                        translation, rotation, odom_ok = point_3d_tracking(
                            prev_right_points, cur_right_points, prev_points_3d, cur_points_3d, rectified_right_intrinsic)

                        # integration change
                        pose.rotate(rotation)
                        pose.translate(translation)

                        # Visualize features
                        feature_img = cv2.cvtColor(
                            cur_img_frames[1].copy(), cv2.COLOR_GRAY2RGB)
                        for pt in cur_right_points:
                            x, y = pt.ravel().astype(int)
                            cv2.circle(feature_img, (x, y), 3, (0, 255, 0), -1)

                        # # Visualize matches
                        # prev_kpts = [cv2.KeyPoint(*pt.ravel(), 1)
                        #             for pt in prev_right_points]
                        # cur_kpts = [cv2.KeyPoint(*pt.ravel(), 1)
                        #             for pt in cur_right_points]
                        # matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(cur_kpts))]
                        # feature_img = cv2.drawMatches(prev_img_frames[1], prev_kpts, cur_img_frames[1], cur_kpts,
                        #                             matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))

                        # format of text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        thickness = 2
                        color = (0, 0, 255)  # red
                        # position of text
                        position_org = (50, 50)
                        orientation_org = (50, 150)
                        pos_text = pose.getPositionString()
                        rpy_text = pose.getRPYString()
                        feature_img = cv2.putText(feature_img, 'Camera Position: ' + pos_text, position_org, font,
                                                fontScale, color, thickness, cv2.LINE_AA)
                        feature_img = cv2.putText(feature_img, 'Camera Orientation: ' + rpy_text, orientation_org, font,
                                                fontScale, color, thickness, cv2.LINE_AA)

                        cv2.imshow("Current Features", feature_img)

            # Update book keeping variables
            prev_img_frames = [x for x in cur_img_frames]
            cur_img_frames = [None] * len(cur_img_frames)

            if cv2.waitKey(1) == "q":
                break

            # # convert from camera to world coordinates
            # cur_points_3d = cur_points_3d[:, [2,0,1]]
            # cur_points_3d[:,2] *= -1.0
