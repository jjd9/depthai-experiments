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


# def points_to_3d(depth_image, image_points, inverse_depth_intrinsic):
#     x_idx = image_points[:,[0]].astype(int)
#     y_idx = image_points[:,[1]].astype(int)
#     pixel_coords = np.hstack((x_idx, y_idx, np.ones((image_points.shape[0], 1)))).T
#     cam_coords = np.dot(inverse_depth_intrinsic, pixel_coords) * \
#                         depth_image[y_idx, x_idx].ravel().astype(float)
#     return cam_coords.T

def points_to_3d(depth_image, image_points, inverse_depth_intrinsic):
    x_idx = image_points[:,[0]].astype(int)
    y_idx = image_points[:,[1]].astype(int)
    pixel_coords = np.hstack((x_idx, y_idx, np.ones((image_points.shape[0], 1)))).T
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
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

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
    iterationsCount=500        # number of Ransac iterations.
    reprojectionError=.5    # maximum allowed distance to consider it an inlier.
    confidence=0.999          # RANSAC successful confidence.
    flags=cv2.SOLVEPNP_ITERATIVE

    # # # Use ICP only
    # _, rotation, translation = best_fit_transform(old_3d_points, new_3d_points)
    # rvec, _ = cv2.Rodrigues(rotation)

    # TODO: Test this
    # Use ICP with known correspondences to get initial guess 
    _, rotation_init, translation_init = best_fit_transform(old_3d_points, new_3d_points)
    rvec_init, _ = cv2.Rodrigues(rotation_init)

    # Ransac with initial guess for robustness
    distCoeffs = np.zeros((4,1), dtype=float)
    useExtrinsicGuess=True
    _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, distCoeffs, rvec_init, translation_init,
                    useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                    None, flags )
    rotation, _ = cv2.Rodrigues(rvec)

    # TODO: Test this
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
        # fast_detector = cv2.FastFeatureDetector_create()
        # fast_detector.setThreshold(50)
        # fast_detector.setNonmaxSuppression(True)
        orb = cv2.ORB_create()

        max_num_features = 200
        max_feature_depth = 3000 # mm
        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(res["width"], res["height"]))
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                cur_img_frames[i] = np.array(image.getFrame())

            # We can only estimate odometry with we have both the current and previous frames
            if not (any([frame is None for frame in cur_img_frames]) or any([frame is None for frame in prev_img_frames])):
                # Visual odometry Algo goes here
                depth_frame = cur_img_frames[0]

                # TODO: Investigate other features
                # find and draw the keypoints
                new_feature_points = cv2.goodFeaturesToTrack(cur_img_frames[1], max_num_features, 0.01,10)
                # kpts = fast_detector.detect(cur_img_frames[1], None)
                # kpts = orb.detect(cur_img_frames[1], None)
                # new_feature_points = np.array([k.pt for k in kpts], dtype=np.float32).reshape((len(kpts), 1, 2))
                termcrit = (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 30, 0.01)
                window_size = (11,11)
                # window_size = (21,21)
                cur_feature_points, status, err = cv2.calcOpticalFlowPyrLK(prev_img_frames[1], cur_img_frames[1], prev_feature_points, new_feature_points, None, None, window_size, 3, termcrit, 0, 0.001)
                # delete unmatched features, features outside the image boundaries, and features that are too far away                
                remove_idx = []
                for i in range(status.shape[0]):
                    pt = cur_feature_points[i].ravel().astype(int)                    
                    if status[i] == 0 or pt[0] < 0 or pt[1] < 0 or pt[0] >= res["width"] or pt[1] >= res["height"] or depth_frame[pt[1], pt[0]] == 0 or depth_frame[pt[1], pt[0]] > max_feature_depth:
                        status[i] = 0
                        remove_idx.append(i)
                cur_feature_points = np.delete(cur_feature_points, remove_idx, axis=0)
                prev_feature_points = np.delete(prev_feature_points, remove_idx, axis=0)
                if prev_points_3d is not None:
                    prev_points_3d = np.delete(prev_points_3d, remove_idx, axis=0)
                n_feature_points = cur_feature_points.shape[0]

                # convert points to 3d                
                cur_points_3d = points_to_3d(depth_frame, cur_feature_points.squeeze(axis=1), inverse_rectified_right_intrinsic)

                if prev_points_3d is not None and n_feature_points >= 4:
                    # apply 3d point tracking and get change in 3d position and orientation
                    # import open3d as o3d
                    # prev_pcl = o3d.geometry.PointCloud()
                    # prev_pcl.points = o3d.utility.Vector3dVector(prev_points_3d)
                    # prev_pcl.colors = o3d.utility.Vector3dVector([(0,0,255) for _ in prev_points_3d])
                    # cur_pcl = o3d.geometry.PointCloud()
                    # cur_pcl.points = o3d.utility.Vector3dVector(cur_points_3d)
                    # cur_pcl.colors = o3d.utility.Vector3dVector([(255,0,0) for _ in cur_points_3d])

                    translation, rotation, odom_ok = point_3d_tracking(prev_feature_points, cur_feature_points, prev_points_3d, cur_points_3d, rectified_right_intrinsic)
                    # transform = np.eye(4)
                    # transform[:3,3] = translation
                    # transform[:3,:3] = rotation
                    # aligned_pcl = o3d.geometry.PointCloud()
                    # aligned_pcl.points = o3d.utility.Vector3dVector(prev_points_3d)
                    # aligned_pcl.colors = o3d.utility.Vector3dVector([(0,255,0) for _ in prev_points_3d])
                    # aligned_pcl.transform(transform)

                    # o3d.visualization.draw_geometries([prev_pcl, cur_pcl, aligned_pcl])
                    # exit()

                    # integration change
                    pose.translate(translation)
                    pose.rotate(rotation)

                    # Visualize features
                    feature_img = cv2.cvtColor(cur_img_frames[1].copy(),cv2.COLOR_GRAY2RGB)
                    for pt in cur_feature_points:
                        x,y = pt.ravel().astype(int)
                        cv2.circle(feature_img,(x,y),3,(0,255,0),-1)
                    
                    # # Visualize matches
                    # prev_kpts = [cv2.KeyPoint(*pt.ravel(),1) for pt in prev_feature_points] 
                    # cur_kpts = [cv2.KeyPoint(*pt.ravel(),1) for pt in cur_feature_points] 
                    # matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(cur_kpts))]
                    # feature_img = cv2.drawMatches(prev_img_frames[1],prev_kpts,cur_img_frames[1],cur_kpts,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))

                    # format of text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    thickness = 2
                    color = (0, 0, 255) # red
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

                # update these book keeping variables
                # To avoid running out of tracked feature points, we add all the valid new features points (not the cur_feature_points)
                remove_idx = []
                for i in range(new_feature_points.shape[0]):
                    pt = new_feature_points[i].ravel().astype(int)                    
                    if pt[0] < 0 or pt[1] < 0 or pt[0] >= res["width"] or pt[1] >= res["height"] or depth_frame[pt[1], pt[0]] == 0 or depth_frame[pt[1], pt[0]] > max_feature_depth:
                        remove_idx.append(i)
                new_feature_points = np.delete(new_feature_points, remove_idx, axis=0)
                # convert points to 3d
                new_points_3d = points_to_3d(depth_frame, new_feature_points.squeeze(axis=1), inverse_rectified_right_intrinsic)

                prev_feature_points = new_feature_points.copy()
                prev_points_3d = new_points_3d.copy()

            elif not any([frame is None for frame in cur_img_frames]):
                prev_feature_points = cv2.goodFeaturesToTrack(cur_img_frames[1], max_num_features, 0.01,10)

            # Update book keeping variables
            prev_img_frames[0] = cur_img_frames[0]
            prev_img_frames[1] = cur_img_frames[1]
            cur_img_frames[0] = None
            cur_img_frames[1] = None
            
            if cv2.waitKey(1) == "q":
                break

            # # convert from camera to world coordinates
            # cur_points_3d = cur_points_3d[:, [2,0,1]]
            # cur_points_3d[:,2] *= -1.0
