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

def point_3d_tracking(old_image_points, new_image_points, old_3d_points, new_3d_points, camera_intrinsics):

    _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, None)
    rotation, _ = cv2.Rodrigues(rvec)

    odom_ok = True
    return translation, rotation, odom_ok


# TODO: Figure out big jumps
# TODO: 6DOF Kalman filter

if __name__ == "__main__":
    print("Initialize pipeline")
    pipeline, streams = create_odom_pipeline()    

    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device(pipeline) as device:
        # get the camera calibration info
        calibData = device.readCalibration()
        right_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, res["width"], res["height"]))
        right_rotation = np.array(calibData.getStereoRightRectificationRotation())
        right_homography = np.matmul(np.matmul(right_intrinsic, right_rotation), np.linalg.inv(right_intrinsic))
        inverse_rectified_right_intrinsic = np.matmul(np.linalg.inv(right_intrinsic), np.linalg.inv(right_homography))
        rectified_right_intrinsic = np.linalg.inv(inverse_rectified_right_intrinsic)

        # setup cv window(s)
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
        t_bufer = []

        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        max_feature_depth = 3000 # mm
        min_feature_depth = 300 # mm
        max_matches = 100 # keep the top N matches
        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(res["width"], res["height"]))
        iter = 0
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                cur_img_frames[i] = np.array(image.getFrame())
            depth_frame = cur_img_frames[0]
            # cur_img_frames[1] = cv2.equalizeHist(cur_img_frames[1])
            # We can only estimate odometry with we have both the current and previous frames
            if not (any([frame is None for frame in cur_img_frames]) or any([frame is None for frame in prev_img_frames])):
                # Visual odometry Algo goes here

                kpts, cur_descriptors = orb.detectAndCompute(cur_img_frames[1], None)
                cur_feature_points = np.array([k.pt for k in kpts], dtype=np.float32).reshape((len(kpts), 1, 2))

                # To avoid running out of tracked feature points, we add all the valid new features points (not the cur_feature_points)
                remove_idx = []
                for i in range(cur_feature_points.shape[0]):
                    pt = cur_feature_points[i].ravel().astype(int)                    
                    if pt[0] < 0 or pt[1] < 0 or pt[0] >= res["width"] or pt[1] >= res["height"] or depth_frame[pt[1], pt[0]] <= min_feature_depth or depth_frame[pt[1], pt[0]] > max_feature_depth:
                        remove_idx.append(i)
                cur_feature_points = np.delete(cur_feature_points, remove_idx, axis=0)
                cur_descriptors = np.delete(cur_descriptors, remove_idx, axis=0)

                # Match descriptors.
                matches = bf.match(prev_descriptors, cur_descriptors)
                if len(matches) > 20:
                    # Sort them in the order of their distance.
                    matches = sorted(matches, key = lambda x:x.distance)[:max_matches]

                    # delete unmatched features, features outside the image boundaries, and features that are too far away
                    old_idx = []
                    new_idx = []
                    for match in matches:
                        old_idx.append(match.queryIdx)
                        new_idx.append(match.trainIdx)
                    cur_feature_points = cur_feature_points[new_idx]
                    cur_descriptors = cur_descriptors[new_idx]
                    prev_feature_points = prev_feature_points[old_idx]
                    prev_descriptors = prev_descriptors[old_idx]
                    prev_points_3d = prev_points_3d[old_idx]

                    # convert points to 3d                
                    cur_points_3d = points_to_3d(depth_frame, cur_feature_points.reshape(-1,2), inverse_rectified_right_intrinsic)
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
                    # transform[:3,3] = translation.reshape(3,)
                    # transform[:3,:3] = rotation
                    # aligned_pcl = o3d.geometry.PointCloud()
                    # aligned_pcl.points = o3d.utility.Vector3dVector(prev_points_3d)
                    # aligned_pcl.colors = o3d.utility.Vector3dVector([(0,255,0) for _ in prev_points_3d])
                    # aligned_pcl.transform(transform)

                    # o3d.visualization.draw_geometries([prev_pcl, cur_pcl, aligned_pcl])
                    # exit()

                    # integration change
                    current_pose = np.eye(4)
                    current_pose[0:3, 0:3] = rotation
                    current_pose[0:3, 3] = translation.reshape(3,)
                    pose.update(current_pose)

                    # Visualize features
                    feature_img = cv2.cvtColor(cur_img_frames[1].copy(),cv2.COLOR_GRAY2RGB)
                    for pt in new_feature_points:
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

            # initialize points
            kpts, new_descriptors = orb.detectAndCompute(cur_img_frames[1], None)
            new_feature_points = np.array([k.pt for k in kpts], dtype=np.float32).reshape((len(kpts), 1, 2))
            

            # To avoid running out of tracked feature points, we add all the valid new features points (not the cur_feature_points)
            remove_idx = []
            for i in range(new_feature_points.shape[0]):
                pt = new_feature_points[i].ravel().astype(int)                    
                if pt[0] < 0 or pt[1] < 0 or pt[0] >= res["width"] or pt[1] >= res["height"] or depth_frame[pt[1], pt[0]] <= min_feature_depth or depth_frame[pt[1], pt[0]] > max_feature_depth:
                    remove_idx.append(i)
            new_feature_points = np.delete(new_feature_points, remove_idx, axis=0)
            new_descriptors = np.delete(new_descriptors, remove_idx, axis=0)

            # convert points to 3d
            new_points_3d = points_to_3d(depth_frame, new_feature_points.reshape(-1,2), inverse_rectified_right_intrinsic)

            if cur_points_3d is not None:
                new_points_3d = np.vstack((cur_points_3d, new_points_3d))
                new_feature_points = np.vstack((cur_feature_points, new_feature_points))
                new_descriptors = np.vstack((cur_descriptors, new_descriptors))

            # Update book keeping variables
            prev_points_3d = new_points_3d.copy()
            prev_feature_points = new_feature_points.copy()
            prev_descriptors = new_descriptors.copy()
            prev_img_frames[0] = cur_img_frames[0]
            prev_img_frames[1] = cur_img_frames[1]
            cur_img_frames[0] = None
            cur_img_frames[1] = None
            
            if cv2.waitKey(1) == "q":
                break

            # # convert from camera to world coordinates
            # cur_points_3d = cur_points_3d[:, [2,0,1]]
            # cur_points_3d[:,2] *= -1.0
