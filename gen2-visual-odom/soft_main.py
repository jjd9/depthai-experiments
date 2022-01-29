#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pose import Pose
from bucket import Bucket
from feature import FeaturePoint, FeatureSet
import time

# Select camera resolution (oak-d-lite only supports THE_480_P  and THE_400_P depth)
# res = {"height": 400, "width": 640,
#        "THE_P": dai.MonoCameraProperties.SensorResolution.THE_400_P}
res = {"height": 480, "width": 640,
       "THE_P": dai.MonoCameraProperties.SensorResolution.THE_480_P}
# res = {"height": 720, "width": 1080, "THE_P": dai.MonoCameraProperties.SensorResolution.THE_720_P}

max_num_features = 1000
max_feature_depth = 4000  # mm
min_feature_depth = 100  # mm
num_vertical_buckets = 30
features_per_bucket = 20

def create_odom_pipeline():
    pipeline = dai.Pipeline()

    # Define sources
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout

    # Define outputs
    rectifiedRightOut = pipeline.createXLinkOut()
    rectifiedRightOut.setStreamName("rectified_right")
    rectifiedLeftOut = pipeline.createXLinkOut()
    rectifiedLeftOut.setStreamName("rectified_left")

    mono_camera_resolution = res["THE_P"]
    left.setResolution(mono_camera_resolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(mono_camera_resolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Linking device side outputs to host side
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.rectifiedRight.link(rectifiedRightOut.input)
    stereo.rectifiedLeft.link(rectifiedLeftOut.input)

    # Book-keeping
    streams = [rectifiedRightOut.getStreamName(),
               rectifiedLeftOut.getStreamName()]

    return pipeline, streams


def point_3d_tracking(old_image_points, new_image_points, old_3d_points, new_3d_points, camera_intrinsics):
    # Translation (t) estimation by use solvePnPRansac
    iterationsCount = 500        # number of Ransac iterations.
    # maximum allowed distance to consider it an inlier.
    reprojectionError = 0.5
    confidence = 0.999          # RANSAC successful confidence.
    flags = cv2.SOLVEPNP_ITERATIVE

    # Ransac without initial guess
    _, rvec, translation, _ = cv2.solvePnPRansac(old_3d_points, new_image_points, camera_intrinsics, None, None, None,
                                                 False, iterationsCount, reprojectionError, confidence,
                                                 None, flags)
    rotation, _ = cv2.Rodrigues(rvec)

    odom_ok = True
    resolution = 1.0 # mm
    translation = np.round(translation / resolution) * resolution
    return translation, rotation, odom_ok


fast_detector = cv2.FastFeatureDetector_create()
fast_detector.setThreshold(30)
fast_detector.setNonmaxSuppression(True)
# orb = cv2.ORB_create(nfeatures=1000)


def appendFeaturePoints(image, features):
    # kpts, _ = orb.detectAndCompute(image, None)
    kpts = fast_detector.detect(image, None)
    features.points += [list(k.pt) for k in kpts]
    features.ages += [0] * len(kpts)
    # corners = cv2.goodFeaturesToTrack(image, 1000, 0.01, 10).reshape(-1,2)
    # features.points += corners.tolist()
    # features.ages += [0] * corners.shape[0]
    return features


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
    return current_features


def deleteUnmatchedFeatures(points0, points1, points2, points3, new_points, status0, status1, status2, status3, feature_ages):
    matched_cond = (status0 != 0) & (status1 != 0) & (status2 != 0) & (status3 != 0) & np.all(points0 >= 0, axis=2) & np.all(
        points1 >= 0, axis=2) & np.all(points2 >= 0, axis=2) & np.all(points3 >= 0, axis=2) & np.all(new_points >= 0, axis=2)\
        & np.all(points0[:, :, [0]] < res['width'], axis=1) & np.all(points1[:, :, [0]] < res['width'], axis=2)\
        & np.all(points2[:, :, [0]] < res['width'], axis=2) & np.all(new_points[:, :, [0]] < res['width'], axis=2)\
        & np.all(points0[:, :, [1]] < res['height'], axis=2) & np.all(points1[:, :, [1]] < res['height'], axis=2)\
        & np.all(points2[:, :, [1]] < res['height'], axis=2) & np.all(new_points[:, :, [1]] < res['height'], axis=2)
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
    norms = np.max(np.abs(prev_points - new_points), axis=2)
    return (norms <= threshold)


def matchFeatures(prev_img_frames, cur_img_frames, odom_points):
    prev_right_frame, prev_left_frame = prev_img_frames
    cur_right_frame, cur_left_frame = cur_img_frames

    if odom_points.size() < max_num_features:
        odom_points = appendFeaturePoints(prev_right_frame, odom_points)
    img_height = cur_right_frame.shape[0]
    bucket_size = img_height / num_vertical_buckets
    odom_points = bucketFeatures(prev_right_frame, odom_points,
                                 bucket_size, features_per_bucket)

    prev_left_points, prev_right_points, cur_left_points, cur_right_points, matched_right_points, odom_points = circularMatchFeatures(
        prev_left_frame, prev_right_frame, cur_left_frame, cur_right_frame, odom_points)

    status = validateMatches(
        prev_right_points, matched_right_points, threshold=1)

    num_points = status.sum()
    if num_points > 0:
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

        rl_extrinsics = np.array(calibData.getCameraExtrinsics(
            dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT))[:3, :]
        R = rl_extrinsics[:3, :3]
        t = rl_extrinsics[:3, 3] * 10.0  # times 10 to go from cm to mm
        # project points in the world frame (i.e. right optical camera frame)
        # Cam1 is the origin
        proj_mat_right = rectified_right_intrinsic @ cv2.hconcat(
            [np.eye(3), np.zeros((3, 1))])
        # R, T from left to right
        proj_mat_left = rectified_left_intrinsic @ cv2.hconcat([R, t])
        proj_mat_right = proj_mat_right.astype(np.float32)
        proj_mat_left = proj_mat_left.astype(np.float32)

        # setup bookkeeping variables
        odom_points = FeatureSet()
        # recitified right frame, recitified left frame
        prev_img_frames = [None, None]
        # recitified right frame, recitified left frame
        cur_img_frames = [None, None]
        # keep track of timestamp
        prev_img_stamp = None
        cur_img_stamp = None

        queue_list = [device.getOutputQueue(
            stream, maxSize=8, blocking=False) for stream in streams]

        pose = Pose()

        last_update_time = time.time()
        fps_vals = []

        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(
            res["width"], res["height"]))
        first = True
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                cur_img_frames[i] = np.array(image.getFrame())
                if i == 0:
                    cur_img_stamp = image.getTimestamp()
            now = time.time()
            fps = int(1.0 / (now - last_update_time))
            last_update_time = now
            fps_vals.append(fps)
            if len(fps_vals) >= 10:
                fps = np.mean(fps_vals)
                fps_vals.pop(0)

            # We can only estimate odometry with we have both the current and previous frames
            if all([frame is not None for frame in cur_img_frames + prev_img_frames]):
                # right, left
                prev_right_frame, prev_left_frame = prev_img_frames
                cur_right_frame, cur_left_frame = cur_img_frames
                num_points, prev_left_points, prev_right_points, cur_left_points, cur_right_points, odom_points = matchFeatures(
                    prev_img_frames, cur_img_frames, odom_points)

                if num_points > 10:
                    # convert points to 3d using left and right image triangulation
                    prev_4d = cv2.triangulatePoints(
                        proj_mat_right, proj_mat_left, prev_right_points.T, prev_left_points.T)
                    prev_points_3d = (prev_4d[:3, :] / prev_4d[3, :]).T
                    cur_4d = cv2.triangulatePoints(
                        proj_mat_right, proj_mat_left, cur_right_points.T, cur_left_points.T)
                    cur_points_3d = (cur_4d[:3, :] / cur_4d[3, :]).T

                    cond = (cur_points_3d[:, 2] < max_feature_depth) & (prev_points_3d[:, 2] < max_feature_depth) & (
                        cur_points_3d[:, 2] >= min_feature_depth) & (prev_points_3d[:, 2] >= min_feature_depth)
                    prev_points_3d = prev_points_3d[cond]
                    cur_points_3d = cur_points_3d[cond]
                    prev_right_points = prev_right_points[cond]
                    cur_right_points = cur_right_points[cond]
                    num_3d_points = cond.sum()
                    if num_3d_points >= 10:
                        print("Num 3d points: ", num_3d_points)
                        translation, rotation, odom_ok = point_3d_tracking(
                            prev_right_points, cur_right_points, prev_points_3d, cur_points_3d, rectified_right_intrinsic)
                        time_delta = cur_img_stamp.total_seconds() - prev_img_stamp.total_seconds()
                        # reject obviously bogus odometry spikes
                        if np.abs(translation).max() < 100:
                            # integration change
                            current_pose = np.eye(4)
                            current_pose[0:3, 0:3] = rotation
                            current_pose[0:3, 3] = translation.reshape(3,)
                            pose.update(current_pose)

                        # Visualize features
                        feature_img = cv2.cvtColor(
                            cur_img_frames[0].copy(), cv2.COLOR_GRAY2RGB)
                        for old_pt, pt in zip(prev_right_points, cur_right_points):
                            x, y = pt.ravel().astype(int)
                            old_x, old_y = old_pt.ravel().astype(int)
                            cv2.circle(feature_img, (x, y), 3, (0, 255, 0), -1)
                            cv2.circle(feature_img, (old_x, old_y), 3, (0, 0, 255), -1)

                        # # Visualize matches
                        # prev_kpts = [cv2.KeyPoint(*pt.ravel(), 1)
                        #             for pt in prev_right_points]
                        # cur_kpts = [cv2.KeyPoint(*pt.ravel(), 1)
                        #             for pt in cur_right_points]
                        # matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(cur_kpts))]
                        # feature_img = cv2.drawMatches(prev_img_frames[0], prev_kpts, cur_img_frames[0], cur_kpts,
                        #                             matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))

                        # format of text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        thickness = 2
                        color = (0, 0, 255)  # red
                        # position of text
                        position_org = (50, 50)
                        orientation_org = (50, 150)
                        fps_org = (50, 250)
                        pos_text = pose.getPositionString()
                        rpy_text = pose.getRPYString()
                        fps_text = "FPS: {}".format(fps)
                        feature_img = cv2.putText(feature_img, 'Camera Position: ' + pos_text, position_org, font,
                                                  fontScale, color, thickness, cv2.LINE_AA)
                        feature_img = cv2.putText(feature_img, 'Camera Orientation: ' + rpy_text, orientation_org, font,
                                                  fontScale, color, thickness, cv2.LINE_AA)
                        feature_img = cv2.putText(feature_img, fps_text, fps_org, font,
                                                  fontScale, color, thickness, cv2.LINE_AA)
                        cv2.imshow("Current Features", feature_img)

            # Update book keeping variables
            prev_img_frames = [x for x in cur_img_frames]
            prev_img_stamp = cur_img_stamp
            cur_img_frames = [None] * len(cur_img_frames)

            if cv2.waitKey(1) == "q":
                break
