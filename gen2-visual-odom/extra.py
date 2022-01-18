# apply 3d point tracking and get change in 3d position and orientation
# import open3d as o3d
# prev_pcl = o3d.geometry.PointCloud()
# prev_pcl.points = o3d.utility.Vector3dVector(prev_points_3d)
# prev_pcl.colors = o3d.utility.Vector3dVector([(0,0,255) for _ in prev_points_3d])
# cur_pcl = o3d.geometry.PointCloud()
# cur_pcl.points = o3d.utility.Vector3dVector(cur_points_3d)
# cur_pcl.colors = o3d.utility.Vector3dVector([(255,0,0) for _ in cur_points_3d])
# transform = np.eye(4)
# transform[:3,3] = translation
# transform[:3,:3] = rotation
# aligned_pcl = o3d.geometry.PointCloud()
# aligned_pcl.points = o3d.utility.Vector3dVector(prev_points_3d)
# aligned_pcl.colors = o3d.utility.Vector3dVector([(0,255,0) for _ in prev_points_3d])
# aligned_pcl.transform(transform)
# o3d.visualization.draw_geometries([prev_pcl, cur_pcl, aligned_pcl])
# exit()
