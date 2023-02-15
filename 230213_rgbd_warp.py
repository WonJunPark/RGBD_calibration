import os
import numpy as np
import pickle, cv2, math
import open3d as o3d

rgb_w = 640 # 1/2
rgb_h = 480 # 1/2

tof_w = 224
tof_h = 172

dir_path = "../r9_calib/calib_6_4_2/dump"
sync_flag = 0

## param setting
rgb_intrinsics = np.array([[247.12658568026868, 0.0, 320.1948487613381], [0.0, 248.74479768439699, 232.8119396067714], [0.0, 0.0, 1.0]])
rgb_distortion = np.array([[0.0017303043391902685], [0.27959053616644225], [-0.589875259035956], [0.4313951377470375]])

tof_intrinsics = np.array([[109.47020270319996, 0.0, 111.70782176371134], [0.0, 110.2185858319306, 87.04513638733886], [0.0, 0.0, 1.0]])
tof_distortion = np.array([[-0.12364235580552249], [1.227282417391097], [-3.6580415506785653], [3.9874616856627543]])

T_DR = np.load('../tof2rgb_cal.npy')

## data load
rgb_tof_list = []

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.png' in file:
            file_path = os.path.join(root, file)
            rgb_tof_list.append(file_path)
            
        if '.tofraw' in file:
            file_path = os.path.join(root, file)
            rgb_tof_list.append(file_path)              
            
rgb_tof_list.sort()
#2300 upper
count = 2300



def undistortion(img, intrinsic, distortion):
    dim1 = img.shape[:2][::-1]
        
    scaled_K = intrinsic * dim1[0] / dim1[0]
    
    scaled_K[2][2] = 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion, dim1, np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def visualization_scale(depth_image):
    w_margin = int((rgb_w - tof_w)/2)
    h_margin = int((rgb_h - tof_h)/2)
            
    up_scale_depth = cv2.copyMakeBorder(depth_image, h_margin, h_margin, w_margin, w_margin, cv2.BORDER_CONSTANT, value=[0,0,0])
    return up_scale_depth

def depth_to_rgb(depth_points, extrinsics, intrinsics):
    """
    Transforms 3D depth points to 2D RGB points
    Parameters:
    - depth_points: (N x 3) numpy array of 3D points
    - extrinsics: (4 x 4) numpy array representing the extrinsic matrix
    - intrinsics: (3 x 3) numpy array representing the intrinsic matrix
    Returns:
    - rgb_points: (N x 2) numpy array of 2D RGB points
    """
    num_points = depth_points.shape[0]
    
    # Transform 3D depth points to 3D camera space points
    cam_points = np.hstack((depth_points, np.ones((num_points, 1)))).transpose()
    cam_points = np.dot(extrinsics, cam_points)
    
    # Project 3D camera space points to 2D image plane
    rgb_points = np.dot(intrinsics, cam_points)
    rgb_points = rgb_points / rgb_points[2, :]
    rgb_points = rgb_points[0:2, :].transpose()
    
    return rgb_points

def visualization_3d(x,y,z):
        import open3d as o3d
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        # ax.scatter(data_y2, data_x2, data_z)

        # 축 라벨 지정
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # 플롯 출력
        plt.show()
    


while count < len(rgb_tof_list): 
    if '.png' in rgb_tof_list[count]:
        rgb_color = cv2.imread(rgb_tof_list[count], cv2.IMREAD_COLOR) # 640*480
        sync_flag = 1
        rgb_color_undis = undistortion(rgb_color, rgb_intrinsics, rgb_distortion)
        
    # cam calibration : https://foss4g.tistory.com/1665
    if sync_flag == 1 and '.tofraw' in rgb_tof_list[count]:
        sync_flag = 0
        
        data = np.reshape(np.fromfile(rgb_tof_list[count], np.float32), [-1, tof_w, 6]) # depth, ir_float, conf_float, noise_float, x, y
        
        # print(np.min(data_x), np.max(data_x))
        
        data_x = np.array(data[:,:,4])
        data_y = np.array(data[:,:,5])
        data_z = np.array(data[:,:,0])
        
        data_xyz = np.stack((data_x, data_y, data_z), axis=2) # h,w,z
        
        # PointCloud 객체 생성
        data_xyz2 = data_xyz.reshape(-1,3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data_xyz2)
        
        # PointCloud 객체를 시각화
        o3d.visualization.draw_geometries([point_cloud])
        
            
        T_RD = np.linalg.inv(T_DR)        
        
        # rgb_points = np.zeros((tof_h, tof_w, 3))
        # rgb_points_normal = np.zeros((tof_h, tof_w, 3))
        

        # tof_cam_origin = np.array([0,0,0])
        # rgb_cam = np.dot(T_DR, np.append(tof_cam_origin, 1))[:3]
        
        # tof_cam_origin2 = tof_cam_origin.reshape(1,3)
        # rgb_cam2 = rgb_cam.reshape(1,3)
        
        # two_cam = np.concatenate([tof_cam_origin2, rgb_cam2], axis=0)
        
        # all_point = np.concatenate([two_cam, data_xyz2], axis=0)
        
        # print(all_point.shape)
        
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(all_point)
        # o3d.visualization.draw_geometries([point_cloud])
    
        
        for i in range(tof_h):
            for j in range(tof_w):
                tof_point = data_xyz[i][j]
                tof_point_h = np.append(tof_point, 1)
                rgb_points[i][j] = np.dot(T_RD, tof_point_h)[:3]
                
                
        # PointCloud 객체 생성
        rgb_points2 = rgb_points.reshape(-1,3)
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(rgb_points2)
        # PointCloud 객체를 시각화
        o3d.visualization.draw_geometries([point_cloud2])
        
        # visualization_3d(rgb_points[0], rgb_points[1], rgb_points[2])
  
    count += 1