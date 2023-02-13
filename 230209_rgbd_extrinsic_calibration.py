import os
import numpy as np
import pickle, cv2, math

rgb_w = 640 # 1/2
rgb_h = 480 # 1/2

tof_w = 224
tof_h = 172
dir_path = "../r9_calib/calib_6_4_2/dump"
# dir_intrinsic = "../r9_calib/calib_6_4_2/tofraw_intrinsic.txt"

sync_flag = 0
txt_data = []

rgb_tof_list = []
pattern_size = np.array([4,6])
gridsize = 0.06 # 6cm
grid = np.zeros((pattern_size[0]*pattern_size[1], 3))

K_camera = np.zeros((3, 3))
distortion_camera = np.zeros((4))
P_camera = np.zeros((3, 4))

K_camera_d = np.zeros((3, 3))
distortion_camera_d = np.zeros((4))
P_camera_d = np.zeros((3, 4))

cnt = 0

for r in range(pattern_size[0]):
    for c in range(pattern_size[1]):
        grid[cnt][0] = c*gridsize
        grid[cnt][1] = r*gridsize
        cnt += 1
        
# with open(dir_intrinsic,"r") as f:
#     lines = f.readlines()

# RGB
K_camera[0,0] = 247.12658568026868 #494.5440
K_camera[1,1] = 248.74479768439699 #494.7848
K_camera[0,2] = 320.1948487613381 #643.5081
K_camera[1,2] = 232.8119396067714 #461.8701

# fisheye distortion
distortion_camera[0] = 0.0017303043391902685 #0.015834
distortion_camera[1] = 0.27959053616644225 #0.092649
distortion_camera[2] = -0.589875259035956 #-0.075424
distortion_camera[3] = 0.4313951377470375 #0.020610

# ToF
# intrinsic_sample = lines[0].split(' ')
# K_camera_d[0,0] = intrinsic_sample[0] # fx
# K_camera_d[1,1] = intrinsic_sample[1] # fy
# K_camera_d[0,2] = intrinsic_sample[2] # cx
# K_camera_d[1,2] = intrinsic_sample[3] # cy
K_camera_d = np.array([[109.47020270319996, 0.0, 111.70782176371134], [0.0, 110.2185858319306, 87.04513638733886], [0.0, 0.0, 1.0]])

# r0, r1, t0, t1, r2
# https://amroamroamro.github.io/mexopencv/matlab/cv.solvePnP.html
# distortion_sample = lines[1][:-1].split(' ')
# distortion_camera_d[0] = distortion_sample[0]
# distortion_camera_d[1] = distortion_sample[1]
# distortion_camera_d[2] = distortion_sample[2]
# distortion_camera_d[3] = distortion_sample[3]
# distortion_camera_d[4] = distortion_sample[4]
distortion_camera_d=np.array([[-0.12364235580552249], [1.227282417391097], [-3.6580415506785653], [3.9874616856627543]])

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.png' in file:
            file_path = os.path.join(root, file)
            rgb_tof_list.append(file_path)
            
        if '.tofraw' in file:
            file_path = os.path.join(root, file)
            rgb_tof_list.append(file_path)              
            
rgb_tof_list.sort()

# print(len(rgb_tof_list))
# print(rgb_tof_list[0][len(dir_path)+1:len(dir_path)+1+8])

#2300 upper
count = 2300
dim2=None
dim3=None

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


while count < len(rgb_tof_list): 
    if '.png' in rgb_tof_list[count]:
        rgb_color = cv2.imread(rgb_tof_list[count], cv2.IMREAD_COLOR) # 640*480
        sync_flag = 1
        rgb_color_undis = undistortion(rgb_color, K_camera, distortion_camera)
        
    # cam calibration : https://foss4g.tistory.com/1665
    if sync_flag == 1 and '.tofraw' in rgb_tof_list[count]:
        sync_flag = 0
        
        data = np.reshape(np.fromfile(rgb_tof_list[count], np.float32), [-1, tof_w, 6]) # depth, ir_float, conf_float, noise_float, x, y
        depth_image = np.clip(np.array(data[:,:,1]).reshape([tof_h, tof_w]), a_min=0, a_max=255).astype(np.uint8)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        
        depth_color_undis = undistortion(depth_image, K_camera_d, distortion_camera_d) 
           
                
        rgb_gray = cv2.cvtColor(rgb_color_undis, cv2.COLOR_BGR2GRAY)
        depth_gray = cv2.cvtColor(depth_color_undis, cv2.COLOR_BGR2GRAY)
        
        rgb_ret, rgb_corners = cv2.findChessboardCorners(rgb_gray, (6,4), None)
        depth_ret, depth_corners = cv2.findChessboardCorners(depth_gray, (6,4), None)
        
        # depth_color_undis_up_scale = visualization_scale(depth_color_undis)
        # add_img = np.vstack((rgb_color_undis, depth_color_undis_up_scale))
        # cv2.imshow("img",add_img)
        # cv2.waitKey(0)   
                                      
        
        if (rgb_ret == True) & (depth_ret == True):
            rgb_imgpoints = [] # 2d points in image plane.
            depth_imgpoints = []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                  
            rgb_corners2 = cv2.cornerSubPix(rgb_gray, rgb_corners,(11,11),(-1,-1), criteria)
            rgb_imgpoints.append(rgb_corners2)
            
            depth_corners2 = cv2.cornerSubPix(depth_gray, depth_corners,(11,11),(-1,-1), criteria)
            depth_imgpoints.append(depth_corners2)
            
            # Draw and display the corners
            rgb_img = cv2.drawChessboardCorners(rgb_color_undis, (6,4), rgb_corners2, rgb_ret)
            depth_img = cv2.drawChessboardCorners(depth_color_undis, (6,4), depth_corners2, depth_ret)
                                                           
            
            _, rvec, tvec = cv2.solvePnP(grid, rgb_corners2, np.eye(3), np.zeros((1,5)))
            _, rvec_i, tvec_i = cv2.solvePnP(grid, depth_corners2, K_camera_d, distortion_camera_d)
            
            
            
            R_RC = np.zeros((3,3))
            R_DC = np.zeros((3,3))

            T_RC = np.zeros((4,4)) 
            T_DC = np.zeros((4,4)) 
            
            cv2.Rodrigues(rvec, R_RC)
            cv2.Rodrigues(rvec_i, R_DC)
            
            
            T_RC[0:3,0:3] = R_RC
            T_DC[0:3,0:3] = R_DC
            
            T_RC[0:3,3] = tvec.T[0]
            T_DC[0:3,3] = tvec_i.T[0]
            
            T_RC[3][3] = 1.0
            T_DC[3][3] = 1.0
            
            
            T_DR = T_DC.dot(np.linalg.inv(T_RC))
                        
            # https://betterprogramming.pub/introduction-to-point-cloud-processing-dbda9b167534
            
            print("[%f, %f, %f, %f,"%(T_DR[0][0], T_DR[0][1], T_DR[0][2], T_DR[0][3]))
            print("%f, %f, %f, %f,"%(T_DR[1][0], T_DR[1][1], T_DR[1][2], T_DR[1][3]))
            print("%f, %f, %f, %f,"%(T_DR[2][0], T_DR[2][1], T_DR[2][2], T_DR[2][3]))
            print("%f, %f, %f, %f]"%(T_DR[3][0], T_DR[3][1], T_DR[3][2], T_DR[3][3]))
            
            np.save("../tof2rgb_cal.npy",T_DR)
                      
            
            depth_img = visualization_scale(depth_img)
            add_img = np.vstack((rgb_img, depth_img))
            cv2.imshow("img",add_img)
            cv2.waitKey(0)                
        
    # cv2.waitKey(0)             
    count += 1
 
