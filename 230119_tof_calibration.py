import os
import numpy as np
import pickle, cv2, math, glob
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
# print(cv2.__version__)

rgb_w = 640
rgb_h = 480

tof_w = 224
tof_h = 172

dir_path = "../r9_calib/calib_6_4_2/dump"
CHECKERBOARD = (4,6)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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

count = 2300

while count < len(rgb_tof_list): 
    if '.tofraw' in rgb_tof_list[count]:
        data = np.reshape(np.fromfile(rgb_tof_list[count], np.float32), [-1, tof_w, 6])
        gray = np.clip(np.array(data[:,:,1]).reshape([tof_h, tof_w]), a_min=0, a_max=255).astype(np.uint8)
        
        if _img_shape == None:
            _img_shape = gray.shape[:2]
            
        
        # Chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # Image points (after refinin them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    count += 1    
        
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

DIM=_img_shape[::-1]
balance=1
dim2=None
dim3=None


# tof 이미지 10장
# DIM=(224, 172)
# K=np.array([[109.47020270319996, 0.0, 111.70782176371134], [0.0, 110.2185858319306, 87.04513638733886], [0.0, 0.0, 1.0]])
# D=np.array([[-0.12364235580552249], [1.227282417391097], [-3.6580415506785653], [3.9874616856627543]])

# tof 이미지 1000장 up
# K=np.array([[117.61401869116764, 0.0, 112.74055269745989], [0.0, 118.21700778604432, 87.41611251270672], [0.0, 0.0, 1.0]])
# D=np.array([[0.01909606636195195], [-0.17209271720695166], [0.700786829244202], [-0.5236568334480048]])

# sensor data
# 110.70763397216796875 110.70763397216796875 113.57366943359375 86.85997772216796875 224 172 # fx fy cx cy
# -0.20094765722751617432 0.035207204520702362061 -0.0014518202515318989754 -0.0030622307676821947098 0 # distortion

# K=np.array([[110.70763397216796875, 0.0, 113.57366943359375], [0.0, 110.70763397216796875, 86.85997772216796875], [0.0, 0.0, 1.0]])
# D=np.array([[-0.12364235580552249], [1.227282417391097], [-3.6580415506785653], [3.9874616856627543]])

K=np.array([[109.47020270319996, 0.0, 111.70782176371134], [0.0, 110.2185858319306, 87.04513638733886], [0.0, 0.0, 1.0]])
D=np.array([[-0.12364235580552249], [1.227282417391097], [-3.6580415506785653], [3.9874616856627543]])

count = 0
while count < len(rgb_tof_list): 
    
    if '.tofraw' in rgb_tof_list[count]:
        data = np.reshape(np.fromfile(rgb_tof_list[count], np.float32), [-1, tof_w, 6])
        gray = np.clip(np.array(data[:,:,1]).reshape([tof_h, tof_w]), a_min=0, a_max=255).astype(np.uint8)
            
        w_margin = int((rgb_w - tof_w)/2)                  
        h_margin = int((rgb_h - tof_h)/2)
               
        gray = cv2.copyMakeBorder(gray, h_margin, h_margin, w_margin, w_margin, cv2.BORDER_CONSTANT, value=[0,0,0])
    
        dim1 = gray.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        
        assert dim1[0]/dim1[1] == dim1[0]/dim1[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
            # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        data = {'dim1': dim1, 
                'dim2':dim2,
                'dim3': dim3,
                'K': np.asarray(K).tolist(),  
                'D':np.asarray(D).tolist(),
                'new_K':np.asarray(new_K).tolist(),
                'scaled_K':np.asarray(scaled_K).tolist(),
                'balance':balance}
        
        cv2.imshow("undistorted", undistorted_img)
        img2 = cv2.imread("2.png")
        # cv2.imshow("none undistorted", img2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
    count += 1

# import json
# with open("fisheye_calibration_data.json", "w") as f:
#     json.dump(data, f)



