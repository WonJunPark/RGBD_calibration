import os
import numpy as np
import pickle, cv2, math

rgb_w = 640
rgb_h = 480

tof_w = 224
tof_h = 172

dir_path = "../r9_calib/calib_6_4_2/dump"
rows = 4
cols = 6

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
objectPointsArray = []
imgPointsArray = []

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


found = 0
count = 0

while count < len(rgb_tof_list): 
    if '.png' in rgb_tof_list[count]:
        img = cv2.imread(rgb_tof_list[count], cv2.IMREAD_COLOR) # 640*480
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 640*480
        isSucces, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
        
        if isSucces:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            cv2.drawChessboardCorners(img, (rows, cols), corners, isSucces)
            found += 1
        
        cv2.imshow('Kalibrasyon', img)
        cv2.waitKey(0)
        
    count += 1
            
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('calibrationdata.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

count = 2300

while count < len(rgb_tof_list): 
    if '.png' in rgb_tof_list[count]:
        img = cv2.imread(rgb_tof_list[count], cv2.IMREAD_COLOR) # 640*480
        h, w = img.shape[:2]
        
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        
        cv2.imshow('Duzeltilmis Goruntu', undistortedImg)
        cv2.waitKey(0)
            
    count += 1
    
cv2.destroyAllWindows()
