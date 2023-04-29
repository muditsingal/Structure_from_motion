"""
Author: Mudit Singal
UID: 119262689
Dir ID: msingal

Submission: Project 4

"""

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

start = time.time()
np.set_printoptions(threshold=sys.maxsize/2)

# np.random.seed(102)

# Function to scale the window size for properly displaying the images
def show_image_reshaped(img, title, width, height):
	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(title, width, height)
	cv2.imshow(title, img)
	return

# Function to convert image to grayscale, detect features and return key points and descriptors from image and specified feature detector
def get_features_in_img(img, detector):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected_keypts, descriptors = detector.detectAndCompute(gray_img, mask=None)
	return detected_keypts, descriptors, gray_img


# For 'iterations' number of iterations, we take 4 random points and find the number of inliers
def find_F_RANSAC(pts1, pts2, matches, threshold = 0.8, iterations = 200):

	pt1_arr = np.empty(shape=(len(matches), 3))
	pt2_arr = np.empty(shape=(len(matches), 3))
	i = 0

	# Finding point in images corresponding to the matching keypoints
	for match in matches:
		pt1_arr[i] = np.hstack([pts1[match.queryIdx].pt, 1])
		pt2_arr[i] = np.hstack([pts2[match.trainIdx].pt, 1])
		i += 1

	n_inliers_best = 0
	best_inliers = []
	
	best_inliers_1 = []
	best_inliers_2 = []
	inliers_1 = []
	inliers_2 = []
	# A_best = A
	
	for _ in range(iterations):
		n_inliers = 0
		A = np.ones([8,9])
		
		# Take 8 random points from the detected features list
		indices = np.random.randint(pt1_arr.shape[0], size=8)

		i=0
		for idx in indices:
			A[i,0] = pt1_arr[idx, 0]*pt2_arr[idx, 0]
			A[i,1] = pt1_arr[idx, 0]*pt2_arr[idx, 1]

			A[i,2] = pt1_arr[idx, 0]
			A[i,3] = pt1_arr[idx, 1]*pt2_arr[idx, 0]
			A[i,4] = pt1_arr[idx, 1]*pt2_arr[idx, 1]
			A[i,5] = pt1_arr[idx, 1]
			A[i,6] = pt2_arr[idx, 0]
			A[i,7] = pt2_arr[idx, 1]
			
			i += 1

		U, sig, Y_T = np.linalg.svd(A)
		# U, sig, Y_T = np.linalg.svd(A.T @ A)

		F = Y_T[-1, :].reshape([3,3])
		# F = rank_2_F(F)/

		for j in range(pt1_arr.shape[0]):
			# print(pt2_arr[j].T @ F @ pt1_arr[j])
			# if abs(pt2_arr[j].T @ F @ pt1_arr[j]) < threshold:
			if abs(pt2_arr[j].T @ F @ pt1_arr[j]) < threshold:
				n_inliers += 1
				inliers_1.append(pt1_arr[j, 0:2])
				inliers_2.append(pt2_arr[j, 0:2])

		if n_inliers > n_inliers_best:
			n_inliers_best = n_inliers
			F_best = F
			best_inliers_1 = inliers_1.copy()
			best_inliers_2 = inliers_2.copy()

		inliers_1.clear()
		inliers_2.clear()

	return F_best, np.array(best_inliers_1), np.array(best_inliers_2)

# Function to convert rank 3 matrix F to rank 2
def rank_2_F(F_og):
	U, sig, V_T = np.linalg.svd(F_og)
	sig[-1] = 0
	diag = np.diag(sig)
	F_new = U@diag@V_T
	return F_new

# Function to find epilines in rectified images and return images with drawn lines
def draw_eplines(img1, img2, lines, pts1, pts2):
	row, col = img1.shape
	img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

	for line, pt1, pt2 in zip(lines, pts1, pts2):
		color = (0,155,145)
		x0 = 0
		y0 = int(-line[2]/line[1])
		x1 = int(col)
		y1 = int(-(line[2] + line[0]*col)/line[1])
		img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
		img1 = cv2.circle(img1, tuple(np.int32(pt1)), 4, color, thickness=2)
		img2 = cv2.circle(img2, tuple(np.int32(pt2)), 4, color, thickness=2)

	return img1, img2

# Find the disparity map, checking full kernel, hence slightly
def find_disparity_match(gray1, gray2, kernel_size=11, max_disparity=60):
	pad = kernel_size//2
	kernel_half = kernel_size//2
	curr_disparity = 0
	disp_img = np.zeros([gray1.shape[0], gray1.shape[1]]).astype(np.uint8)
	for y in range(pad, gray1.shape[0]-pad):
		for x in range(pad, gray1.shape[1]-pad):
			best_ssd = 65534
			best_dis = 0

			for dis in range(0, max_disparity+1):
				# print("For disparity: ", dis)
				# print(gray1[y-pad:y+pad+1, x-pad:x+pad+1].shape)
				# print(gray2[y-pad:y+pad+1, x-pad-dis:x+pad-dis+1].shape)
				# print(x-pad-dis)
				if (x-pad-dis < 0) or (x+pad-dis+1 > gray1.shape[1]+1):
					continue

				disp_mat = gray1[y-pad:y+pad+1, x-pad:x+pad+1] - gray2[y-pad:y+pad+1, x-pad-dis:x+pad-dis+1]
				disp_mat = np.square(disp_mat)
				ssd = np.sum(disp_mat)
				if ssd < best_ssd:
					best_ssd = ssd
					best_dis = dis
					# print(best_ssd)

			disp_img[y,x] = int((best_dis*255/max_disparity))

	return disp_img

# Find the disparity map, checking individual kernel elements, hence slow
def find_disparity_match_slow(gray1, gray2, kernel_size=11, max_disparity=60):
	pad = kernel_size//2
	kernel_half = kernel_size//2
	curr_disparity = 0
	disp_img = np.zeros([gray1.shape[0], gray1.shape[1]])
	im1 = gray1.astype(int)
	im2 = gray2.astype(int)
	for y in range(pad, gray1.shape[0]-pad):
		print("Currently {}% of image parsed".format(int(((y-pad)/(gray1.shape[0]-pad))*100)), flush=True)
		for x in range(pad, gray1.shape[1]-pad):
			best_ssd = 65534
			best_dis = 0

			for dis in range(max_disparity):
				ssd_temp = 0
				ssd = 0
				# print("For disparity: ", dis)
				# print(gray1[y-pad:y+pad+1, x-pad:x+pad+1].shape)
				# print(gray2[y-pad:y+pad+1, x-pad-dis:x+pad-dis+1].shape)
				# print(x-pad-dis)
				for ii in range(-kernel_half, kernel_half):
					for jj in range(-kernel_half, kernel_half):
						# iteratively sum the sum of squared differences value for this block
						# left[] and right[] are arrays of uint8, so converting them to int saves
						# potential overflow
						ssd_temp = int(im1[y+ii, x+jj]) - int(im2[y+ii, (x+jj) - dis])  
						ssd += ssd_temp**2
				
				# if this value is smaller than the previous ssd at this block
				# then it's theoretically a closer match. Store this value against
				# this block..
				if ssd < best_ssd:
					best_ssd = ssd
					best_dis = dis

			disp_img[y,x] = int((best_dis/max_disparity) * 255)

	return disp_img


# Tried to vectorize the operations, not getting the desired results
def find_disparity_match_eff(gray1, gray2, kernel_size=11, max_disparity=60):
	pad = kernel_size//2
	disp_img = np.zeros([gray1.shape[0], gray1.shape[1]]).astype(np.uint8)
	disp_masks1 = np.zeros(shape=(gray1.shape[0], gray1.shape[1], kernel_size, kernel_size)).astype(np.int16)
	disp_masks2 = np.zeros(shape=(gray1.shape[0], gray1.shape[1], kernel_size, kernel_size)).astype(np.int16)

	for y in range(pad, gray1.shape[0]-pad):
		for x in range(pad, gray1.shape[1]-pad):
			disp_masks1[y,x] = gray1[y-pad:y+pad+1, x-pad:x+pad+1].astype(np.int16)
			disp_masks2[y,x] = gray2[y-pad:y+pad+1, x-pad:x+pad+1].astype(np.int16)
			# if (np.sum(disp_masks2[y,x])) == 0 and x < 1700:
			# 	print("Zeros at: x,y", x, y)
			# print("Mask is :", disp_masks1[y,x])
			# print("Mask is :", disp_masks1[y,x])

	# print(disp_masks1[5:-5,5:-5])

	disp_masks1 = np.repeat(np.array([disp_masks1]), max_disparity, axis=0)
	disp_masks2 = np.repeat(np.array([disp_masks2]), max_disparity, axis=0)
	# print(np.sum(np.square((disp_masks1[1, 200:400, 200:400] - disp_masks2[1, 200:400, 200:400])), axis=(2,3)))
	# print(disp_masks2[1, 200:400, 200:400])

	for dis in range(1, max_disparity):
		disp_masks2[dis] = np.roll(disp_masks2[dis], shift=-dis, axis=1)
		disp_masks2[dis,:,-dis:] = 0
		# print(disp_masks2[dis])

	ssd_mega_arr = disp_masks1 - disp_masks2
	ssd_mega_arr = np.square(ssd_mega_arr)
	# ssd_mega_arr = np.absolute(ssd_mega_arr)
	ssd_mega_arr = np.sum(ssd_mega_arr, axis=(3,4))
	ssd_mega_arr = np.absolute(ssd_mega_arr)
	ssd_mega_arr = np.int16((ssd_mega_arr/np.max(ssd_mega_arr))*255)
	# print("SSD final array after norm: \n", ssd_mega_arr)

	disp_img = (np.argmin(ssd_mega_arr, axis=0)*255/max_disparity).astype(np.uint8)


	# print("Middle ssd column data: ", ssd_mega_arr[0, 450, 50:60])
	# print("Middle ssd column data: ", ssd_mega_arr[0, 550, 50:60])
	# print("Middle ssd column data: ", ssd_mega_arr[0, 750, 50:60])
	# print("Middle ssd column data: ", ssd_mega_arr[0, 480, 50:60])
	# print("Middle ssd column data: ", ssd_mega_arr[0, 510, 50:60])

	# print(np.argmin(ssd_mega_arr, axis=0))

	return disp_img




curr_pwd = os.getcwd()
W = np.array([[0,1,0], [-1,0,0], [0,0,1]])

case_in = 1

while case_in <= 0 or case_in > 3:
	case_in = int(input("Please enter the scenario:\n1 for artroom\n2 for chess room\n3 for ladder room\n"))

# Reading the images and setting the calibration matrices K corresponding to each set
img_art_1 = cv2.imread(curr_pwd + '/artroom/im0.png')
img_art_2 = cv2.imread(curr_pwd + '/artroom/im1.png')
K_art = np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89], [0, 0, 1]])

img_chess_1 = cv2.imread(curr_pwd + '/chess/im0.png')
img_chess_2 = cv2.imread(curr_pwd + '/chess/im1.png')
K_chess = np.array([[1758.23, 0, 829.15],[0, 1758.23, 552.78], [0, 0, 1]])

img_ladder_1 = cv2.imread(curr_pwd + '/ladder/im0.png')
img_ladder_2 = cv2.imread(curr_pwd + '/ladder/im1.png')
K_ladder = np.array([[1734.16, 0, 333.49],[0, 1734.16, 958.05], [0, 0, 1]])

orb = cv2.ORB_create(4000)


if case_in == 1:
	img1 = img_art_1
	img2 = img_art_2
	K = K_art
	baseline = 536.62
	f_len = 1733.74
	threshold_ = 0.3
	n_matches_ = 1350
	np.random.seed(102)

elif case_in == 2:
	img1 = img_chess_1
	img2 = img_chess_2
	K = K_chess
	baseline = 97.99
	f_len = 1758.23
	threshold_ = 0.21
	np.random.seed(217)
	n_matches_ = 280

else:
	img1 = img_ladder_1
	img2 = img_ladder_2
	K = K_ladder
	baseline = 228.38
	f_len = 1734.16
	threshold_ = 0.18
	n_matches_ = 1100
	np.random.seed(102)

# Detect features in each set of images using orb detector
detected_kps1, des1, gray1 = get_features_in_img(img1, detector=orb)
detected_kps2, des2, gray2 = get_features_in_img(img2, detector=orb)

# Creating a brute force matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Find matches between art image 1 and image 2 and sort them using distance
matches_1_2 = bf_matcher.match(des1, des2)
matches_1_2 = sorted(matches_1_2, key = lambda x:x.distance)


# Visualize the matches
img_match = cv2.drawMatches(gray1, detected_kps1, gray2, detected_kps2, matches_1_2[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img_match2 = cv2.drawMatches(gray3, detected_kps3, gray4, detected_kps4, matches_3_4[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


##################################################################################################################################################
# Find the F matrix using Ransac method
F_matrix, in_pts1, in_pts2 = find_F_RANSAC(detected_kps1, detected_kps2, matches_1_2[:n_matches_], threshold = threshold_, iterations = 2000)
F_matrix = rank_2_F(F_matrix)

##################################################################################################################################################

print("The Fundamental Matrix F is: \n", F_matrix)

# Find the Essential matrix E
E_matrix = K.T @ F_matrix @ K
print("The Essential Matrix E is: \n", E_matrix)

# 
U, D, V_T = np.linalg.svd(E_matrix)
C_vec = U[:, 2]
R_mat = U @ W @ V_T

print("The Rotation matrix is: \n" ,R_mat)
print("The translation vector is: \n", C_vec)


show_image_reshaped(img_match, "Matches visualization", img_match.shape[1]//3, img_match.shape[0]//3)

_, H1, H2 = cv2.stereoRectifyUncalibrated(in_pts1, in_pts2, F_matrix, imgSize=(gray1.shape[1], gray1.shape[0]))

print("The homography matrix for image 1 is: \n",H1)
print("The homography matrix for image 2 is: \n",H2)

img1_rectified = cv2.warpPerspective(gray1, H1, (gray1.shape[1], gray1.shape[0]))
img2_rectified = cv2.warpPerspective(gray2, H2, (gray2.shape[1], gray2.shape[0]))

img1_rectified_half = cv2.resize(img1_rectified, (img1_rectified.shape[1] // 2, img1_rectified.shape[0] // 2))
img2_rectified_half = cv2.resize(img2_rectified, (img2_rectified.shape[1] // 2, img2_rectified.shape[0] // 2))

# disparity_map = find_disparity_match(img1_rectified_half, img2_rectified_half, kernel_size=7, max_disparity=40)

# disparity_map = find_disparity_match_slow(img1_rectified_half, img2_rectified_half, kernel_size=7, max_disparity=40)
disparity_map = find_disparity_match_eff(img1_rectified_half, img1_rectified_half, kernel_size=7, max_disparity=40)

show_image_reshaped(img1_rectified, "rectified_1 artroom", img1_rectified.shape[1]//2, img1_rectified.shape[0]//2)
show_image_reshaped(img2_rectified, "rectified_2 artroom", img2_rectified.shape[1]//2, img2_rectified.shape[0]//2)
show_image_reshaped(disparity_map, "disparity image", disparity_map.shape[1], disparity_map.shape[0])

stop = time.time()

print("Time taken: ", stop - start)

# Drawing keypoints in images for visualization (not visualized currently)
# detected_kps_img1 = cv2.drawKeypoints(gray1, detected_kps1, img1)
# detected_kps_img2 = cv2.drawKeypoints(gray2, detected_kps2, img2)


 
# Find epilines for points in the right image and drawing its lines on left image
lines_1 = cv2.computeCorrespondEpilines(in_pts2.reshape(-1, 1, 2), 2, F_matrix)
lines_1 = lines_1.reshape(-1,3)
img5, img6 = draw_eplines(img1_rectified, img2_rectified, lines_1, in_pts1, in_pts2)
# Find epilines for points in the left image and drawing its lines on right image
lines_2 = cv2.computeCorrespondEpilines(in_pts1.reshape(-1, 1, 2), 1, F_matrix)
lines_2 = lines_2.reshape(-1,3)
img3, img4 = draw_eplines(img2_rectified, img1_rectified, lines_2, in_pts2, in_pts1)
plt.figure()
plt.subplot(121)
plt.title("Epilines Image 1")
plt.imshow(img5)
plt.subplot(122)
plt.title("Epilines Image 2")
plt.imshow(img3)
plt.show()

# Plot the disparity heatmap
plt.figure()
plt.title("Disparity Heatmap")
plt.imshow(disparity_map, cmap='hot', interpolation='gaussian')
plt.show()

# Calculating depth from disparity map and saving the depth map
depth = (baseline*f_len) / (disparity_map + 1e-8)
depth[depth > 100000] = 100000
depth_map = np.uint8(depth * 255 / np.max(depth))
cv2.imshow('Depth map grayscale', depth_map)

plt.figure()
plt.title("Depth Heatmap")
plt.imshow(depth_map, cmap='hot', interpolation='gaussian')
plt.show()

cv2.waitKey(0)

cv2.destroyAllWindows()
