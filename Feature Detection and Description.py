#---------------------------------------------------------------------------------------------------------------------#
# Introduction to SIFT (Scale-Invariant Feature Transform)
# "Distinctive Image Features from Scale-Invariant Keypoints", 2004.

"""
LoG acts as a blob detector which detects blobs in various sizes due to change in Sigma
Sigma acts as a scaling parameter. 

Gaussian kernel with low sigma gives high value for small corner, 
while guassian kernel with high \sigma fits well for larger corner.

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('assets/box.png')
img2 = cv2.imread('assets/box_in_scene.png')
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## SIFT based matching
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None) #find keypoints and descriptors
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher()

## ORB based matching
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # create BFMatcher object


## match()
matches = bf.match(des1,des2) # Match descriptors.
matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2) # Draw first 10 matches.
plt.imshow(img3),plt.show()


## knnMatch()
# matches = bf.knnMatch(des1, des2, k=2)
# good = []
# for m, n in matches:
#     if m.distance < 0.3 * n.distance:
#         good.append([m])
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# plt.imshow(img3), plt.show()

#img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imwrite('./quiz_image/sift_keypoints.jpg',img)


