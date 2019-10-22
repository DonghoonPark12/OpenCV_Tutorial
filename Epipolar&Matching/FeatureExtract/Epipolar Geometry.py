import numpy as np
import cv2
import sys

sys.path.append("D:\문서\Visual Studio 2015\Projects\Python\OpenCV")

def drawLines(img1, img2, lines, pts1, pts2):
    ''' 
    image on which we draw the epilines for the points in img2
    lines - corresponding epilines 
    '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist()) ##
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [0, -(r[2] + r[0]*c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 3, color, -1)
        cv2.circle(img2, tuple(pt2), 3, color, -1)
    return img1, img2

def sift():
    img1 = cv2.imread('epipolar_geometry_dvd_left.jpg',1)
    img2 = cv2.imread('epipolar_geometry_dvd_right.jpg',1)

    # 에러 해결
    # https://stackoverflow.com/questions/30506126/open-cv-error-215-scn-3-scn-4-in-function-cvtcolor

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Feature point Detection
    sift = cv2.xfeatures2d.SIFT_create()

    # 키포인터를 검출 : 하나의 Output
    #kp1 = sift.detect(gray1, None)
    #kp2 = sift.detect(gray2, None)

    # 키포인터를 검출하고, 디스크립터를 계산 : 두개의 Output
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    img1_k = cv2.drawKeypoints(gray1, kp1, img1)
    img2_k = cv2.drawKeypoints(gray2, kp2, img2)

    cv2.imwrite('sift_keypoint_L.jpg', img1_k)
    cv2.imwrite('sift_keypoint_R.jpg', img2_k)

def epipolar():
    img1 = cv2.imread('epipolar_geometry_dvd_left.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('epipolar_geometry_dvd_right.jpg', cv2.IMREAD_GRAYSCALE)

    # 1) SIFT Feature Detection
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # 키포인터를 검출하고, 디스크립터를 계산
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # 2) Nearest neighbor descriptor matching
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    '''
    FLANN
    
    
    '''

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good ,pts1, pts2 = [] , [] , []

    # 3) RANSAC
    '''
    
    '''
    # ratio test as per Lowe's paper : 1-NN
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # 4) Fine Fundamental Matrix
    #F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS) #두 이미지 사이의 Fundamental Metrix를 계산


    # We select only inlier points
    #pts1 = pts1[mask.ravel() == 1]
    #pts2 = pts2[mask.ravel() == 1]

    # 오른쪽 특징점에 대응하는 Epipoline을 왼쪽 이미지에 그려준다.
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawLines(img1, img2, lines1, pts1, pts2)

    # 왼쪽 특징점에 대응하는 Epipoline을 오른쪽 이미지에 그려준다. 
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawLines(img2, img1, lines2, pts2, pts1)

    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow('img1', img5)
    cv2.imshow('img2', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

epipolar()
#sift()

# https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html