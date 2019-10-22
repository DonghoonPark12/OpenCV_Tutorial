# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
#----------------------------------------------------------------------------------------------------------------------#
'''
 openCV는 BGR로 사용하지만, Matplotlib는 RGB로 이미지를 보여준다.
 <opencv Show> 
 k = cv2.waitKey(0)
 if k == 27: # esc key
    cv2.destroyAllWindow()
    
<plt show>
plt.imshow(img)
# plt.xticks([]) # x축 눈금
# plt.yticks([]) # y축 눈금
plt.show()  
'''
#----------------------------------------------------------------------------------------------------------------------#
# 1. Image Point Processing
img = cv2.imread("./lena.jpg",cv2.IMREAD_COLOR)
#b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리, cv2.split은 비효율적
# b = img[:, :, 0]  # 0 : Blue, 1 : Green, 2 : Red
# g = img[:, :, 1]  # 0 : Blue, 1 : Green, 2 : Red
# r = img[:, :, 2]  # 0 : Blue, 1 : Green, 2 : Red
# img = cv2.merge([r,g,b])   # b, r을 바꿔서 Merge

# cv2.imshow("input image",255 - img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# 2. Image Blurring
'''
cv2.blur(src, ksize) → dst
Parameters: src – Channel 수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
            ksize – kernel 사이즈(ex; (3,3))

cv2.GaussianBlur(img, ksize, sigmaX)
Parameters: img – Chennel 수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
            ksize – (width, height) 형태의 kernel size. width 와 height 는 서로 다를 수 있지만, 양수의 홀수로 지정해야 함.
            sigmaX – Gaussian kernel standard deviation in X direction.

cv2.medianBlur(src, ksize)
Parameters: src – 1,3,4 channel image. depth 가 CV_8U, CV_16U, or CV_32F 이면 ksize 는 3또는5, CV_8U이면 더 큰 ksize가능
            ksize – 1보다 큰 홀수
            
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
Parameters: src – 8-bit, 1 or 3 Channel image
            d – filtering 시 고려할 주변 pixel 지름
            sigmaColor – Color 를 고려할 공간. 숫자가 크면 멀리 있는 색도 고려함.
            sigmaSpace – 숫자가 크면 멀리 있는 pixel 도 고려함.
            
'''
# img = cv2.imread("./messi5.jpg",1)
# bluredimg = cv2.blur(img,(5,5))
# blurbilateral = cv2.bilateralFilter(img,19,75,75)
# cv2.imshow("blurredimg",bluredimg)
# cv2.imshow("blurbilateral",blurbilateral)
# cv2.imshow("input image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# 3. Image Sharpening(Gradient)
'''
cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) → dst
Parameters: src – input image
            ddepth – output image의 depth, -1이면 input image와 동일.
            dx – x축 미분 차수.
            dy – y축 미분 차수.
            ksize – kernel size(ksize x ksize)

cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) → dst
이미지의 가로와 세로에 대한 Gradient를 2차 미분한 값
Sobel filter에 미분 차수 dx와 dy가 2인 경우
blob(주위의 pixel과 확연한 picel차이를 나타내는 덩어리)검출에 많이 사용됩니다.?

cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) → dst
Parameters:	src – source image
            ddepth – output iamge의 depth.
            
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges
Parameters:	image – 8-bit input image
            threshold1 – Hysteresis Thredsholding 작업에서의 min 값
            threshold2 – Hysteresis Thredsholding 작업에서의 max 값
'''
# canny = cv2.Canny(img,30,70)
#
# laplacian = cv2.Laplacian(img,cv2.CV_8U)
# sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
# sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
# sobelxy = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
# images = [img,laplacian, sobelx, sobely, sobelxy, canny]
# titles = ['Origianl', 'Laplacian', 'Sobel X', 'Sobel Y','Soble XY','Canny']
#
# for i in range(6):
#     #plt.subplot(3,3,i+1),plt.imshow(images[i]), plt.title([titles[i]])
#     #plt.xticks([]),plt.yticks([])
#     plt.imshow(images[i]), plt.title(titles[i])
#     plt.show()
#----------------------------------------------------------------------------------------------------------------------#
# Image Pyramid

# 1) Gaussian Pyramid
# lower_reso = cv2.pyrDown(img) # 원본 이미지의 1/4 사이즈
# higher_reso = cv2.pyrUp(img) #원본 이미지의 4배 사이즈
#
# cv2.imshow('img', img)
# cv2.imshow('lower', lower_reso)
# cv2.imshow('higher', higher_reso)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 2) Laplacian Pyramid
# G = img.copy()
# gpA = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpA.append(G)

# for i in gpA:
#     cv2.imshow('img', i)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# lpA = [gpA[5]] # n번째 추가된 Gaussian Image
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpA[i]) # n번째 추가된 Gaussian Image를 Up Scale함.
#     #print(GE.shape[:2][1], GE.shape[1], GE.shape[:2])
#     temp = cv2.resize(gpA[i-1], (GE.shape[:2][1], GE.shape[:2][0])) # 행렬의 크기를 동일하게 만듬.
#     L = cv2.subtract(temp,GE) # n-1번째 이미지에서 n번째 Up Sacle한 이미지 차이 -> Laplacian Pyramid
#     lpA.append(L)

# # 4단계
# # Laplician Pyramid를 누적으로 좌측과 우측으로 재결함
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
#     LS.append(ls)
#
#
# # 5단계
# ls_ = LS[0] # 좌측과 우측이 결합된 가장 작은 이미지
# for i in xrange(1,6):
#     ls_ = cv2.pyrUp(ls_) # Up Sacle
#     temp = cv2.resize(LS[i],(ls_.shape[:2][1], ls_.shape[:2][0])) # 외곽선만 있는 이미지
#     ls_ = cv2.add(ls_, temp) # UP Sacle된 이미지에 외곽선을 추가하여 선명한 이미지로 생성
#
# # 원본 이미지를 그대로 붙인 경우
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
#
# cv2.imshow('real', real)
# cv2.imshow('blending', ls_)
# cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------#
# Image F
"""
Fourier Transform을 적용.
적용을 하면 0,0, 즉 화면 좌측상단점이 중심이고, 거기에 저주파가 모여 있음.
분석을 용이하게 하기 위해 0,0을 이미지의 중심으로 이동 시키고 Log Scaling을 하여 분석이 용이한 결과값으로 변환
"""
f = np.fft.fft2(img) # 이미지에 푸리에 변환 적용
fshift = np.fft.fftshift(f) #분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴. 중앙에 저주파가 모이게 됨.
magnitude_spectrum = 20*np.log(np.abs(fshift)) #spectrum 구하는 수학식.

rows, cols = img.shape
crow, ccol = rows/2, cols/2 # 이미지의 중심 좌표

# 중앙에서 10X10 사이즈의 사각형의 값을 1로 설정함. 중앙의 저주파를 모두 제거
# 저주파를 제거하였기 때문에 배경이 사라지고 경계선만 남게 됨.
d = 10
fshift[crow-d:crow+d, ccol-d:ccol+d] = 1

#푸리에 변환결과를 다시 이미지로 변환
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

#threshold를 적용하기 위해 float type을 int type으로 변환
img_new = np.uint8(img_back);
ret, thresh = cv2.threshold(img_new,30,255,cv2.THRESH_BINARY_INV)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
plt.title('FT'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(thresh, cmap = 'gray')
plt.title('Threshold With FT'), plt.xticks([]), plt.yticks([])
plt.show()


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/lena_gray.png',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

# 아래는 d 사이지의 사각형을 생성한 후, 사각형 바깥쪽을 제거하는 형태임.
# 즉, 고주파영역을 제거하게 됨.
# d값이 작을수록 사각형이 작고, 바깥영역 즉, 고주파영역이  많이 제거되기 때문에 이미지가 뭉게지고
# d값이 클수록 사각형이 크고, 바깥영역 즉, 고주파 영역이 적게 제거되기 때문에 원래 이미지와 가까워짐.

d = 30
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-d:crow+d, ccol-d:ccol+d] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('FT'), plt.xticks([]), plt.yticks([])
plt.show()


#----------------------------------------------------------------------------------------------------------------------#
# [Geometric Transformations of Images]
'''
191022 컴퓨터 비전 공학 수업
Image Traslation은 2x2 Matrix로 표현이 안된다. --> Homogeneous Coordinate


'''
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# img = cv2.imread('image1.jpg')
#
# hist, bins = np.histogram(img.flatten(), 256,[0,256])
# cdf = hist.cumsum()
# # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
# # mask처리가 되면 Numpy 계산에서 제외가 됨
# # 아래는 cdf array에서 값이 0인 부분을 mask처리함
# cdf_m = np.ma.masked_equal(cdf,0)
# #History Equalization 공식
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# # Mask처리를 했던 부분을 다시 0으로 변환
# cdf = np.ma.filled(cdf_m,0).astype('uint8')
#
# img2 = cdf[img]
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.subplot(122),plt.imshow(img2),plt.title('Equalization')
# plt.show()
#
# #%%
#
# import cv2
#
# cap = cv2.VideoCapture('20160225_185348000_iOS.MOV')
# cnt = 0;
# temp = cap.read()
#
# cv2.namedWindow('gray',0)
# cv2.namedWindow('frame',0)
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         hist, bins = np.histogram(img.flatten(), 256,[0,256])
#         cdf = hist.cumsum()
#         # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
#         # mask처리가 되면 Numpy 계산에서 제외가 됨
#         # 아래는 cdf array에서 값이 0인 부분을 mask처리함
#         cdf_m = np.ma.masked_equal(cdf,0)
#         #History Equalization 공식
#         cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#         # Mask처리를 했던 부분을 다시 0으로 변환
#         cdf = np.ma.filled(cdf_m,0).astype('uint8')
#
#         img2 = cdf[img]
#         th,dst = cv2.threshold(gray,200,255,0)
#         cv2.imshow('gray',img)
#         cv2.imshow('frame',img2)
#         cv2.waitKey(33)
#     else:
#         break
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     cnt = cnt + 1
# cap.release()
# cv2.destroyAllWindows()
#
# #%%
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('c://image.jpg')
#
# # [x,y] 좌표점을 4x2의 행렬로 작성
# # 좌표점은 좌상->좌하->우상->우하
# pts1 = np.float32([[504,1003],[243,1525],[1000,1000],[1280,1685]])
#
# # 좌표의 이동점
# pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])
#
# # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
# cv2.circle(img, (504,1003), 20, (255,0,0),-1)
# cv2.circle(img, (243,1524), 20, (0,255,0),-1)
# cv2.circle(img, (1000,1000), 20, (0,0,255),-1)
# cv2.circle(img, (1280,1685), 20, (0,0,0),-1)
#
# M = cv2.getPerspectiveTransform(pts1, pts2)
#
# dst = cv2.warpPerspective(img, M, (1100,1100))
#
# plt.subplot(121),plt.imshow(img),plt.title('image')
# plt.subplot(122),plt.imshow(dst),plt.title('Perspective')
# plt.show()

#%%
#dst = cv2.threshold(img,220,255,0)

#kernel = np.ones((5,5),np.float32)/25
#
#dst = cv2.filter2D(img,-1,kernel)
#blurblur = cv2.blur(img,(5,5))
#blurbilateral = cv2.bilateralFilter(img,19,75,75)
#median = cv2.medianBlur(img,5)
#blurg = cv2.GaussianBlur(img,(5,5),0)
#edges = cv2.Canny(img,100,200)
#
#
#b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리
#img2 = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
#plt.subplot(121),plt.imshow(img2),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.show()
#
# 
#cv2.imwrite("filtered.bmp",edges);
##cv2.imwrite("bilateral.bmp",blur);
#cv2.imshow("img",edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



