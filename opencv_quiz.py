import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#img = cv2.imread("./lena.bmp",cv2.IMREAD_COLOR)
# b, g, r = cv2.split(img)   # img파일을 b,g,r로 분리, cv2.split은 비효율적
# b = img[:, :, 0]  # 0 : Blue, 1 : Green, 2 : Red
# g = img[:, :, 1]  # 0 : Blue, 1 : Green, 2 : Red
# r = img[:, :, 2]  # 0 : Blue, 1 : Green, 2 : Red
# img = cv2.merge([r,g,b])   # b, r을 바꿔서 Merge
#----------------------------------------------------------------------------------------------------------------------#
# 1 [DONE]
# img = cv2.imread("./lena.bmp", 0)
# original_img = img.copy()
# for i in range(img.shape[1]):
#     for j in range(img.shape[0]):
#         if img[i][j] < 128:
#             img[i][j] = 255 * 2 * (img[i][j] / 255) * (img[i][j] / 255)
#             #img[i][j][0] = 255 * 2 * (img[i][j][0]/255)*(img[i][j][0]/255)
#             #img[i][j][1] = 255 * 2 * (img[i][j][1]/255)*(img[i][j][1]/255)
#             #img[i][j][2] = 255 * 2 * (img[i][j][2]/255)*(img[i][j][2]/255)
#         else:
#             img[i][j] = 255 * ( -2 * ((255-img[i][j]) / 255) *((255-img[i][j]) / 255) + 1)
#             #img[i][j][0] = 255 * ( -2 * ((255-img[i][j][0]) / 255) *((255-img[i][j][0]) / 255) + 1)
#             #img[i][j][1] = 255 * ( -2 * ((255-img[i][j][1]) / 255) *((255-img[i][j][1]) / 255) + 1)
#             #img[i][j][2] = 255 * ( -2 * ((255-img[i][j][2]) / 255) *((255-img[i][j][2]) / 255) + 1)
#
# cv2.imshow("Original image",original_img)
# cv2.imshow("Enhanced image",img)
# cv2.imwrite("./quiz_image/ Enhanced_contrast_img.bmp", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------#
# 2 [DONE]
LP = []
std = 4
kernel_size = 3
def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size: size+1, -size_y: size_y+1]
    g = np.exp(-(x**2 + y**2)/(2*std*std))
    return g / float(2 * np.pi * std * std)

def gausian_blur(img, kernel_size):
    img_pad = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)  # Each kernel_size/2 * 2
    img_pad[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img               # kernel_size/2
    img_gau = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(0, img_pad.shape[1]-2):                             # Each kernel_size/2 * 2
        for j in range(0, img_pad.shape[0]-2):
            #_sum = 0
            for y in range(0, 3):                                      # kernel_size
                for x in range(0, 3):
                    img_gau[i][j] += img_pad[i+y][j+x] * gaussian_kernel(kernel_size/2)[y][x]
            #img_gau[i][j] = _sum
    return img_gau

img = cv2.imread("./lena.bmp", 0)
original_img = img.copy()

for i in range(3):
    img_gau = gausian_blur(img, kernel_size=3)
    plt.imshow(img_gau)
    plt.show()
    cv2.imwrite("img.jpg", img_gau+100)
    res1 = cv2.subtract(img, img_gau)
    LP.append(res1)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("img", img)
    cv2.imshow("res", res1)
    cv2.waitKey(0)
    # plt.imshow(res1)
    # plt.imshow(img)
    # plt.show()

# http://subsurfwiki.org/wiki/Gaussian_filter
#----------------------------------------------------------------------------------------------------------------------#
# 3 Image Restore

for i in range(3):
    img = cv2.resize(img, None, fx=2, fy=2)
    img = cv2.add(img, LP[2-i])
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# 4 [DONE]
# img = cv2.imread("./lena.bmp", 0)
# original_img = img.copy()
# f = np.fft.fft2(img)        # 이미지에 푸리에 변환 적용
# fshift = np.fft.fftshift(f) # 주파수가 0인 부분을 중앙에 위치시킴
#
# rows, cols, = img.shape
# crow, ccol = rows/2, cols/2 # 이미지의 중심 좌표
# d = 30  # 중앙에서 10X10 사이즈의 사각형의 값을 1로 설정함. 중앙의 고주파를 모두 제거
# mask = np.zeros((rows,cols),np.uint8)
# mask[int(crow-d):int(crow+d), int(ccol-d):int(ccol+d)] = 1
# fshift = fshift*mask # mast 적용
#
# f_ishift = np.fft.ifftshift(fshift) #푸리에 변환결과를 다시 이미지로 변환
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
#
# plt.imshow(original_img, cmap='gray')
# plt.imshow(img_back, cmap='gray')
# cv2.imwrite("./quiz_image/Filtered_Image.jpg", img_back)
# plt.show()
#----------------------------------------------------------------------------------------------------------------------#