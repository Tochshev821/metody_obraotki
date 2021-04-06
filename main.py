
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
filename = 'IMI119'
dataset = pydicom.dcmread(filename)

plt.imshow(dataset.pixel_array, cmap=pylab.cm.bone)
plt.show()
image = cv2.imread('green-leaves.jpg')
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img) ## 1
green_low = np.array([45 , 100, 50] )
green_high = np.array([75, 255, 255])
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
hsv_img[curr_mask > 0] = ([75,255,200])
viewImage(hsv_img) ## 2
## converting the HSV image to Gray inorder to be able to apply
## contouring
RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
viewImage(gray) ## 3
ret, threshold = cv2.threshold(gray, 90, 255, 0)
viewImage(threshold) ## 4
contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
viewImage(image) ## 5


def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_contour_index = i
        i += 1

    return largest_area, largest_contour_index


# to get the center of the contour
cnt = contours[13]
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
largest_area, largest_contour_index = findGreatesContour(contours)
print(largest_area)
print(largest_contour_index)
print(len(contours))
print(cX)
print(cY)




# import cv2
# import numpy as np
# import pydicom as dicom
#
#
#
# def viewImage(image):
#     cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
#     cv2.imshow('Display', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# def grayscale_17_levels (image):
#     high = 255
#     while(1):
#         low = high - 15
#         col_to_be_changed_low = np.array([low])
#         col_to_be_changed_high = np.array([high])
#         curr_mask = cv2.inRange(gray, col_to_be_changed_low,col_to_be_changed_high)
#         gray[curr_mask > 0] = (high)
#         high -= 15
#         if(low == 0 ):
#             break
# image = cv2.imread('ombre_circle_grayscale.png')
# viewImage(image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grayscale_17_levels(gray)
# viewImage(gray)
# ## получение зеленого представления цвета HSV
# green = np.uint8 ([[[0, 255, 0]]])
# green_hsv = cv2.cvtColor (green, cv2.COLOR_BGR2HSV)
# print (green_hsv)

#[133, 133, 133]
# import matplotlib.pyplot as plt
# from matplotlib import pylab
# import pydicom
#
# filename = 'IMI119'
# dataset = pydicom.dcmread(filename)
#
# plt.imshow(dataset.pixel_array, cmap=pylab.cm.bone)
# plt.show()

#артефакты связанные с железом