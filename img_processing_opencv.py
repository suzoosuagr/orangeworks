import cv2
import numpy as np

image_path = './data/IMG_5572.jpeg'
img = cv2.imread(image_path)
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
   
# Using cv2.putText() method

   


img_thresh_low = cv2.inRange(img_HSV, np.array([0, 135, 135]), np.array([15, 255, 255]))
# img_thresh_high = cv2.inRange(img_HSV, np.array([159, 135, 80]), np.array([179, 255, 255]))
# img_thresh_or = cv2.bitwise_or(img_thresh_low, img_thresh_high)
# img_thresh = img_thresh_low
kernel = np.ones((5, 5))
img_thresh_opened = cv2.morphologyEx(img_thresh_low, cv2.MORPH_OPEN, kernel)
img_thresh_opened_closed = cv2.morphologyEx(img_thresh_opened, cv2.MORPH_CLOSE, kernel)

img_thresh_blured = cv2.GaussianBlur(img_thresh_opened,(3,3),cv2.BORDER_DEFAULT)

img_edges = cv2.Canny(img_thresh_blured, 80, 160)

contours, hierarchy= cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img, contours, -1, (100, 0, 0), 3)



# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# cv2.imshow("gray",gray)
cv2.imshow("orange low", img_thresh_low)
# cv2.imshow("orange high", img_thresh_high)
cv2.imshow("orange low open", img_thresh_opened)
cv2.imshow("orange low open -> close", img_thresh_opened_closed)
cv2.imshow("orange canny", img_edges)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.06 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        cv2.drawContours(img, [cnt], -1, (255, 0, 0), 3)
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        area = cv2.contourArea(cnt)
        cv2.putText(img, f'Area:{area:.2f}', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)




cv2.imshow('Contours', img)

cv2.waitKey(0)
# cv2.imshow("orange high", img_thresh_high)
cv2.destroyAllWindows()
# cv2.imshow("otsu", thresh)
