import numpy as np
import cv2


def nothing(x):  # a dummy function
    pass


img = cv2.imread("sample1.jpg")
cv2.imshow("Orignal Image", img)
cv2.waitKey(0)
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)  # to change width and height
blur = cv2.GaussianBlur(img, (25, 25), 7)  # to scale blur and sharpness

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)
cv2.imshow("HSV image", hsv)
cv2.waitKey(0)
lower_hsv = np.array([0, 70, 60])  # to use trackbar input in image array manipulation
upper_hsv = np.array([240, 180, 180])

mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # to scale range of hsv acc to image
res = cv2.bitwise_and(hsv, hsv, mask=mask)

minSizeMM = 40
knownDistancePx = 170
knownDistanceMM = 150
thStone = 100
thShadow = 128
pix2mm = knownDistancePx / knownDistanceMM
scale = 0.5
pix2mm /= scale
minSizePx = minSizeMM / pix2mm
closerSize = minSizePx / 2.0
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
dst = cv2.equalizeHist(closing)

contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # mask is used to draw contours

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.006 * cv2.arcLength(contour, True),
                              True)  # approximate contours are made assuming arc length variable
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 3)
    if len(approx) > 20:  # constraints to define shape
        cv2.putText(img, "Rocks ", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)

cv2.imshow("stabilized Image Mask", dst)
cv2.waitKey(0)
cv2.imshow("Result", img)
cv2.waitKey(0)

cv2.destroyAllWindows()