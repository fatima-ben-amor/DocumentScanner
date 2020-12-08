import NaturalFeatureDoc
import cv2
import numpy as np


def resizeImage(img, ratio):
    colones, lines = img.shape
    newSize = (int(lines * ratio / 100), int(colones * ratio / 100))
    newImg = cv2.resize(img, newSize)
    return newImg


def resizeImages(img, ratio):
    colones, lines, colors = img.shape
    newSize = (int(lines * ratio / 100), int(colones * ratio / 100))
    newImg = cv2.resize(img, newSize)
    return newImg


sourceImg = cv2.imread("des pierres.jpg", cv2.IMREAD_GRAYSCALE)  # image source
markerImg = cv2.imread("fullEcran.png", cv2.IMREAD_GRAYSCALE)  # image marqueur a chercher dans l image sourec
SourceOigrinaleResize = resizeImage(sourceImg, 45)
MarkerOigrinaleResize = resizeImage(markerImg, 55)

# Initiate ORB detector
orb = cv2.ORB_create()

# create BFMatcher object
bf = cv2.BFMatcher()  # (cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.


# find the keypoints and descriptors with ORB
kpSource, desSource = orb.detectAndCompute(SourceOigrinaleResize,
                                           None)  # calcul des pts d'interet' et de leur descripteur
kpMarker, desMarker = orb.detectAndCompute(MarkerOigrinaleResize,
                                           None)  # calcul des pts d'interet' et de leur descripteur

sourceKeyPoints = cv2.drawKeypoints(SourceOigrinaleResize, kpSource, None)
markerKeyPoints = cv2.drawKeypoints(MarkerOigrinaleResize, kpMarker, None)
cv2.imshow("sourceKeyPoints", sourceKeyPoints)
cv2.imshow("markerKeyPoints", markerKeyPoints)
matches = bf.knnMatch(desSource, desMarker, k=2)
print(len(matches))

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 200:  # 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(sourceImg, kpSource, markerImg, kpMarker, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3Resize = resizeImages(img3, 45)

cv2.imshow("img3Resized", img3Resize)

if len(good) > 30:
    good = []
    srcPts = np.float32([kpSource[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destPts= np.float32([kpMarker[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(srcPts, destPts, cv2.RANSAC, 5)
    print(matrix)

while True:
    key = cv2.waitKey(1)
    if key == 27: break
cv2.destroyAllWindows()
