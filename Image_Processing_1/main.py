import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np

def CalcOfDamageAndNoneDamage (image):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(image, kernel)
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    markers = np.zeros((image.shape[1], image.shape[0]), dtype="int32")
    markers[90:140, 90:140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1

    leaves_area_BGR = cv.watershed(image_erode, markers)

    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    ill_part = leaves_area_BGR - healthy_part

    mask = np.zeros_like(image, np.uint8)
    mask[leaves_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)

    return mask


originalImage = cv.imread("12.jpg")

if originalImage is None:
    sys.exit("Could not read the image.")

originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)

# Bilateral Filter
kernel = np.ones((5, 5), np.float32)/25
bilateral = cv.bilateralFilter(originalImage, 15, 75, 75)

# Blur
blur = cv.blur(originalImage, (5, 5))

# Gaussian Blur
gblur = cv.GaussianBlur(originalImage, (5, 5), 0)

# Median Filter
median = cv.medianBlur(originalImage, 5)

# Non-Local Means
nlmeans = cv.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)

originalWatershed = CalcOfDamageAndNoneDamage(originalImage)
blurWatershed = CalcOfDamageAndNoneDamage(blur)
gblurWatershed = CalcOfDamageAndNoneDamage(gblur)
medianWatershed = CalcOfDamageAndNoneDamage(median)
bilateralWatershed = CalcOfDamageAndNoneDamage(bilateral)
nlmeansWatershed = CalcOfDamageAndNoneDamage(nlmeans)
titles = ['Original Image', 'Bilateral', 'Blur', 'Gaussian Blur', 'Non-Local Means', 'Median Filter', 'Watershed on Original',
          'Watershed with bilateral', 'Watershed with Blur', 'Watershed with Gaussian',
          'Watershed with Nl Means', 'Watershed with Median']
images = [originalImage, bilateral, blur, gblur, nlmeans, median, originalWatershed, bilateralWatershed, blurWatershed,
          gblurWatershed, nlmeansWatershed, medianWatershed]

for i in range(12):
    plt.subplot(2, 6, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
