import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import color
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


def detect_eclipse(img):
    # Load picture, convert to grayscale and detect edges
    image_rgb = img.copy()
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.55, high_threshold=0.8)
    # Perform hough detetection
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    result.sort(order='accumulator')

    if len(result) != 0:
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        return (cy, cx, 1)
    return None


def color_mask(img):
    # Using HSV color map to get the dark region
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([255, 255, 60])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def get_hole(mask):
    # Get the shape of the can's hole
    output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    (numLabels, _, stats, centroids) = output

    max = 0
    pos = -1
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max:
            max = area
            pos = i

    x = stats[pos, cv2.CC_STAT_LEFT]
    y = stats[pos, cv2.CC_STAT_TOP]
    w = stats[pos, cv2.CC_STAT_WIDTH]
    h = stats[pos, cv2.CC_STAT_HEIGHT]

    return (x, y, w, h), centroids[pos]


def draw_on_original(ori, ori_stat, rec_stat):
    # Draw detected rect on original image
    df_x, df_y = ori_stat
    x, y, w, h = rec_stat
    _x = x + df_x
    _y = y + df_y
    output = ori.copy()
    cv2.rectangle(output, (_x, _y), (_x + w, _y + h), (0, 255, 0), 3)
    return output

def slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs((y2-y1)/(x2-x1))


if __name__ == "__main__":
    img = cv2.imread("frame0.jpg")
    try:
        cy, cx = detect_eclipse(img)
    except:
        print('Cannot detect top can')
        exit

    can_top = img[cy.min():cy.max(),
                  cx.min():cx.max()]

    mask = color_mask(can_top)
    rec_stat, center = get_hole(mask)
    ori_stat = (cx.min(), cy.min())
    plt.imshow(draw_on_original(img, ori_stat, rec_stat)), plt.show()

    middle_can = ((cx.min() + cx.max())//2, (cy.min() + cy.max())//2)
    middle_hole = center + np.array([cx.min(), cy.min()])
    _slope = slope(middle_can, middle_hole)
    print(math.degrees(math.atan(_slope)))
