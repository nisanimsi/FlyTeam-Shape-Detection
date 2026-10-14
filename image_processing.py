# FlyTeam - Image Processing
# HSV color detection, contours, shape classification and centroid

import cv2
import numpy as np

# --- HSV renk aralıkları (H: 0-179, S: 0-255, V: 0-255) ---
BLUE_LOWER = np.array([90, 170, 110])
BLUE_UPPER = np.array([130, 255, 255])

# Kırmızı H çarkında iki uçta olduğu için iki aralık gerekli
RED_LOWER_1 = np.array([0, 100, 100])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 100, 100])
RED_UPPER_2 = np.array([179, 255, 255])


def create_color_masks(frame):
    """
    BGR kareyi alıp kırmızı ve mavi için ikili maskeler üretir.

    Girdi:  frame -> BGR renkli kare (NumPy dizisi)
    Çıktı:  {"red": kırmızı maske, "blue": mavi maske}
            Maskeler tek kanallı, 255 = renk aralığında, 0 = dışında.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    mask_red1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    return {"red": mask_red, "blue": mask_blue}