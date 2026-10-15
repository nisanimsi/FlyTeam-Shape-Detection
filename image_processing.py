
# FlyTeam - Image Processing
# HSV color detection, contours, shape classification and centroid
# Contour detection, shape classification and centroid calculation

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

MIN_AREA = 400

def find_contours(mask):
    """
    Renk maskesindeki konturlari bulur.
    Belirlenen alan esiginin altindaki konturlari eler.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= MIN_AREA:
            valid_contours.append(contour)

    return valid_contours


def classify_shape(contour):
    """
    Konturun kose sayisini ve kenar uzunluklarini inceleyerek
    eskenar ucgen veya duzgun altigen olup olmadigini belirler.
    """
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

    if not cv2.isContourConvex(approx):
        return None

    corner_count = len(approx)

    if corner_count != 3 and corner_count != 6:
        return None

    points = approx.reshape(-1, 2).astype(np.float32)
    edges = []

    for i in range(corner_count):
        p1 = points[i]
        p2 = points[(i + 1) % corner_count]

        length = np.linalg.norm(p1 - p2)
        edges.append(length)

    shortest = min(edges)
    longest = max(edges)

    if shortest <= 0:
        return None

    edge_ratio = longest / shortest

    if corner_count == 3 and edge_ratio <= 1.35:
        return "triangle"

    if corner_count == 6 and edge_ratio <= 1.45:
        return "hexagon"

    return None


def calculate_centroid(contour):
    """Konturun merkez koordinatlarini cv2.moments ile hesaplar."""
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy)


def detect_targets(masks):
    """
    Kirmizi ucgen ve mavi altigen hedeflerini tespit eder.
    Hedefin adini, merkezini, konturunu ve alanini dondurur.
    """
    targets = []

    target_configs = [
        ("Kirmizi Ucgen", masks["red"], "triangle"),
        ("Mavi Altigen", masks["blue"], "hexagon")
    ]

    for name, mask, expected_shape in target_configs:
        contours = find_contours(mask)

        for contour in contours:
            shape = classify_shape(contour)

            if shape != expected_shape:
                continue

            center = calculate_centroid(contour)

            if center is None:
                continue

            targets.append({
                "name": name,
                "center": center,
                "contour": contour,
                "area": cv2.contourArea(contour)
            })

    return targets