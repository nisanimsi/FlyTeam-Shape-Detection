# Color detection, contour filtering and shape classification

import cv2
import numpy as np

BLUE_LOWER = np.array([90, 170, 110])
BLUE_UPPER = np.array([130, 255, 255])

RED_LOWER_1 = np.array([0, 100, 100])
RED_UPPER_1 = np.array([10, 255, 255])

RED_LOWER_2 = np.array([170, 100, 100])
RED_UPPER_2 = np.array([179, 255, 255])

MIN_AREA = 400


def create_color_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    mask_red1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    return {"red": mask_red, "blue": mask_blue}


def find_contours(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_AREA]


def classify_shape(contour):
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
        edges.append(np.linalg.norm(p1 - p2))

    shortest = min(edges)
    longest = max(edges)

    if shortest <= 0:
        return None

    ratio = longest / shortest

    if corner_count == 3 and ratio <= 1.35:
        return "triangle"
    elif corner_count == 6 and ratio <= 1.45:
        return "hexagon"
    else:
        return None


def calculate_centroid(contour):
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy)


def detect_targets(masks):
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
