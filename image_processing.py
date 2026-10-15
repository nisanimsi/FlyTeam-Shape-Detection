
# FlyTeam - Image Processing
# HSV color detection, contours, shape classification and centroid

import cv2
import numpy as np


MIN_AREA = 400


def find_contours(mask):
    """Renk maskesindeki yeterince buyuk konturlari bulur."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_AREA
    ]


def classify_shape(contour):
    """Konturun ucgen veya duzgun altigen olup olmadigini belirler."""
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return None

    epsilon = 0.025 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    if not cv2.isContourConvex(approx):
        return None

    points = approx.reshape(-1, 2).astype(np.float32)
    corner_count = len(points)

    if corner_count not in (3, 6):
        return None

    # Butun kenarlarin uzunlugunu hesapla.
    edges = [
        np.linalg.norm(
            points[i] - points[(i + 1) % corner_count]
        )
        for i in range(corner_count)
    ]

    shortest = min(edges)
    longest = max(edges)

    if shortest <= 0:
        return None

    ratio = longest / shortest

    # ESKENAR UCGEN
    if corner_count == 3:
        if ratio <= 1.35:
            return "triangle"

        return None

    # DUZGUN ALTIGEN
    if corner_count == 6:
        if ratio <= 1.45:
            return "hexagon"

        return None

    return None


def calculate_centroid(contour):
    """Konturun geometrik merkezini hesaplar."""
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy)


def create_test_masks(frame):
    """
    Dusuk isikta kirmizi ve mavi hedefleri
    daha kararli algilamak icin HSV maskeleri.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Parlaklik kanalina CLAHE uygula.
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    v = clahe.apply(v)

    hsv = cv2.merge((h, s, v))

    # KIRMIZI HSV ARALIKLARI
    # Turuncu tonlarini azaltmak icin daraltilmistir.
    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 135, 45]),
        np.array([5, 255, 255])
    )

    red_mask2 = cv2.inRange(
        hsv,
        np.array([175, 135, 45]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    # MAVI HSV ARALIGI
    blue_mask = cv2.inRange(
        hsv,
        np.array([100, 100, 45]),
        np.array([140, 255, 255])
    )

    # MASKE TEMIZLEME
    kernel = np.ones((3, 3), np.uint8)

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return red_mask, blue_mask


def detect_targets(frame):
    """Yalnizca kirmizi ucgen ve mavi altigeni tespit eder."""
    red_mask, blue_mask = create_test_masks(frame)

    targets = []

    configurations = [
        ("Kirmizi Ucgen", red_mask, "triangle"),
        ("Mavi Altigen", blue_mask, "hexagon")
    ]

    for name, mask, expected_shape in configurations:
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
