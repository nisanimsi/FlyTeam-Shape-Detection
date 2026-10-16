# Camera capture, position calculation and visualization

import cv2
from image_processing import create_color_masks, detect_targets


def calculate_offset(target_center, camera_center):
    target_x, target_y = target_center
    camera_x, camera_y = camera_center

    dx = target_x - camera_x
    dy = camera_y - target_y

    if dx > 0:
        horizontal = "SAG"
    elif dx < 0:
        horizontal = "SOL"
    else:
        horizontal = "MERKEZDE"

    if dy > 0:
        vertical = "YUKARI"
    elif dy < 0:
        vertical = "ASAGI"
    else:
        vertical = "MERKEZDE"

    return abs(dx), horizontal, abs(dy), vertical

def draw_target_info(frame, label, target_center, camera_center, contour):
    dx_px, horizontal, dy_px, vertical = calculate_offset(target_center, camera_center)

    target_point = (int(target_center[0]), int(target_center[1]))
    camera_point = (int(camera_center[0]), int(camera_center[1]))

    cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
    cv2.line(frame, target_point, camera_point, (0, 255, 255), 2)
    cv2.circle(frame, camera_point, 6, (0, 255, 0), -1)
    cv2.circle(frame, target_point, 6, (0, 0, 255), -1)

    if "Kirmizi" in label:
        shape_name = "KIRMIZI UCGEN"
        shape_color = (0, 0, 255)
    else:
        shape_name = "MAVI ALTIGEN"
        shape_color = (255, 100, 0)

    cv2.rectangle(frame, (5, 5), (290, 117), (0, 0, 0), -1)

    lines = [
        f"Sekil: {shape_name}",
        f"Sekil merkezi: {target_point}",
        f"Kamera merkezi: {camera_point}",
        f"Sag/Sol: {dx_px} px {horizontal}",
        f"Yukari/Asagi: {dy_px} px {vertical}"
    ]

    for i, line in enumerate(lines):
        if i == 0:
            text_color = shape_color
        else:
            text_color = (255, 255, 255)

        cv2.putText( frame, line, (12, 23 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Kare okunamadi.")
            break

        masks = create_color_masks(frame)

        targets = detect_targets(masks)

        h, w = frame.shape[:2]
        camera_center = (w // 2, h // 2)

        cv2.circle(frame, camera_center, 6, (0, 255, 0), -1)

        for target in targets:
            draw_target_info(
                frame,
                target["name"],
                target["center"],
                camera_center,
                target["contour"]
            )

        cv2.imshow("FlyTeam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()