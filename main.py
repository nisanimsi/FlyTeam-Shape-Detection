# FlyTeam - Real-Time Shape Detection
# Camera capture, position calculation and visualization
import cv2
from image_processing import create_color_masks

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

def draw_target_info(frame, label, target_center, camera_center):
    dx_px, horizontal, dy_px, vertical = calculate_offset(target_center, camera_center)

    target_point = (int(target_center[0]), int(target_center[1]))
    camera_point = (int(camera_center[0]), int(camera_center[1]))

    cv2.line(frame, target_point, camera_point, (0, 255, 255), 2)
    cv2.circle(frame, camera_point, 6, (0, 255, 0), -1)
    cv2.circle(frame, target_point, 6, (0, 0, 255), -1)

    cv2.putText(frame, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Sekil merkezi: {target_point}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Kamera merkezi: {camera_center}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"yatay: {dx_px} px {horizontal}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"dikey: {dy_px} px {vertical}",(10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)  # 0 = varsayılan webcam
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kare okunamadı.")
            break

        masks = create_color_masks(frame)

        h, w = frame.shape[:2]
        camera_center = (w // 2, h // 2)

        # TODO: etiket ve merkez, kontur fonksiyonundan gelecek
        target_center = None
        if target_center is not None:
            draw_target_info(frame, "KIRMIZI UCGEN", target_center, camera_center)

        cv2.imshow("FlyTeam", frame)

        cv2.imshow("Mavi Maske", masks["blue"])
        cv2.imshow("Kirmizi Maske", masks["red"])
                
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()