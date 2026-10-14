# FlyTeam - Real-Time Shape Detection
# Camera capture, position calculation and visualization
import cv2
from image_processing import create_color_masks

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

        cv2.imshow("FlyTeam", frame)
        masks = create_color_masks(frame)

        cv2.imshow("Mavi Maske", masks["blue"])
        cv2.imshow("Kirmizi Maske", masks["red"])
                
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()