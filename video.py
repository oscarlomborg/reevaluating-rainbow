import cv2
import matplotlib.pyplot as plt

video_path = "./experiments/MinAtar_Breakout-v1/monitor/rl-video-episode-500.mp4"

cap = cv2.VideoCapture(video_path)

ok, frame = cap.read()
cap.release()

if ok:
    # OpenCV reads BGR; matplotlib expects RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(frame.shape, frame.dtype)

    plt.imshow(frame, interpolation="nearest")
    plt.axis("off")
    plt.show()