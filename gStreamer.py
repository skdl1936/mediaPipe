import cv2

# GStreamer를 사용한 OpenCV 카메라 스트리밍
video = cv2.VideoCapture("libcamerasrc ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)

if not video.isOpened():
    print("❌ GStreamer를 통한 카메라 초기화 실패!")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        continue

    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
