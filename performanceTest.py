import cv2
import mediapipe as mp
import time

class HandGestureRecognition:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # 웹캠 열기
        self.frame_count = 0
        self.start_time = time.time()

        # Mediapipe 손 검출 모델 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def measure_performance(self):
        """ FPS 및 Inference Time을 측정하고 화면에 표시하는 메서드 """
        while self.video.isOpened():
            ret, img = self.video.read()
            if not ret:
                break

            frame_start = time.time()  # 프레임 시작 시간 기록

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(img_rgb)  # 손 탐지 실행

            frame_end = time.time()  # 프레임 종료 시간 기록

            # FPS 계산 (1초 동안 몇 개의 프레임을 처리하는지)
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time

            # Inference Time 계산 (각 프레임당 소요 시간)
            inference_time = frame_end - frame_start

            # 화면에 표시
            cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Inference Time: {inference_time:.4f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Game', img)

            if cv2.waitKey(1) == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    hand_gesture = HandGestureRecognition()
    hand_gesture.measure_performance()
