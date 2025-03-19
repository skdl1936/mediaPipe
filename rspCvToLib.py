import cv2
import numpy as np
import subprocess
import shlex
import mediapipe as mp
import random


# 인식 가능한 11가지 동작
gesture = {
    0: 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6: 'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

# 가위바위보 동작 만들기
rps_gesture = {
    0: 'rock', 5:'paper', 9:'scissors', 1: 'scissors',
}

# mediapipe 사용하기
# 손 찾기 관련 기능 불러오기
mp_hands = mp.solutions.hands

# 손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils

# 손 찾기 관연 세부 설정
hands = mp_hands.Hands(
    max_num_hands=1,  # 탐지할 최대 손의 객수
    min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
    min_tracking_confidence=0.5  # 표시할 관절의 최소 정확도
)

# 동작 인식 모델 만들기 (knn모델)
file = np.genfromtxt('./data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
lable = file[:,-1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, lable)

# 컴퓨터가 랜덤으로 가위바위보 선택
computer_rsp = random.choice(['rock','paper','scissors'])
print('com: ',computer_rsp)

# 사용자 가위바위보 선택
user_rsp = None

# libcamera-vid 명령어 설정 (MJPEG 스트리밍)
cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 10 -o - --camera 0'

# libcamera-vid 프로세스 실행
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    buffer = b""
    while True:
        buffer += process.stdout.read(8192)  # 4096바이트씩 읽음
        a = buffer.find(b'\xff\xd8')  # JPEG 시작 마커
        b = buffer.find(b'\xff\xd9')  # JPEG 종료 마커

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]  # JPEG 이미지 추출
            buffer = buffer[b+2:]  # 버퍼 업데이트

            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = cv2.flip(img, 1)  # 좌우 반전
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
            result = hands.process(img)  # Mediapipe 손 탐지
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 다시 BGR 변환 (OpenCV 출력용)

            # 손 인식 후 랜드마크 표시
            if result.multi_hand_landmarks:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                    angle = np.degrees(angle)

                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])

                    if idx in rps_gesture:
                        cv2.putText(img, rps_gesture[idx].upper(),
                                    (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if not user_rsp:
                            user_rsp = rps_gesture[idx]
                            print('user:', user_rsp)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 가위바위보 승자 결정
            if user_rsp:
                if (user_rsp == "rock" and computer_rsp == "scissors") or \
                   (user_rsp == "scissors" and computer_rsp == "paper") or \
                   (user_rsp == "paper" and computer_rsp == "rock"):
                    winner = "Win!"
                elif user_rsp == computer_rsp:
                    winner = "Draw!"
                else:
                    winner = "Lose!"

                cv2.putText(img, winner, (int(img.shape[1] / 2) - 100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(img, f"Computer: {computer_rsp}", (int(img.shape[1] / 2) - 100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow('Game', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                user_rsp = None
                idx = None
                computer_rsp = random.choice(["rock", "paper", "scissors"])
                print('com:', computer_rsp)

finally:
    process.terminate()
    cv2.destroyAllWindows()
