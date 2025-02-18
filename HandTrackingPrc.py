import cv2
import mediapipe as mp

# MediaPipe Hands 객체 초기화 ( 손 찾기 관련 기능 불러오기)
mp_hands = mp.solutions.hands

# 손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils

# 손 찾기 관련 세부 설정
hands = mp_hands.Hands(
    max_num_hands=2, # 최대 인식할 손의 개수
    static_image_mode=False, #동적 영상 처리
    min_detection_confidence=0.7, # 손 인식에 대한 신뢰도 기준
    min_tracking_confidence=0.7 # 손 추적에 대한 신뢰도 기준
)

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, img = video.read()
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not ret:
        break

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    k = cv2.waitKey(30)
    if k == 49:
        break

    cv2.imshow('hand', img)

video.release()
cv2.destroyAllWindows()
