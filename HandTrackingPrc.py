import cv2
import mediapipe as mp

# MediaPipe Hands 객체 초기화 ( 손 찾기 관련 기능 불러오기)
mp_hands = mp.solutions.hands

# 손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils

# 손 찾기 관련 세부 설정
hands = mp_hands.Hands(
    max_num_hands=2, # 최대 인식할 손의 개수
    static_image_mode=False, #동적 영상 처리 (True: 정지 이미지 모드, False: 실시간 비디오 모드)
    min_detection_confidence=0.7, # 손 인식에 대한 최소 신뢰도 기준 (0.7 = 70% 이상 확실해야 감지)
    min_tracking_confidence=0.7 # 손 추적에 대한 최소 신뢰도 기준 (0.7 = 70 % 이상 확실해야 추적)
)

video = cv2.VideoCapture(0) #웹캠 열기 ( 외장 카메라는 1, 2 로 설정가능) 
while video.isOpened(): # 웹캠이 정상적으로 열려 있는 동안 실행
    ret, img = video.read() # 카메라에서 프레임(이미지) 읽기
    img = cv2.flip(img,1) # 좌우 반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv의 BGR 형식을 RGB로 변환 -> mediaPipe는 RGB형식을 필요로 하기때문에 변경
    result = hands.process(img) # 손 추적 모델 적용

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 다시 BGR로 변환 (OpenCV에서 출력하려면 필요, cv2.imshow할때 BGR로 변환해야함)
    if not ret:
        break

    if result.multi_hand_landmarks is not None: # 손이 감지된 경우
        for res in result.multi_hand_landmarks: # 각 손에 대해 실행
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) #랜드마크(관절) 그림

    k = cv2.waitKey(30) #키 입력 대기
    if k == 49: #숫자 1 입력시 종료
        break

    cv2.imshow('hand', img) #화면에 영상 출력

video.release() #웹캠 종료
cv2.destroyAllWindows() #모든 창 닫기
