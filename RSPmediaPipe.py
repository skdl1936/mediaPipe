import cv2
import mediapipe as mp
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
import time
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


video = cv2.VideoCapture(0) #웹캠 키기


while video.isOpened(): #웹캠이 켜져있는동안
    ret, img = video.read() # 한 프레임씩 읽어온다.
    if not ret:
        continue

    img = cv2.flip(img, 1) #좌우 반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 변환
    result = hands.process(img)  # 손 탐지하기

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 찾은 손 표시하기
    if result.multi_hand_landmarks is not None: # 손 인식하면, 게임이 끝나지 않은 경우

        # 이미지 손 표현하기
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))  # 21개 관절, xyz값 저장할 배열 생성 -> 즉 21,3의 배열 생성
            # enumerate = for문의 순서 표현
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 연결할 관절 번호 가져오기
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1  # 뼈의 값 (x,y,z좌표값 > 백터값)

            # 유클리디안 길이로 변롼(피타고라스)
            # 뼈의 값(직선 값) -> 크기가 1인 벡터가 나옴
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 뼈의 값으로 뼈사이의 각도 구하기, 변화값이 큰 15개 => angle이 라디안으로 나옴
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

            # radian 각도를 degree각도로 변경하기
            angle = np.degrees(angle)

            # 구한 각도를 knn모델에 예측시키기
            # 학습을 위해서 타입 변경 (2차원 array)
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            
            #gesture에 존재하는 값으로 나옴
            idx = int(results[0][0])

            # 사용자가 낸 가위바위보 표시
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text = rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
               int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

                if not user_rsp :
                    user_rsp = rps_gesture[idx]
                    print('user: ', user_rsp)

            # 손가락 마디마디에 랜드마크 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    # # 사용자가 선택한 가위바위보 한 번만 저장
    # if user_rsp is None and idx is not None:
    #     user_rsp = rps_gesture[idx]
    #     print('user: ', user_rsp)
    #     idx = None

    # 가위바위보 승자 결정
    result = None
    if computer_rsp == 'rock':
        if user_rsp == 'rock':
            result = 'DRAW'
        elif user_rsp == 'paper':
            result = 'WIN'
        elif user_rsp == 'scissors':
            result = 'LOSE'
    elif computer_rsp == 'paper':
        if user_rsp == 'rock':
            result = 'LOSE'
        elif user_rsp == 'scissors':
            result = 'WIN'
        elif user_rsp == 'paper':
            result = 'DRAW'
    elif computer_rsp == 'scissors':
        if user_rsp == 'rock':
            result = 'WIN'
        elif user_rsp == 'paper':
            result = 'LOSE'
        elif user_rsp == 'scissors':
            result = 'DRAW'

    if result:
        cv2.putText(img, text=result, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3, color=(0, 0, 255), thickness=5)
        cv2.putText(img, text=f'computer: {computer_rsp}',
                    org=(int(img.shape[1] / 2), 170),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2)

        #다른 제스쳐
        # cv2.putText(img, text = gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
        #             int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1, color=(0, 0, 255), thickness=2)


    cv2.imshow('Game', img) # 그림 만들기
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('r'):  # 'r' 키를 누르면 게임 리셋
        user_rsp = None
        idx = None
        computer_rsp = random.choice(['rock', 'paper', 'scissors'])  # 컴퓨터 선택도 다시 랜덤
        print('com: ',computer_rsp)

video.release()
cv2.destroyAllWindows()