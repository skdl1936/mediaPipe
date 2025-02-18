import numpy as np

# CSV 파일 불러오기
file = np.genfromtxt('./data/train.csv', delimiter=',', skip_header=1)

# NaN 값을 0으로 변경
file = np.nan_to_num(file)

# 수정된 데이터를 다시 저장
np.savetxt('./data/train_modified.csv', file, delimiter=',', fmt='%f')

print("NaN을 0으로 대체한 CSV 저장 완료!")
