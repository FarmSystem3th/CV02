'''
이미지 처리 - 아날로그 이미지 처리, 디지털 이미지 처리

디지털 이미지 처리 - 컴퓨터가 디지털 이미지를 처리 (이미지 향상, 이미지 복원, 특징 추출 등)

처리 단계
1. 이미지 획득
2. 이미지 개선
3. 이미지 분석
4. 이미지 해석 및 이해

컴퓨터 비전 - 기계가 시각적 데이터를 이해하고 분석하는 능력을 개발하는 과학 분야

이미지 처리의 목표 - 디지털 이미지의 향상, 변형, 복원 등
컴퓨터 비전의 목표 - 이미지 처리에서 생성된 이미지를 분석하고 해석

낮은 수준 비전 작업 : 노이즈 제거, 대비 향상, 채도 향상, 에지 검출
중간 수준 비전 작업 : 이미지 영역 분할, 이미지 객체로 분할, 이미지 광학 흐름 추정
높은 수준 비전 작업 : 객체 인식, 장면 재구성, 이미지 학습 및 추론

합성곱 신경망(CNN, Convolution Neural Network) - 제공된 데이터에서 추출된 특징의 공간적 계층 구조를 자동으로 학습하도록 설계 (이미지 분류, 물체 감지, 의미적 분할 작업에 사용)

Python
다향성 부분
def pet_sound(pet:Pet):
	pet.make_sound()
pet:Pet 은 타입 힌팅으로, 변수나 함수 매개변수, 반환값의 예상 타입을 명시하며 변수명:타입 형식으로 지정한다.

텐서플로
-편의성 : 고수준 API 지원(Keras) API:소프트웨어 컴포넌트들이 서로 상호작용할 수 있도록 정의된 규칙과 인터페이스
-이식성 및 호환성 : 다양한 플랫폼이나 기기에서 실행 가능
-확장성 : 모델 개발과 배포 모두를 위한 확장 가능한 솔루션 제공(분산 컴퓨팅, CPU GPU TPU 확장성, 대규모 모델 배포)
-유연성 : 사용자 정의 층, 사용자 정의 손실 함수, tf.data를 사용한 데이터 처리
'''
'''!wget https://raw.githubusercontent.com/Cobslab/imageBible/main/image/like_lenna224.png

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
image = cv2.imread('https://raw.githubusercontent.com/Cobslab/imageBible/main/image/like_lenna224.png', cv2.IMREAD_GRAYSCALE)
if image is not None:
  print("이미지를 읽어왔습니다.")
else:
  print("이미지를 읽어오지 못했습니다.")

print(f"변수 타입 : {type(image)}")
print(f"이미지 배열의 형태 : {image.shape}")
cv2_imshow(image)

image_small = cv2.resize(image, (100, 100)) # (100,100)사이즈로 수정
cv2_imshow(image_small)

image_big = cv2.resize(image, dsize = None, fx = 2, fy = 2) # fx fy를 2배율로 수정
cv2_imshow(image_big)

image_fliped = cv2.flip(image, 0) # 수평축 반전
cv2_imshow(image_fliped)

image_fliped = cv2.flip(image, 1) # 세로축 반전
cv2_imshow(image_fliped)

height, width = image.shape # 원하는 각도로 회전
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
result = cv2.warpAffine(image, matrix, (width, height))
cv2_imshow(result)
#image.shape는 세로 가로 순이므로 height width 순으로 대입하고, 함수에는 x축 y축 순으로 대입해야하기 때문에 width height순으로 대입한다. 1은 몇배율로 할것인지

cv2_imshow(image[:100, :100])
cv2_imshow(image[50:150, 50:150])

croped_image = image[50:150, 50:150].copy() # copy 를 사용하면 원본 이미지가 바뀌지않음
croped_image[:] = 200
cv2_imshow(image)

croped_image = image[50:150, 50:150] # 원본 이미지에 영향 O
croped_image[:] = 100 # 회색으로 칠함
cv2_imshow(image)

space = np.zeros((500,1000), dtype = np.uint8)
line_color = 255
space = cv2.line(space,(100,100),(800,400),line_color, 3, 1) #100,100 : 선 시작점, 800,400 : 선 끝점, 3 : 두께, 1 : 연속선
cv2_imshow(space)

space = np.zeros((500,1000), dtype=np.uint8)
color = 255
space = cv2.circle(space, (600,200), 100, color, 4, 1) #600,200 : 원 중심, 100 : 반지름, 4 : 두께, 1 : 연속선
cv2_imshow(space)

space = np.zeros((768,1388), dtype=np.uint8)
line_color = 255
space = cv2.rectangle(space, (500,200), (800,400), line_color, 5, 1) #500,200 : 왼쪽 상단 모서리, 800,400 : 오른쪽 하단 모서리
cv2_imshow(space)

space = np.zeros((768, 1388), dtype = np.uint8)
line_color = 255
space = cv2.ellipse(space, (500,300), (300,200),0,90,250,line_color,4) #500,300 : 타원 중심, 300,200 : 타원의 축 길이로 300은 긴축 200은 짧은 축
#0 : 회전 각도, 90 : 타원 시작 각도(0일 경우 가로축에서 시작), 250 : 타원 끝 각도
cv2_imshow(space)

space = np.zeros((768,1388), dtype = np.uint8)
color = 255
obj1 = np.array([[300,500],[500,500],[400,600],[200,600]])
obj2 = np.array([[600,500],[800,500],[700,200]])
space = cv2.polylines(space, [obj1], True, color, 3) #[obj1] : 그릴 점들의 배열, True : 다각형을 닫힌 형태로 그림 (마지막 점과 첫번째점을 연결)
space = cv2.fillPoly(space, [obj2], color, 1) # 다각형을 채우는 함수로 닫힌 형태 여부(True/False)를 명시할 필요 없음
cv2_imshow(space)
'''
import cv2
import numpy as np

# 이미지 파일 경로 (다운로드한 이미지의 경로를 지정)
image_path = '1장/like_lenna224.png' 

# 이미지를 읽어오기
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is not None:
    print("이미지를 읽어왔습니다.")
else:
    print("이미지를 읽어오지 못했습니다.")

print(f"변수 타입 : {type(image)}")
print(f"이미지 배열의 형태 : {image.shape}")

# 이미지 출력
cv2.imshow('Original Image', image)

# 이미지 크기 조정%
image_small = cv2.resize(image, (100, 100))
cv2.imshow('Small Image', image_small)

image_big = cv2.resize(image, dsize=None, fx=2, fy=2)
cv2.imshow('Big Image', image_big)

# 수평축 반전
image_fliped = cv2.flip(image, 0)
cv2.imshow('Flipped Image (Horizontal)', image_fliped)

# 세로축 반전
image_fliped = cv2.flip(image, 1)
cv2.imshow('Flipped Image (Vertical)', image_fliped)

# 이미지 회전
height, width = image.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
result = cv2.warpAffine(image, matrix, (width, height))
cv2.imshow('Rotated Image', result)

# 이미지 슬라이싱
cv2.imshow('Cropped Image (100x100)', image[:100, :100])
cv2.imshow('Cropped Image (50:150, 50:150)', image[50:150, 50:150])

# 이미지 복사 및 변경
croped_image = image[50:150, 50:150].copy()
croped_image[:] = 200
cv2.imshow('Original Image After Change', image)

# 원본 이미지에 영향
croped_image = image[50:150, 50:150]
croped_image[:] = 100
cv2.imshow('Original Image After Change (Influenced)', image)

# 선 그리기
space = np.zeros((500, 1000), dtype=np.uint8)
line_color = 255
space = cv2.line(space, (100, 100), (800, 400), line_color, 3, 1)
cv2.imshow('Line', space)

# 원 그리기
space = np.zeros((500, 1000), dtype=np.uint8)
space = cv2.circle(space, (600, 200), 100, line_color, 4, 1)
cv2.imshow('Circle', space)

# 사각형 그리기
space = np.zeros((768, 1388), dtype=np.uint8)
space = cv2.rectangle(space, (500, 200), (800, 400), line_color, 5, 1)
cv2.imshow('Rectangle', space)

# 타원 그리기
space = np.zeros((768, 1388), dtype=np.uint8)
space = cv2.ellipse(space, (500, 300), (300, 200), 0, 90, 250, line_color, 4)
cv2.imshow('Ellipse', space)

# 다각형 그리기
space = np.zeros((768, 1388), dtype=np.uint8)
obj1 = np.array([[300, 500], [500, 500], [400, 600], [200, 600]])
obj2 = np.array([[600, 500], [800, 500], [700, 200]])
space = cv2.polylines(space, [obj1], True, line_color, 3)
space = cv2.fillPoly(space, [obj2], line_color, 1)
cv2.imshow('Polygons', space)

# 이미지 창 유지
cv2.waitKey(0)
cv2.destroyAllWindows()
