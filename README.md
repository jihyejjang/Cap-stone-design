# Child Abuse Detection
**어린이집 CCTV로 학대상황 감지하기 (~20.7 capstone design)**  
> reference 모델을 활용해 어린이집 cctv 폭력 상황을 감지  
> 기존 논문에서는 다양한 비정상 상황(총기, 폭력, 폭발 등..) 데이터를 활용하였으나, 해당 프로젝트에서는 어린이집 CCTV의 학대 영상만을 활용해 학습함 (어린이집은 동적인 상황도 normal한 경우가 많기 때문)


Implementation of [Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/pdf/1801.04264.pdf) paper from CVPR 2018.  
Original implementation from authors is [here](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).  
Reference of source code is [here](https://github.com/ptirupat/AnomalyDetection_CVPR18)

----

## Models
### C3D

영상을 16frame별 clip으로 잘라 3D convolution
sport1m pretrained weights를 통해, 영상의 feature extraction 진행
즉, 16개의 frame을 4096개의 feature로 압축하게 됨
![1](https://user-images.githubusercontent.com/61912635/191239782-94e3f216-23ab-411a-9036-dae9929e488d.png)
3D convolution은 시간적 특성까지 추출할 수 있다고 알려짐

### Classifier
extracted feature로 비정상/정상 상황에 대한 0~1사이 sigmoid output을 출력(clip단위)

## Anomaly detection [📜]((https://arxiv.org/pdf/1801.04264.pdf))
- clip단위가 아닌, video 단위의 labeling을 통해 시간 소모 줄임
- MIL 알고리즘에서, video는 bag이 되고 clip은 instance가 됨
    - 즉, 높은 비정상 score(1에 가까울수록 비정상)가 부여된 clip이 있는 video는 abnormal로 분류
### Deep MIL(Multiple instance learning)

**loss formulation**
1. Ranking loss
 TP score, FP score의 차이를 키우는 loss (Positive : 비정상 상황일수록 값이 1에 가까워야 함)
 <img width="488" src="https://user-images.githubusercontent.com/61912635/191239787-34aa058a-42e3-4085-a930-b52a5f4c48f9.png">
2. (1) temporal loss : clip i, i+1 간의 score 차이가 부드럽게 이어지도록 만들어주는 loss
    (2) sparse loss : video bag에서 비정상 상황은 아주 짧은 순간이기 때문에, 나머지 순간은 0이 되도록(sparse) 만들어주는 loss
    <img width = "380" src = "https://user-images.githubusercontent.com/61912635/191239791-1528ef67-146a-4d72-bc00-8055f2efdda1.png">

Total loss는 1,2의 합으로 구성

### Visualization

<img width="500" alt="스크린샷 2022-09-20 오후 7 57 38" src="https://user-images.githubusercontent.com/61912635/191240583-04b95c5c-bf48-4bc3-8002-43fbdb45e2d8.png">

<img width="500" alt="스크린샷 2022-09-20 오후 8 00 16" src="https://user-images.githubusercontent.com/61912635/191241038-d1c9a33a-8573-4ffc-b605-2d0cbffb481b.png">



---
### Review
- 처음 딥러닝으로 진행한 프로젝트라 이해가 부족했음
- 기존 모델과의 비교 진행하지 못함
- 처음 목표는 alarm system까지였지만 시간 관계상 진행하지 못함
- ROC curve로 평가까지 하지 못해 아쉬움
- 영상의 3D feature extraction으로 c3d를 사용하였는데, p3d나 i3d같은 모델이 더욱 성능이 좋다고 알려짐 
    - i3d는 1024개의 특징을 추출
