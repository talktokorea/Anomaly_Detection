# Anomaly_Detection : 정의 / 모델 / 논문 

## 정의
1. Supervised 
   - 정상 sample과 비정상 sample의 Data와 Label이 모두 존재하는 경우
   - pros : 양/불 판정 정확도가 높다.
   - cons : 비정상 sample을 취득하는데 시간과 비용이 많이 든다. Class-Imbalance 문제를 해결해야 한다.

2. Semi-supervised
   - 정상 sample만 이용해서 모델을 학습하는 경우
   - pros : 비교적 활발하게 연구가 진행되고 있으며, 정상 sample만 있어도 학습이 가능하다.
   - cons : Supervised Anomaly Detection 방법론과 비교했을 때 상대적으로 양/불 판정 정확도가 떨어진다.
   - method : One-Class SVM & Deep SVDD
 
3. Unsupervised 
   - 대부분의 데이터가 정상 sample이라는 가정을 하여 Label 취득 없이 학습을 시키는 경우
   - pros : Labeling 과정이 필요하지 않다.
   - cons : 양/불 판정 정확도가 높지 않고 hyper parameter에 매우 민감하다.
   - method : PCA & Auto-Encoder based Method 


![image](https://user-images.githubusercontent.com/67107675/114683454-0102c980-9d4b-11eb-9f95-01c5483bc9a8.png)

- Novelty Detection vs Outlier Detection : 비정상 sample을 구분. If 강아지 = normal class
   - Novel sample(=Detection) : 새로운 형태의 강아지
   - Outlier/Abnormal sample(=Detection) : 호랑이, 말(강아지와 관련 x)

- Out-of-distribution Detection : if In-dstribution 데이터 셋 = CIFAR-10, Out-of-distribution 데이터 셋 = LSUN, SVHN
   - CIFAR-10인 In-distribution 데이터 셋을 얼마나 정확히 분류 하는지
   - LSUN, SVHN인 Out-of-distribution 데이터 셋은 얼마나 잘 걸러낼 수 있는지를 살펴보는 방식

## 모델 : [작성 中](https://github.com/sejin-sim/Anomaly_Detection/blob/main/Methods/README.md)

## 논문 정리
| Num | Title | Summary
|:-:|---|---|
|1| [[2021 ACM] Deep Learning for Anomaly Detection, A Review](https://github.com/sejin-sim/Anomaly_Detection/blob/main/paper/Deep_Learning_for_Anomaly_Detection_A_Review.pdf)|Review 논문 <br> - Major Problem Complexities <br> - Main Challenges Tackled <br>  - Categorization of Deep Anomaly Detection <br>   1. Deep Learning for Feature Extraction <br>   2. Learning Feature Representations of Normality <br>   3. End-to-end Anomaly Score Learning| 
|2|[[2019 ICCV] Memorizing Normality to Detect Anomaly; MemAE](https://github.com/sejin-sim/Anomaly_Detection/blob/main/paper/Memorizing_Normality_to_Detect_Anomaly_(MemAE).pdf)|- AE의 한계점 : 종종 이상치 재구성이 잘 됨 <br>- 제안론 구조 : 인코더-메모리모듈(hard shrinkage operator)-디코더 <br>  1. 인코더 : 인풋을 인코딩하여 쿼리로 사용 <br> 2. 메모리 모듈 <br>   1) 표준 정상 패턴을 기록<br>    2) Hard Shrinkage(유사도가 임계치보다 낮은 경우 0으로 만듬) <br> → 메모리 효율성 & 이상치의 희소성 ↑ <br>  3. 디코더 : 찾은 유사한 정상 패턴을 통해 재구성<br> - 결론 : 재구성 에러가 높은 샘플 = 이상치|
|3|[[2020 KDD] UnSupervised Anomaly Detection on Multivariate Time : USAD](https://github.com/sejin-sim/Anomaly_Detection/blob/main/paper/USAD%20UnSupervised%20Anomaly%20Detection%20on%20Multivariate%20Time.pdf)|- 연구 배경 : 프랑스 통신기업인 Orange에서 IT 모니터링을 위해 개발한 방법론<br>- 구조 : GAN에서 아이디어를 얻어 적대적으로 훈련된 AE를 사용<br>&nbsp;&nbsp;1. Training Stage<br> &nbsp;&nbsp;&nbsp;: 기존 AE 훈련 + 일종의 생성자인 AE_1, 판별자 AE_2를 각각 Loss를 나눠서 훈련<br>&nbsp;&nbsp;&nbsp;ㄴ 훈련 단계에서는 정상 데이터만을 가지고 학습함<br>&nbsp;&nbsp;2. Detectoin stage <br>&nbsp;&nbsp;&nbsp;&nbsp;: Test로, 알파&베타 하이퍼파라미터를 통해 민감도를 정해서 이상치 결정<br>- 실험 결과 : 공공데이터&Orange데이터에서 SOTA 달성<br>- Code : test 시 with torch.no_grad(): 조차 안되어 있음 → 공식 코드는 없지만, 구조 이해에 도움이 됨|

- reference   
https://hoya012.github.io/blog/anomaly-detection-overview-1/    
https://flonelin.wordpress.com/2017/03/29/novelty%EC%99%80-outlier-detection/   
https://m-insideout.tistory.com/21   
https://spri.kr/posts/view/23193?code=industry_trend    
https://kh-kim.github.io/blog/2019/12/12/Deep-Anomaly-Detection.html    
https://www.makinarocks.ai/ko/blog/view/750
