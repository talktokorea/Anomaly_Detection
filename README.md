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

## 모델 : [작성 中](https://github.com/sejin-sim/Anomaly_Detection/blob/main/Model.md)

## 논문 : Title 클릭 시 상세 내용 정리
| Num | Title | Summary
|:-:|---|---|
|1| [2021 ACM] Deep Learning for Anomaly Detection, A Review|[note](https://colab.research.google.com/github/sejin-sim/Practical_statistics/blob/main/1_%ED%83%90%EC%83%89%EC%A0%81_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%B6%84%EC%84%9D(EDA).ipynb)|
|2|[2019 ICCV] Memorizing Normality to Detect Anomaly; MemAE|[note](https://colab.research.google.com/github/sejin-sim/Practical_statistics/blob/main/2_데이터와_표본분포.ipynb)|


- reference   
https://hoya012.github.io/blog/anomaly-detection-overview-1/    
https://flonelin.wordpress.com/2017/03/29/novelty%EC%99%80-outlier-detection/   
https://m-insideout.tistory.com/21   
https://spri.kr/posts/view/23193?code=industry_trend    
https://kh-kim.github.io/blog/2019/12/12/Deep-Anomaly-Detection.html    
https://www.makinarocks.ai/ko/blog/view/750
