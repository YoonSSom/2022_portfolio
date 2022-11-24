# 2022_portfolio
### OCR(Optical Character Recognition)
- [한국어 메뉴판 OCR](https://github.com/YoonSSom/Korean_OCR)(2022.09-10)
- [자동차 번호판 인식](진행 중)(2022.10)
<br><br>
### CV(Computer Vision)
- [Safe_Driving](https://github.com/YoonSSom/Safe_Driving)(2022.04)
* [Car Damage Detection](https://github.com/YoonSSom/CarDamage_Detection)(2022.07-08)
* [Kaggel Competition : HuBMAP + HPA - Hacking The Human Body](https://github.com/YoonSSom/Kaggle)(2022.08-09)
<br><br>
### NLP(Natural Language Processing)
- [국민 청원 데이터 분석](https://github.com/YoonSSom/Text_mining_ML/blob/master/%EA%B5%AD%EB%AF%BC%EC%B2%AD%EC%9B%90%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D(%EC%9B%8C%EB%93%9C%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C).ipynb)(2022.03)
- [기업 동향 분석](https://github.com/YoonSSom/Text_mining_ML/blob/master/%EA%B8%B0%EC%97%85%EB%8F%99%ED%96%A5%EB%B6%84%EC%84%9D.ipynb)(2022.03)
- [뉴스 기사 요약](https://github.com/YoonSSom/Text_mining_ML/blob/master/%EB%89%B4%EC%8A%A4%EA%B8%B0%EC%82%AC%EC%9A%94%EC%95%BD.ipynb)(2022.03)
- [서울시 구별 범죄 분석](https://github.com/YoonSSom/Text_mining_ML/blob/master/%EC%84%9C%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%89%E1%85%B5_%EB%B2%94%EC%A3%84_%EB%B6%84%EC%84%9D.ipynb)(2022.03)
* MUSINSA - Recommending Items, [Predicting Star-rate Using Review](https://github.com/YoonSSom/MUSINSA)(2022.06-07)
- [GPT2활용 자동 문장 생성](https://github.com/YoonSSom/poetry_generator)(2022.05-06)
***
## **TextMining with ML**
### [Repositories](https://github.com/YoonSSom/Text_mining_ML)
### Introduction
* 우크라사태가 한국 경제에 미치는 영향을 파악하기 위해 웹 크롤링한 신문기사에 워드 클라우드, 토픽 모델링 기법을 활용
* 로지스틱 리그레션을 활용해 우크라사태 이전/이후 신문기사 내용의 감성 분석을 진행
<br><br>
### Data and Models
* 네이버 웹툰 장르별 회차별 독자 참여 수치 19622건
* 우크라사태 이전/이후 신문기사 제목 및 본문 2912건
* TF-IDF / LDA / Logistic Regression
<br><br>
### Envs and Requirements
* Google Colab, VScode
* BeautifulSoup, Pandas, RE, KoNLPy, Scikit-Learn, Gensim, Matplotlib, Seaborn
<br><br>
### Progress
* 웹 크롤링 및 텍스트 데이터 정제
* 데이터 분석 및 시각화
* 토큰화 및 코퍼스 사전 구축
* 불용어 사전 및 사용자 사전 구축
* TF-IDF 워드 클라우드
* LDA 토픽 모델링
* 바이그램 / 트라이그램 적용
* 응집도 / 복잡도 기준 최적 에폭 및 토픽 갯수 설정
* 로지스틱 리그레션을 활용한 감성 분석
<br><br>
### Reference
* 텍스트마이닝을 위한 한국어 불용어 목록 연구 - 길호현
* https://www.crummy.com/software/BeautifulSoup/
* https://pandas.pydata.org/
* https://konlpy.org/
* https://radimrehurek.com/gensim/
* https://scikit-learn.org/
* https://www.pythoncheatsheet.org/   

[Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **Distinguish Click-bait**
### [Repositories](https://github.com/heokwon/Distinguish_Click-bait.git)
### Introduction
* 기사와 내용일 일치하지않은, 클릭을 유도하는 피시성 기사를 가리기 위해 진행한 프로젝트
* 딥러닝을 이용해 기사의 내용을 간략하게 요약하고, 헤드라인과 비교
<br><br>
### Data and Models
* 데이터 종류 : 신문기사
* 데이터 형태 : 뉴스 택스트
* 수량 : 원문데이터 30만건
* 출처 : [Ai-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
* model : mT5
<br><br>
### Envs and Requirements
* Google Colab, VScode
* Tensorflow, Hugging Face, Transformers, Pandas
<br><br>
### Progress
* 데이터셋 구축
* 데이터셋 정제
* json -> dictionary
* 데이터셋에서 제목, 원문, 추출요약, 생성요약 추출
* transform AutoTokenizer 사용
* Hugging Face 이용하여 모델 구축 - mT5 모델에 fine-tuning
<br><br>
### Referece
* https://www.tensorflow.org/?hl=ko
* https://huggingface.co/docs/transformers/index
* https://pandas.pydata.org/
* https://aihub.or.kr/
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **MUSINSA - Recommending Items**
### [Repositories]()
### 업데이트중
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **MUSINSA - Predicting Star-rate Using Review**
### [Repositories](https://github.com/heokwon/NLP-MUSINSA/tree/main/Predicting%20Star-Rate%20Using%20Review)
### Introduction
* 신뢰성을 가진 별점 예측을 통해 musinsa 입점브랜드에 관한 실질적 평가지표를 제시
<br><br>
### Data and Models
* Web Crawling을 통한 무신사 댓글
* Web Crawling을 통한 네이버 쇼핑몰 부정댓글
* KoGPT2, KoBERT
<br><br>
### Envs and Requirements
* Google Colab, VScode
* BeautifulSoup, Selenium, Pandas, Hugging Face, Transformers
<br><br>
### Progress
* 무신사 댓글 crawling
* 별점이 3점 이하인 댓글을 부정댓글로 설정
* 네이버 쇼핑몰 부정댓글 crawling - 긍정과 부정댓글의 편차가 심한 이유로 부족한 부정댓글을 추가 crawling
* 별점을 댓글에 대한 라벨로 사용 (1점 ~ 5점)
* Text Augmentation   
Back Translatrion - 기존 텍스트를 외국어로 번역한 뒤, 다시 한글로 번역하여 증식, googletrans 라이브러리의 Translator 모듈 사용     
KoEDA - 단어를 삽입/삭제/위치 변경/ 유의어로 대체 하여 증식하는 기법   
Generation Method - 키워드의 앞,뒤 상관관계 및 유사도를 기반하여 글자 생성을 통한 증식기법   
* Modeling - KoBERT을 사용한 댓글 감성분석을 통해 별점을 다시 매김
<br><br>
### Referece
* https://github.com/SKTBrain/KoBERT
* https://github.com/SKT-AI/KoGPT2
* https://www.crummy.com/software/BeautifulSoup/
* https://pandas.pydata.org/
* https://konlpy.org/
* https://www.tensorflow.org/?hl=ko
* https://huggingface.co/docs/transformers/index
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **Survey-Analysis**
### [Repositories](https://github.com/heokwon/Survey-analysis.git)
### Introduction
* 설문조사 분석
* 데이터를 쉽게 해석하기 위한 시각화 작업
* 데이터 분석을 위한 토픽모델링 및 LDA
<br><br>
### Data and Models
* 농협 설문조사 자료 - 외주
* TF-IDF, N-gram, LDA
<br><br>
### Envs and Requirements
* Google Colab, Jupyter Notebook, Conda
* Pandas, Matplotlib, Seaborn, Gensim, WordCloud, Re, KoNLPy
<br><br>
### Progress
* 데이터프레임 정제
* 원하는 데이터만을 추출하여 시각화
* TF-IDF를 사용하여 단어 빈도수를 추출한 뒤 워드클라우드를 이용하여 시각화
* TF-IDF, LDA를 통한 토픽 모델링
* 바이그램 / 트라이그램 적용
* 응집도 / 복잡도 기준 최적 에폭 및 토픽 갯수 설정
* 설몬조사 플랫폼에 사용할 목적으로, 비슷한 양식의 또다른 설문조사 데이터를 집어넣어도   
함수 한줄로 plot, wordcloud가 가능하도록 함수화 진행
<br><br>
### Referece
* https://pandas.pydata.org/
* https://konlpy.org/
* https://radimrehurek.com/gensim/
* https://scikit-learn.org/
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **Dog_Classification**
### [Repositories](https://github.com/heokwon/Dog_Classification.git)
### Introduction
14종의 강아지 품종 이미지를 분류하는 모델 만들기<br>
|Species|Label|Species|Label|
|:-------:|:-----:|:-------:|:-----:|
|코카스파니엘|1|삽살개|8|
|푸들|2|시베리안 허스키|9|
|그레이하운드|3|말라뮤트|10|
|말티즈|4|닥스훈트|11|
|퍼그|5|웰시코기|12|
|비숑|6|리트리버|13|
|진돗개|7|포메라니안|14|<br>
<br>

### Data and Model
* Data
  + Selenium을 통해 Naver와 Google에서 크롤링
  + 총 14개의 label
  + 11,289장 
* Model
  + ResNet50
  + Transfer learning
    - conv layer 일부 재학습(12층, 15층, 30층) + 분류기 학습
    - Dropout
  + Fine-tuning
<br><br>
### Envs and Requirements
* Google Colab, VScode
* Tensorflow, Selenium, Pickle, OpenCV, Matplotlib, Numpy, Sklearn
<br><br>
### Progress
* Selenium을 통해 Naver와 Google에서 총 14품종의 강아지 이미지를 크롤링
  + 팀원 한명당 2개 이상의 품종 크롤링
  + 평균적으로 label당 100장의 이미지 크롤링, 총 11,289장
* 이미지 전처리
  + Resize
  + Zero-centering
  + Gray scale
* 크롤링한 이미지를 Pickle로 저장
  + 14개의 pickle을 병합
* 학습
  + model : ResNet50
  + conv layer 일부 재학습 + 분류기 학습
  + fine - tuning
<br><br>
### Results 
||150x150, 10e|200x200, 10e|220x220,8e|
|:--:|:-------:|:----------:|:--------:|
|train accuracy|0.9260|0.9387|0.9312|
|test accuracy|0.8136|0.8441|0.8539|
|after fine-tuning||0.8565||<br>   
<br>   

* table 1) 이미지 사이즈 변화에 따른 성능 변화(gray scale, conv layer 15층 + 분류층 학습)
  + 이미지가 커짐에 따라 확실히 정확도가 높아지는 것을 확인
  + RAM에 부담이 안되는 크기에 정확도가 가장 컸던 200x200, 10e 만 fine-tuning진행
  + accuracy가 조금 상승되는 것을 확인할 수 있음   
 <br>   

||12층, Dropout(0.5) 1번|12층, Dropout(0.5) 2번|30층, Dropout(0.5) 1번|
|:--:|:----------------:|:--------------------:|:--------------------:|
|train accuracy|0.7837|0.9633|0.9922|
|test accuracy|0.8667|0.8940|0.8887|
|after fine-tuning|0.8760|0.8920|0.8920|<br>   
<br>   

* table 2) Model 수정에 따른 성능 변화(data > int화, 180x180, Gray scale X)
  + Color image 사용(Gray scale X)
  + Drop out, conv layer 학습하는 층수를 변경
  + 최대 0.89까지 상승
<br><br>
### Reference
* [https://selenium-python.readthedocs.io/index.html](https://selenium-python.readthedocs.io/index.html)
* [https://keras.io/api/applications/resnet/](https://keras.io/api/applications/resnet/)
* [https://docs.python.org/ko/3/library/pickle.html](https://docs.python.org/ko/3/library/pickle.html)
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **DietService-Objectdetection**
### [Repositories](https://github.com/heokwon/DietService-ObjectDetection.git)
### Introduction
* 현재 자기관리가 하나의 트랜드처럼 자리잡고있음

* 운동만큼 식단의 중요성이 부각됨

* 핸드폰 카메라 하나만으로 식단관리가 가능한 서비스가 있으면 어떨까 하는 생각에 시작한 프로젝트

* Object Detection을 통해 음식을 인식하고 칼로리와 영양정보를 제공
<br><br>
### Data and Models
* YOLO V4

* 12,000개의 한식, 일식, 양식 이미지 데이터

* 총 100개의 class

* multiple food-item, single food-item으로 구성
<br><br>
### Envs and Requirements
* Google Colab, VScode
* Tensorflow, Glob, Shutil, Pillow, OpenCV, Darknet
<br><br>
### Progress
* 데이터셋 구축

* 데이터셋 정제   
이미지 한 폴더에 모으기 - 각각의 클래스 폴더별로 이미지가 나뉘어있음   
이미지 사이즈 정규화 - bbox 좌표 정규화

* Train & Inference
<br><br>
### Referece
* https://arxiv.org/abs/2004.10934
* https://github.com/AlexeyAB/darknet
* https://opencv.org/
* https://pillow.readthedocs.io/en/stable/
* https://www.tensorflow.org/?hl=ko
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **Car Damage Detection**
### [Repositories](https://github.com/heokwon/CV-CarDamageDetection.git)
### Introduction
* Semantic Segmentation을 이용한 자동차 파손부위 탐지   

* 사람이 직접 파손부위를 하나하나 검수해야 하는 부담을 덜 수 있고, 회사 입장에서도   
인적,시간적 자원 절약 측면에서 좋을 것이라 생각하여 진행하게 된 프로젝트

* 사진이나 영상 속 객체를 pixel단위로 탐지하여 object detection보다 세부적으로   
detecting이 가능한 Semantic Segmentation을 선택
<br><br>
### Data and Models
* AI-hub에 socar dataset이 올라오기 이전   

* 구글링을 통하여 segmentation annotation이 포함된 차량파손이미지 수집   

* via tool - 부족한 데이터셋 보충, 좀 더 세밀한 mask를 통해 성능개선을 기대   
차량 파손 이미지 COCOdataset을 사용, via tool을 사용하여 이미지에 polygon을 직접 달아줌으로써   
mask의 좌표를 생성 후 annotation.json 파일 생성   
2명의 팀원이서 1일동안 총 400장의 데이터셋 생성   

* 차량 파손 부위가 아닌 파손 "형태"를 detecting하는 작업으로, 차량 이외에도 스크래치나 이격과   
같은 파손형태 데이터셋도 사용   

* Augmentation - Pytorch의 albumentation을 사용하여 offline으로 데이터증식 진행   
HorizontaFlip, VerticalFlip, Blur, OpticalDistortion, Resize, RandomRotate90

* Binary 와 Multi 로 진행   
Binary Label : background - 0 , damaged - 1   
Multi Label : background - 0 , scratch - 1 , dent - 2 , spacing - 3 , broken - 4
<br><br>
### Envs and Requirements
* Semantic Segmentation에서 가장 많이 쓰이는 모델 선정   

* DeepLabV3   
reference를 git clone하여 하이퍼파라미터 변경 및 inference추가
pre-trained model에 fine-tuning

* Unet   
reference를 git clone하여 하이퍼파라미터 변경 및 inference추가 논문내용을 직접 구현하여 사용   
<br><br>
### Envs and Requirements
* Google Colab, VScode, AWS
* Pytorch, Pillow, OpenCV, Numpy, Matplotlib, via, albumentation, Weights and Biases
<br><br>
### Progress
* 데이터셋 구축 - 구글링, via프로그램사용하여 직접만들기

* 데이터셋 정제   
annotation info가 담겨있는 json파일을 이용하여 polygon2mask진행   
확장자를 jpg에서 png로 바꾸기   
binary형태의 데이터셋에서 class별로 array값을 다르게 부여햐여 multi dataset구축   
unet에서 사용하기 위해 img형식의 mask.png를 array로 바꿔 mask.npy로 변경   
split-folders를 사용하여 폴더안의 파일들을 train-set과 valid-set으로 나눔   

* 데이터셋 증식   
albumentation을 이용하여 오프라인에서 augmentation진행   
HorizontaFlip, VerticalFlip, Blur, OpticalDistortion, Resize, RandomRotate90   

* DeepLabV3 & Unet Reference 찾기

* DeepLabV3 Reference 튜닝, 최적의 hyperparameter찾기, pre-trained 모델에 fine-tuning 시키기

* Weights and Biases를 연동하여 train-log 관리

* Unet 논문 및 유투브 참고하여 직접구현 후 학습 진행

* Label별로 학습시킨 후 ensemble 시도
<br><br>
### Referece
* via tool : https://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html
* DeepLabV3 : https://github.com/msminhas93/DeepLabv3FineTuning.git
* Unet : U-Net: Convolutional Networks for Biomedical Image Segmentation
* albumentation : https://albumentations.ai/
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## **Kaggle Competition : HuBMAP + HPA - Hacking The Human Body**
### Repositories - [Train & Inference](https://github.com/heokwon/KaggleCompetiton-Train-and-Inference.git), [Data Handling](https://github.com/heokwon/KaggleCompetiton-DataHandling.git)
### Introduction
* Semantic Segmentation으로 HuBMAP 데이터셋을 학습하여 FTU를 찾는 대회
* 점수를 높이기 위한 Dataset Handling
* encoder를 EfficientNet과 ResNeSt를 사용하는 Unet의 Train
* 학습시킨 데이터셋의 size, encoder, model간의 ensemble과 학습한 모델을 Inference
<br><br>
### Data and Models
* HPA에서 제공한 3000x3000 size의 train image 351장
* class : kidney, prostate, largeintestine, spleen, lung
* EfficientNet (b1 - b5), ResNeSt (101, 200, 269), Unet
<br><br>
### Envs and Requirements
* Google Colab, VScode, AWS, Jupyter notebook
* Pandas, Pytorch, Fast-Ai, MMSegmentation, Pillow, OpenCV, Imageio, Matplotlib, Rasterio, Sklearn, Weights and Biases,  
<br><br>
### Progress
* mmsegmentation
* Dataset Handling   
1. rle to mask - train set의 이미지가 3000x3000으로, 메모리가 매우 큼   
메모리를 줄이기 위해 mask좌표를 rle로 표현   
rle : 마스크 이미지의 array정보가 0과 1로만 표현되어있는 상태에서, 마스크값인 1의 시작위치와 끝 위치, 그 다음 1의 시작위치와   
끝 위치를 반복해서 나타냄으로써 메모리를 줄이는 방법   

2. Convert - 해상도 손실 없이 학습시키는 이미지의 사이즈를 줄이기 위해 원본데이터를 자름   
reduce값을 설정해 원본이미지를 resize시키고 설정한 size만큼 convert하는데, reduce값에 따라 생기는 패딩의 크기가 다름   
패딩이 최소로 생기는 reduce값의 데이터셋 ( 256x256 reduce 4, 6, 12 / 512x512 reduce 2, 3, 6 )생성
stride를 추가하여 convert할 때 좌푝값에 보폭을 추가, 샘플 수 도 늘리고 중첩되는 ground truth 가 많아짐   
stride가 있는 데이터셋을 학습시켰을 때 성능이 훨씬 좋음   
256x256의 경우, stride값을 128과 64로 설정한 뒤 데이터셋을 구축해봄   
stride 128 - 10943개 / stride 64 - 34412개   

3. 예측해야하는 test 이미지의 크기가 150x150 - 4500x4500 으로 매우 다양함   
다양한 크기의 test 이미지를 좀 더 잘 예측하기 위해, 학습시킬 데이터셋의 크기를 다양하게 만든 후 하나의 데이터셋으로 구축   
256x256 multi scale dataset - reduce 4, 6, 12의 이미지를 하나의 데이터셋으로 구축
512x512 multi scale dataset - reduce 2, 3, 6의 이미지를 하나의 데이터셋으로 구축   

4. binary class와 multi class 둘 다 진행하기 위하여, binary클래스인 데이터셋을 클래스별로 이미지를 추출하여 label을 부여   
kidney - 1 , prostate - 2 , largeintestine - 3 , spleen - 4 , lung - 5

* Modeling   
1. Efficient를 encoder로 사용하는 Unet   

2. b0 - b7까지 성능실험 / 256, 512, 768 사이즈로 진행   
b1, b3 에서 256x256 multi scale with stride 128 (10948개)데이터셋의 성능이 가장 뛰어남   
b5 에서는 256x256 multi scale with stride 64 (34412개)데이터셋의 성능이 가장 뛰어남   
학습 성능 자체는 stirde값이 128인 데이터셋이 더 좋아 보이나, 모델복잡도가 매우 큰 b5의 경우 더 많은 샘플수를 가진 stride 64   
데이터셋에서 성능이 더 좋았음   

3. kfold를 사용하여 교차검증을 진행한 후, inference에서 stacking ensemble 진행   

4. train code는 Fast-Ai를 사용하여 모델의 head train과 전체적인 full train을 진행   

* Inference tuning   
1. test이미지를 prediction할 때, size와 reduce를 입력하여 원하는 사이즈의 타일로 나눠 예측할 수 있음   
size와 reduce값을 바꿔가며 inferece를 진행한 결과, size = 512 / reduce = 3 / threshold = 0.225 일 때 성능이 가장 좋음   

2. 이미지 array의 mean값과 std값을 변경해가며 가장 성능이 좋은 값을 찾음   

3. 테스트이미지를 전처리 하는 과정에서 ratio값을 추가해 여러개의 타일로 나눠 예측하던 방식을 하나의 타일로 예측하도록 바꿈   

* Ensemble   
1. 다양한 크기의 test set을 보다 더 잘 예측하기 위해 다양한 사이즈로 학습시킨 모델들로 stacking ensemble 진행 - 점수상승폭이 좋음   

2. EfficientNet에서 점수가 가장 좋았던 b1, b3, b5의 encoder model끼리 앙상블   

3. encoder를 ResNeSt101, 200, 269로 바꾸어 학습한 파일들도 추가하여 앙상블   
EfficientUnet b1, b3, b5 256x256, 512x512, 768x768 + UneSt(ResNeSt + Unet)101, 200, 269 256x256, 512x512
<br><br>
### Result
* Private Score : 0.76 (Final Result), Public Score : 0.78   
<img width="571" alt="kaggle competiton score" src="https://user-images.githubusercontent.com/106142393/193986860-e9300d10-9d97-4342-94a1-55bd3905df4f.PNG">   

* Rank : 124 / 1245 teams (90 percentile)   
<img width="571" alt="kaggle score" src="https://user-images.githubusercontent.com/106142393/193986925-74e0c59b-fa8b-4625-a90f-002252503e9b.PNG">   

<br>

### Referece
* https://www.kaggle.com/code/befunny/hubmap-fast-ai-starter-efficientnet
* https://www.kaggle.com/code/shuheiakahane/inference-hubmap-fast-ai-starter-efficientnet
* https://github.com/twyunting/Laplacian-Pyramids
* https://www.kaggle.com/code/nghihuynh/data-augmentation-laplacian-pyramid-blending
* https://www.kaggle.com/code/alejopaullier/how-to-create-a-coco-dataset
* https://github.com/Mr-TalhaIlyas/Mosaic-Augmentation-for-Segmentation
* https://www.kaggle.com/code/thedevastator/converting-to-256x256
* https://www.kaggle.com/code/e0xextazy/multiclass-dataset-768x768-with-stride-for-mmseg
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
