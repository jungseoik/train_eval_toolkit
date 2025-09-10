**전처리 -\> 오토라벨링 -\> 학습 -\> 평가** 

### **프로젝트 구조**

```
/VLM-instruction-tuning
├── data/
│   ├── raw/
│   │   └── (원본 이미지 및 텍스트 데이터)
│   └── processed/
│       ├── images/
│       └── annotations.json
│
├── configs/
│   ├── preprocess.yaml
│   ├── autolabel.yaml
│   ├── train.yaml
│   └── evaluate.yaml
│
├── src/
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── image_processor.py
│   │   └── text_processor.py
│   │
│   ├── autolabel/
│   │   ├── __init__.py
│   │   ├── gpt_labeler.py
│   │   └── filtering.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py
│       └── logging_utils.py
│
├── scripts/
│   ├── run_preprocess.py
│   ├── run_autolabel.py
│   ├── run_train.py
│   └── run_evaluate.py
│
├── requirements.txt
└── README.md
```

-----

### **각 모듈 설명**

#### **📁 `data/`**

데이터를 관리하는 디렉토리입니다.

  * **`raw/`**: 원본 이미지, 텍스트 파일 등 가공되지 않은 데이터를 저장합니다.
  * **`processed/`**: 전처리 및 오토라벨링을 거친 가공된 데이터를 저장합니다. 예를 들어, 크기가 조정된 이미지나 instruction-following 형식으로 변환된 JSON 파일 등이 여기에 해당됩니다.

#### **📁 `configs/`**

프로젝트의 모든 설정을 관리하는 파일들을 모아둡니다. 각 단계별로 필요한 파라미터(예: 파일 경로, 모델 하이퍼파라미터, 배치 사이즈 등)를 YAML 또는 JSON 형식으로 저장하여 코드와 설정을 분리합니다.

  * **`preprocess.yaml`**: 이미지 크기, 정규화 방식 등 전처리 관련 설정을 정의합니다.
  * **`autolabel.yaml`**: 라벨링에 사용할 모델(예: GPT-4), 프롬프트 템플릿 등의 설정을 정의합니다.
  * **`train.yaml`**: 학습에 사용할 모델, 학습률, 에포크, 배치 사이즈 등의 하이퍼파라미터를 정의합니다.
  * **`evaluate.yaml`**: 평가 데이터셋 경로, 사용할 메트릭 등의 설정을 정의합니다.

#### **📁 `src/` (소스 코드)**

프로젝트의 핵심 로직이 담긴 파이썬 코드를 관리합니다. 각 기능별로 하위 패키지를 두어 구조화합니다.

  * **`preprocess/`**: 데이터 전처리 관련 모듈입니다.
      * `image_processor.py`: 이미지 리사이징, 정규화, 증강 등의 기능을 수행합니다.
      * `text_processor.py`: 텍스트 정제, 토크나이징, instruction 형식으로 변환하는 기능을 담당합니다.
  * **`autolabel/`**: 자동으로 라벨을 생성하는 모듈입니다.
      * `gpt_labeler.py`: GPT와 같은 외부 LLM을 활용하여 이미지나 텍스트에 대한 instruction 데이터를 생성합니다.
      * `filtering.py`: 생성된 데이터의 품질을 검사하고 부적절한 데이터를 필터링하는 로직을 포함합니다.
  * **`training/`**: 모델 학습 관련 모듈입니다.
      * `dataset.py`: 전처리된 데이터를 불러와 모델에 입력할 형태로 만들어주는 `Dataset` 및 `DataLoader` 클래스를 정의합니다.
      * `model.py`: 파인튜닝할 VLM 모델 아키텍처를 정의하거나 Hugging Face 같은 라이브러리에서 모델을 불러오는 코드를 작성합니다.
      * `trainer.py`: 실제 학습 루프(loop), 옵티마이저 설정, 손실 함수 계산, 체크포인트 저장 등의 기능을 포함하는 `Trainer` 클래스를 구현합니다.
  * **`evaluation/`**: 학습된 모델의 성능을 평가하는 모듈입니다.
      * `metrics.py`: BLEU, ROUGE, CIDEr 또는 VQA 정확도 등 VLM 성능 평가에 사용되는 지표를 계산하는 함수들을 정의합니다.
      * `evaluator.py`: 평가 데이터셋으로 모델의 예측을 생성하고, `metrics.py`의 함수를 이용해 최종 성능을 계산하고 결과를 로깅하는 클래스를 구현합니다.
  * **`utils/`**: 프로젝트 전반에서 공통으로 사용되는 유틸리티 함수들을 모아둡니다.
      * `io_utils.py`: 파일 읽기/쓰기, 경로 관리 등 입출력 관련 함수를 포함합니다.
      * `logging_utils.py`: 학습 및 평가 과정의 로그를 기록하는 함수를 정의합니다.

#### **📁 `scripts/`**

각 파이프라인 단계를 실행하는 스크립트 파일입니다. 이 스크립트들은 `configs/`의 설정 파일을 읽어와 `src/`에 구현된 기능들을 실행하는 역할을 합니다.

  * **`run_preprocess.py`**: 데이터 전처리 파이프라인을 실행합니다.
  * **`run_autolabel.py`**: 데이터 자동 라벨링을 실행합니다.
  * **`run_train.py`**: 모델 학습을 시작합니다.
  * **`run_evaluate.py`**: 학습된 모델의 성능 평가를 진행합니다.

#### **📄 기타 파일**

  * **`requirements.txt`**: 프로젝트에 필요한 파이썬 라이브러리 및 버전 정보를 명시합니다. (`pip install -r requirements.txt`로 한번에 설치 가능)
  * **`README.md`**: 프로젝트에 대한 설명, 설치 방법, 사용법 등을 기록하는 문서입니다.

-----

### **구조의 장점**

  * **모듈성 및 재사용성**: 각 기능이 독립된 모듈로 분리되어 있어 코드 이해가 쉽고, 다른 프로젝트에서도 재사용하기 용이합니다.
  * **확장성**: 새로운 기능(예: 새로운 모델, 다른 데이터셋)을 추가할 때 기존 구조에 쉽게 통합할 수 있습니다.
  * **유지보수**: 코드와 설정이 분리되어 있어 하이퍼파라미터 변경이나 경로 수정 시 코드 전체를 수정할 필요가 없습니다. 또한, 각 기능의 책임이 명확하여 디버깅이 편리합니다.

이 구조를 바탕으로 GitHub 레포지토리를 채워나가시면 체계적이고 효율적인 VLM instruction 튜닝 프로젝트를 진행하실 수 있을 겁니다.