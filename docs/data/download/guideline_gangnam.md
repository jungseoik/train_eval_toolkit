# Gangnam 데이터셋 — 업로드 & 다운로드 가이드

## 학습 전 필수 다운로드 순서

### Step 1 — InternVL3-2B 체크포인트를 `ckpts/`에 저장
```bash
# repo root 기준
mkdir -p ckpts/InternVL3-2B
hf download OpenGVLab/InternVL3-2B \
    --repo-type=model \
    --local-dir ckpts/InternVL3-2B
```

### Step 2 — Gangnam 데이터셋을 `data/processed/`에 저장
```bash
# repo root 기준
mkdir -p data/processed
hf download PIA-SPACE/Gangnam_elevator gangnam.tar.gz \
    --repo-type=dataset \
    --local-dir data/processed

tar -xf data/processed/gangnam.tar.gz -C data/processed

# 압축 해제 후 원본 압축 파일 삭제
rm -f data/processed/gangnam.tar.gz
```

---

## 데이터셋 구조

```
gangnam/
├── gaepo1/
│   ├── Test_dataset_gaepo1st/     32M
│   └── Train_dataset_gaepo1st/    1.7G
├── gaepo1_v2/
│   ├── Test/                      338M
│   └── Train/                     3.0G
├── gaepo4/
│   ├── Test/                      129M
│   └── Train/                     820M
├── samsung/
│   ├── Test/                      43M
│   └── Train/                     299M
├── yeoksam2/
│   ├── Test_dataset_yeoksam2st/   4.0M
│   └── Train_dataset_yeoksam2st/  169M
└── yeoksam2_v2/
    ├── Test/                      23M
    └── Train/                     197M
```

> 총 용량: 약 6.6G
> QA 폴더는 압축에서 제외됨

---

## HuggingFace 업로드

### Step 1 — hf CLI 설치
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Step 2 — 로그인
```bash
hf auth login
```
> 토큰은 https://huggingface.co/settings/tokens 에서 발급 (Write 권한 필요)

### Step 3 — 압축
```bash
cd data/processed/

tar -cf - \
  gangnam/gaepo1/Test_dataset_gaepo1st \
  gangnam/gaepo1/Train_dataset_gaepo1st \
  gangnam/gaepo1_v2/Test \
  gangnam/gaepo1_v2/Train \
  gangnam/gaepo4/Test \
  gangnam/gaepo4/Train \
  gangnam/samsung/Test \
  gangnam/samsung/Train \
  gangnam/yeoksam2/Test_dataset_yeoksam2st \
  gangnam/yeoksam2/Train_dataset_yeoksam2st \
  gangnam/yeoksam2_v2/Test \
  gangnam/yeoksam2_v2/Train \
  | pigz -1 -p 16 > gangnam.tar.gz
```
> `pigz`는 병렬 gzip 압축 도구 (level 1 = 최고속, 16코어 사용). 없으면 `sudo apt install pigz` 로 설치

### Step 4 — 업로드
```bash
hf upload PIA-SPACE/Gangnam_elevator gangnam.tar.gz \
    --repo-type=dataset
```

---

## 다운로드 & 압축 해제 (Step 2 상세)

### Step 1 — 압축 파일만 다운로드
```bash
mkdir -p data/processed
hf download PIA-SPACE/Gangnam_elevator gangnam.tar.gz \
    --repo-type=dataset \
    --local-dir data/processed
```
> 파일명을 직접 지정하면 README 등 불필요한 파일 없이 해당 파일만 받을 수 있음

### Step 2 — 압축 해제
```bash
# 일반 해제
tar -xf data/processed/gangnam.tar.gz -C data/processed

# 빠른 해제 (pigz 병렬 처리, 권장)
pigz -dc data/processed/gangnam.tar.gz | tar -xf - -C data/processed

# 일반/빠른 해제 중 하나 실행 후 압축 파일 삭제
rm -f data/processed/gangnam.tar.gz
```

### 압축 해제 후 결과 구조
```
data/processed/
└── gangnam/
    ├── gaepo1/
    │   ├── Test_dataset_gaepo1st/
    │   └── Train_dataset_gaepo1st/
    ├── gaepo1_v2/
    │   ├── Test/
    │   └── Train/
    ├── gaepo4/
    │   ├── Test/
    │   └── Train/
    ├── samsung/
    │   ├── Test/
    │   └── Train/
    ├── yeoksam2/
    │   ├── Test_dataset_yeoksam2st/
    │   └── Train_dataset_yeoksam2st/
    └── yeoksam2_v2/
        ├── Test/
        └── Train/
```

---

## 저장소 파일 목록 확인 (선택사항)

다운로드 전 저장소에 어떤 파일이 있는지 확인:
```bash
hf repo-files PIA-SPACE/Gangnam_elevator --repo-type=dataset
```
