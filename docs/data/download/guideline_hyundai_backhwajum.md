# Hyundai Backhwajum 데이터셋 — 업로드 & 다운로드 가이드

## 학습 전 필수 다운로드 순서

### Step 1 — InternVL3-2B 체크포인트를 `ckpts/`에 저장
```bash
# repo root 기준
mkdir -p ckpts/InternVL3-2B
hf download OpenGVLab/InternVL3-2B \
    --repo-type=model \
    --local-dir ckpts/InternVL3-2B
```

### Step 2 — Hyundai 데이터셋을 `data/processed/`에 저장
```bash
# repo root 기준
mkdir -p data/processed
hf download PIA-SPACE/Hyundai_backhwajum hyundai_backhwajum.tar.gz \
    --repo-type=dataset \
    --local-dir data/processed

tar -xf data/processed/hyundai_backhwajum.tar.gz -C data/processed

# 압축 해제 후 원본 압축 파일 삭제
rm -f data/processed/hyundai_backhwajum.tar.gz
```

---

## 데이터셋 구조

```
hyundai_backhwajum/
├── abb_hyundai/
├── dtro_hyundai/
├── hyundai_01_16_QA/
├── hyundai_01_27_QA/
├── hyundai_01_27_QA_hard_negative/
├── hyundai_ai/
├── hyundai_hard_negative_2st/
├── hyundai_hard_negative_2st_box/
├── hyundai_image_gen_ai_1st/
├── hyundai_image_gen_ai_only_sangrak/
├── hyundai_PoC_25camera_capture/
└── hyundai_PoC_5camera_gen_ai/

```

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
tar -cf - --exclude='./hyundai_video_macs_test' -C . hyundai_backhwajum \
    | pigz -1 -p 16 > hyundai_backhwajum.tar.gz
```
> `pigz`는 병렬 gzip 압축 도구 (level 1 = 최고속, 16코어 사용). 없으면 `sudo apt install pigz` 로 설치

### Step 4 — 업로드
```bash
hf upload PIA-SPACE/Hyundai_backhwajum hyundai_backhwajum.tar.gz \
    --repo-type=dataset
```

---

## 다운로드 & 압축 해제 (Step 2 상세)

### Step 1 — 압축 파일만 다운로드
```bash
mkdir -p data/processed
hf download PIA-SPACE/Hyundai_backhwajum hyundai_backhwajum.tar.gz \
    --repo-type=dataset \
    --local-dir data/processed
```
> 파일명을 직접 지정하면 README 등 불필요한 파일 없이 해당 파일만 받을 수 있음

### Step 2 — 압축 해제
```bash
# 일반 해제
tar -xf data/processed/hyundai_backhwajum.tar.gz -C data/processed

# 빠른 해제 (pigz 병렬 처리, 권장)
pigz -dc data/processed/hyundai_backhwajum.tar.gz | tar -xf - -C data/processed

# 일반/빠른 해제 중 하나 실행 후 압축 파일 삭제
rm -f data/processed/hyundai_backhwajum.tar.gz
```

### 압축 해제 후 결과 구조
```
data/processed/
└── hyundai_backhwajum/
    ├── abb_hyundai/
    ├── dtro_hyundai/
    ├── hyundai_01_16_QA/
    ├── hyundai_01_27_QA/
    ├── hyundai_01_27_QA_hard_negative/
    ├── hyundai_ai/
    ├── hyundai_hard_negative_2st/
    ├── hyundai_hard_negative_2st_box/
    ├── hyundai_image_gen_ai_1st/
    ├── hyundai_image_gen_ai_only_sangrak/
    ├── hyundai_PoC_25camera_capture/
    └── hyundai_PoC_5camera_gen_ai/
```

---

## 저장소 파일 목록 확인 (선택사항)

다운로드 전 저장소에 어떤 파일이 있는지 확인:
```bash
hf repo-files PIA-SPACE/Hyundai_backhwajum --repo-type=dataset
```
