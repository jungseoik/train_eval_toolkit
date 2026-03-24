# data_check - JSON 라벨 카테고리 분포 점검

하위 폴더를 재귀 탐색하며 JSON 라벨 파일의 카테고리 분포를 점검합니다.

## 사용법

```bash
# JSON 라벨 점검
python main.py data_check -i data/processed/gangnam -t json

# 낮은 비율 기준 변경 (기본 49%)
python main.py data_check -i data/processed/gangnam -t json --threshold 0.1

# JSONL 점검
python main.py data_check -i data/instruction/train/train.jsonl -t jsonl
```

## 옵션

| 옵션 | 필수 | 설명 |
|---|---|---|
| `-i`, `--input` | O | 점검할 디렉토리(json) 또는 파일 경로(jsonl) |
| `-t`, `--type` | O | `json`: 라벨 디렉토리 점검, `jsonl`: 어노테이션 파일 점검 |
| `--threshold` | X | 낮은 비율 카테고리 기준값 (기본: 0.49) |

## 점검 항목 (json 모드)

폴더별 `category` 값의 비율을 계산하고, 기준(기본 49%) 미만인 카테고리를 경고 표시합니다. 데이터 불균형을 사전에 파악할 수 있습니다.

## 출력 예시

```text
============================================================
  JSON 라벨 점검 시작: /path/to/data/processed/gangnam
============================================================

  [yeoksam2_v2/Train/video/violence/clip] JSON 120개
    violence       : 98개 (81.7%)
    normal         : 22개 (18.3%) (* 낮은 비율)

============================================================
  점검 결과 요약
============================================================

  총 JSON 파일: 120개

  [낮은 비율 카테고리] (기준: < 49%)
    카테고리: NORMAL (22개)
      폴더: yeoksam2_v2/Train/video/violence/clip - 22개

============================================================
  점검 완료
============================================================
```

## 관련 파일

- 구현: `src/data_checker/stats/json_checker.py`
- JSONL 점검: `src/utils/jsonl_inform_check.py`
