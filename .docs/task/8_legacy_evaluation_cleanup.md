# Task 8: 레거시 모듈 정리

## 개요
Task 1~7 리팩토링 이후 미사용/레거시 파일 17개를 식별하여 삭제. 코드베이스 경량화 및 유지보수성 향상.

## 작업 일자
2026-03-21

## 삭제 파일 목록 (15개 파일 + launch.json 설정 2건)

### A. 완전 미사용 파일 (11개)

| 경로 | 삭제 사유 |
|------|-----------|
| `configs/config_auto_labeling.py` | YAML 전환으로 미사용 |
| `configs/config_gj_label.py` | 빈 파일 |
| `configs/constants.py` | APP_PROMPT만 정의, 어디서도 import 없음 |
| `src/utils/csv_utils.py` | broken import (존재하지 않는 경로 참조) |
| `src/utils/data_profiler.py` | 미사용 |
| `src/preprocess/labeling_google_sheet.py` | NAS 절대경로 하드코딩, 미사용 |
| `src/evaluation/evaluate_caption.py` | 프로젝트 범위 외 (캡셔닝) |
| `src/evaluation/evaluate_pia_hf_bench.py` | eff 버전으로 대체됨 |
| `src/evaluation/evaluate_qualitative_video.py` | threshold_image 버전으로 대체 |
| `src/evaluation/evaluate_qualitative_video_violence.py` | threshold_image로 통일 |
| `src/evaluation/evaluate_qualitative_video_violence_batch.py` | threshold_image가 최종 버전 |

### B. 스크립트 참조 레거시 (4개)

| 경로 | 삭제 사유 |
|------|-----------|
| `src/evaluation/evaluate_video_classfication_edit_error.py` | edit의 이전 버전 (70%+ 중복) |
| `src/evaluation/evaluate_video_classfication_multi.py` | edit의 단순 버전 (고정 프롬프트) |
| `scripts/eval/eval_sequence/video_eval_cctv_set.sh` | edit_error 참조 |
| `scripts/eval/video_cls_multi_eval.sh` | multi 참조 |

### C. 부수 정리

| 경로 | 변경 내용 |
|------|-----------|
| `.vscode/launch.json` | 삭제된 파일 참조 디버그 설정 2건 제거 |

## 유지 대상 (건드리지 않음)

- **활성 평가 파일**: `evaluate_image_classfication.py`, `evaluate_video_classfication_edit.py`, `evaluate_qualitative_video_threshold_image.py`
- **HF 벤치마크**: `evaluate_pia_hf_bench_eff.py` + `video_hf_vio_eval.sh`
- **활성 config**: `config_gemini.py`, `config_preprocess.py`
- **Untracked 파일**: test*.py, config.json, sample.*, vllm 관련

## 검증 결과

| 검증 항목 | 결과 |
|-----------|------|
| `python main.py --help` | cv2 미설치로 인한 환경 에러 (삭제와 무관) |
| 삭제 파일 잔여 참조 grep | 참조 없음 확인 |
| 활성 평가 파일 3개 AST 파싱 | 모두 정상 |

## 커밋

- `cleanup: 미사용 레거시 모듈 17개 삭제` (56bac58)

## 효과

- 총 2,992줄 삭제
- 코드베이스 경량화로 탐색 및 유지보수 부담 감소
