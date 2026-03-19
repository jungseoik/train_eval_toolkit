# 데이터 수집 레거시 코드 정리

## 작업 일자
2026-03-19

## 작업 목적
데이터 수집 파트의 레거시 코드를 삭제하고 README에서 해당 내용을 제거하여 코드베이스를 간소화한다.

## 삭제된 파일

| 파일/디렉토리 | 설명 |
|---|---|
| `src/preprocess/gj/` | 강진 데이터 전처리 (gj_split) - 전처리 완료, 미사용 |
| `src/preprocess/aihub/` | AIHub 매장 데이터 전처리 (aihub_store_split) - 전처리 완료, 미사용 |
| `src/train_test_split_folder.py` | main.py 미통합 독립 스크립트, 레거시 |
| `scripts/utils/video_split_gj.sh` | gj_split 호출 셸 스크립트 |
| `scripts/utils/video_split_aihub_store.sh` | aihub_store_split 호출 셸 스크립트 |
| `scripts/utils/json_train_test_split.sh` | train_test_split_folder.py 호출 셸 스크립트 |

## 유지된 파일 (활성 코드)

| 파일 | 이유 |
|---|---|
| `src/preprocess/video_splitter.py` | `main.py preprocess`에서 사용 |
| `src/preprocess/train_test_split.py` | `main.py train_test_split`에서 사용 |

## 수정된 파일

### main.py
- `gj_split` 관련 import, 함수, argparse 서브커맨드 제거
- `aihub_store_split` 관련 import, 함수, argparse 서브커맨드 제거

### README.md
- mermaid 플로우차트에서 `데이터 수집` 노드 및 `video split / folder split` 제거
- 도구 매핑 테이블에서 데이터 수집 행 삭제
- "단계별 실행 가이드" 중 "1) 데이터 수집/정리(레거시)" 섹션 전체 삭제
- 섹션 번호 재정렬 (2~6 → 1~5)

## 검증 결과
- `main.py` 내 `gj_split`, `aihub_store_split`, `train_test_split_folder` 참조 없음 확인
- `README.md` 내 해당 레거시 참조 없음 확인
