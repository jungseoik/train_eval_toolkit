# 20. 평가 파이프라인 메모리 누수 해결 (subprocess 분리)

## 배경

API 서버에서 벤치마크 평가를 장시간 실행하면 Python 프로세스의 메모리가 지속적으로 증가하여
`systemd-oomd`에 의해 OOM kill이 발생하는 문제.

### 증상
- API 서버(uvicorn): idle 90MB → 평가 중 7.6GB → 18.7GB → OOM kill
- 13개 벤치마크(총 1,882개 영상) 평가 시 반복적으로 발생
- concurrency 값(10, 200)과 무관하게 발생

### 근본 원인
Python의 메모리 할당자(pymalloc)가 해제된 메모리를 OS에 반환하지 않는 heap fragmentation 문제.
- `asyncio.run()` 이 비디오마다 새 이벤트 루프 + 스레드풀 생성
- `httpx.AsyncClient`가 비디오마다 커넥션풀 할당
- `cv2` 프레임 추출 + base64 인코딩의 임시 버퍼
- 이 모든 것이 Python heap에 흩어져 OS에 반환 불가능한 fragmentation 발생

## 해결 방법

### 핵심: 벤치마크 단위 subprocess 분리

`evaluator.py`의 `run_evaluation()`에서 각 벤치마크 평가를 `multiprocessing.Process`로 실행.
프로세스 종료 시 OS가 메모리를 100% 회수하므로 메모리 누적이 원천 차단됨.

```
[변경 전]
API 서버 (1개 프로세스)
  └→ evaluate_benchmark(벤치1)  ← 메모리 누적
  └→ evaluate_benchmark(벤치2)  ← 메모리 더 누적
  └→ ... → 18GB → OOM kill

[변경 후]
API 서버 (90MB 유지)
  └→ subprocess(벤치1) → 종료 → 메모리 회수
  └→ subprocess(벤치2) → 종료 → 메모리 회수
  └→ ... → 부모는 항상 90MB
```

### 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/lmdeploy_pipeline/evaluator.py` | `run_evaluation()`이 `multiprocessing.Process`로 각 벤치마크 실행 |
| `src/evaluation/lmdeploy_bench_eval.py` | `_update_video_progress()`에 파일 기반 progress 전달 추가 |

### 기술 세부사항

- `multiprocessing.get_context("spawn")` 사용 (스레드 내 fork 안전성)
- 진행도(progress)는 JSON 파일 기반 IPC로 전달
  - 자식: `_update_video_progress()`가 progress_file에 atomic write (임시파일 + rename)
  - 부모: polling thread가 1초마다 파일 읽어 `progress_state` dict 업데이트
- 결과(성공/실패/에러)도 JSON 파일로 전달
- 벤치마크 내부의 병렬 처리(concurrency)는 그대로 유지

## 검증 결과

### 메모리 테스트 (실제 cv2, KISA_Falldown 20개 영상)

```
기존 방식 (in-process):
  시작: 22 MB → 최종: 895 MB (증가: +873 MB, OS에 반환 안 됨)

subprocess 방식:
  부모: 항상 +0 MB (증가 없음)
  자식: 800~900 MB 사용 후 프로세스 종료 시 전부 회수

절감율: 100%
```

### 호환성 확인

| 항목 | 결과 |
|------|------|
| 벤치마크 내 병렬 처리 (concurrency) | 그대로 유지 (subprocess 안에서 동일하게 동작) |
| stdout/stderr 로그 출력 | 자식 프로세스 출력이 부모에 정상 전달 |
| /pipeline/status progress API | JSON 파일 IPC로 video/frame 진행도 정상 반영 |
| 기존 CLI 실행 (python -m) | 영향 없음 (evaluator.py만 변경, bench_eval.py는 하위호환) |
| pickle/spawn 호환성 | `_benchmark_worker`와 cfg_dict 모두 정상 pickle 가능 |
| 프로세스 비정상 종료 감지 | exitcode 확인 및 에러 메시지 전달 |

## 속도 영향

- 벤치마크 간 subprocess 시작 오버헤드: ~0.5초 × 13개 = ~6초 추가
- 전체 평가 시간(수 시간) 대비 무시 가능
- 벤치마크 내부 처리 속도: 변화 없음
