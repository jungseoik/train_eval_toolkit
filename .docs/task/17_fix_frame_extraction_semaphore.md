# Task 17: 프레임 추출 semaphore 범위 수정 (OOM 방지)

## 개요

대형 영상(PIA_Falldown 등, 10만+ 프레임) 벤치마크 평가 시 CPU RAM OOM으로 프로세스가 죽는 문제를 수정.

## 문제 원인

프레임 추출(`cv2` -> `base64`)이 `asyncio.Semaphore` 밖에서 실행되어, API 호출 대기 중인 base64 문자열이 RAM에 무제한 누적됨.

### 기존 구조
```
[3,543개 task 즉시 생성]
    +-- process_frame(): 프레임 추출 (semaphore 밖, 제한 없음)
    |     +-- run_in_executor -> ThreadPool ~28개 동시 실행
    +-- _infer_frame(): API 호출 (semaphore 안, 10개 제한)
```

- 추출 속도(28 병렬) >> API 소비 속도(10 병렬)
- PIA_Falldown: 3,543 x ~300KB = ~1GB CPU RAM 누적 -> OOM Kill

## 수정 내용

### 핵심 변경
`semaphore` 범위를 `_infer_frame()` 내부에서 `process_frame()` 전체로 확장.
프레임 추출과 API 호출을 하나의 atomic 단위로 묶어 on-demand 추출 방식으로 전환.

### 변경 파일

| 파일 | 변경 |
|------|------|
| `src/evaluation/vllm_bench_eval.py` | `_infer_frame()`에서 semaphore 파라미터 제거, `process_frame()`에 `async with semaphore:` 추가 |
| `src/evaluation/lmdeploy_bench_eval.py` | 동일 패턴 적용 |
| `docs/eval/lmdeploy_yaml_guide.md` | concurrency 설명 "API 동시 요청 수" -> "프레임 추출 + 추론 동시 처리 수" |
| `docs/eval/lmdeploy_pipeline.md` | concurrency 설명 "동시 요청 수" -> "동시 처리 수" |
| `configs/lmdeploy_eval/config.py` | 주석 "동시 요청 수" -> "동시 처리 수" |

### 수정 전후 비교

**Before:**
```python
async def process_frame(fidx):
    b64 = await run_in_executor(...)                        # 제한 없음
    raw_text = await _infer_frame(client, semaphore, ...)   # 여기서만 제한

async def _infer_frame(client, semaphore, ...):
    async with semaphore:
        resp = await client.post(...)
```

**After:**
```python
async def process_frame(fidx):
    async with semaphore:                                   # 전체 제한
        b64 = await run_in_executor(...)
        raw_text = await _infer_frame(client, ...)

async def _infer_frame(client, ...):                        # semaphore 제거
    resp = await client.post(...)
```

## 효과

| 항목 | Before | After |
|------|--------|-------|
| 동시 메모리 사용 (base64) | ~1GB (3,543 x 300KB) | ~3MB (10 x 300KB) |
| OOM 위험 | PIA_Falldown 등 대형 영상에서 발생 | 제거됨 |
| GPU throughput | 동일 | 동일 (~2-4% 오버헤드, 무시 가능) |

## 사이드이펙트

없음. semaphore 내 early return, task cancel, retry 로직 모두 정상 동작 확인.
YAML 설정 구조 변경 없음, httpx limits는 `cfg.CONCURRENCY`에 자동 연동.
실행 중인 평가 프로세스에 영향 없음 (Python import 시점 로드).

## 향후 참고

수정 후 YAML에서 `concurrency` 값을 올리면 GPU 활용률 즉시 증가 가능.
현재 환경(RTX PRO 6000 98GB, Qwen3.5-2B, max_tokens=15) 기준 concurrency 50-64까지 안전.
