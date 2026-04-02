# 평가 프로세스 설계 문서

LMDeploy / vLLM 벤치마크 평가 파이프라인의 프로세스 구조와 설계 배경을 정리한 문서.
두 파이프라인은 동일한 subprocess 분리 + Docker 재시작 아키텍처를 공유한다.

---

## 전체 프로세스 흐름도

```
사용자
  │
  ├─ CLI 실행: python -m src.lmdeploy_pipeline -c config.yaml
  │  └→ runner.py
  │
  └─ API 요청: curl -F 'file=@config.yaml' http://server:9000/pipeline/run/file
     └→ main.py (FastAPI/uvicorn)
        └→ pipeline_worker.py (SSE 스트리밍 + 백그라운드 스레드)

        ┌─────────────────────────────────────────────────┐
        │  공통 파이프라인 흐름                           │
        │                                                 │
        │  1. YAML 검증 + 설정 로드                       │
        │  2. 모델 확인 (없으면 HuggingFace 다운로드)     │
        │  3. Docker 컨테이너 기동 (LMDeploy 서버)        │
        │  4. LMDeploy 서버 준비 대기 (health check)      │
        │                                                 │
        │  5. 벤치마크 평가 루프                          │
        │     ┌─────────────────────────────────────┐     │
        │     │ 벤치마크 1                          │     │
        │     │  └→ [subprocess] 프로세스 생성      │     │
        │     │      └→ 영상 순차 처리              │     │
        │     │          └→ 프레임 병렬 추론        │     │
        │     │             (asyncio + Semaphore)   │     │
        │     │             concurrency=N 동시 요청 │     │
        │     │      └→ CSV 결과 저장               │     │
        │     │  └→ subprocess 종료 → 메모리 회수   │     │
        │     │  └→ Docker 재시작 (interval 설정)   │     │
        │     ├─────────────────────────────────────┤     │
        │     │ 벤치마크 2 (동일 구조)              │     │
        │     ├─────────────────────────────────────┤     │
        │     │ ...                                 │     │
        │     └─────────────────────────────────────┘     │
        │                                                 │
        │  6. 결과 제출 (Gradio 리더보드)                 │
        │  7. Docker 컨테이너 정리                        │
        └─────────────────────────────────────────────────┘
```

### 프레임 단위 처리 상세

```
비디오 1개 (예: 18,000 프레임, window_size=30 → 600 샘플)

asyncio 이벤트 루프 내부:
  Semaphore(concurrency)
    ├─ 프레임 0    → [스레드풀] cv2 추출 + JPEG + base64 → [async] httpx POST → 결과
    ├─ 프레임 30   → [스레드풀] cv2 추출 + JPEG + base64 → [async] httpx POST → 결과
    ├─ 프레임 60   → ...
    ├─ ...         (최대 concurrency개 동시 처리)
    └─ 프레임 17970

  → 샘플 프레임 예측값 수집
  → 인터폴레이션 (forward/backward)으로 전체 프레임 예측 생성
  → CSV 저장
```

### 프로세스 간 관계

```
API 서버 (uvicorn, 상주)
  │
  └→ 백그라운드 스레드 (threading.Thread)
      │
      └→ evaluator.run_evaluation()
          │
          ├→ [subprocess 1] evaluate_benchmark("ABB_Falldown")
          │    └→ asyncio.run() → 프레임 병렬 처리
          │    └→ 종료 → OS 메모리 회수
          │    └→ progress.json ←→ 부모 polling thread
          │
          ├→ Docker 재시작 (stop → start → wait_for_ready)
          │
          ├→ [subprocess 2] evaluate_benchmark("Coupang_Falldown")
          │    └→ ...
          │
          └→ ...

Docker 컨테이너 (system.slice, 별도 cgroup)
  ├→ PID 1: lmdeploy API 서버 (요청 라우팅)
  └→ PID 83: worker (vision 전처리 + LLM 추론)
```

---

## 설계 배경: 왜 이 구조인가

### 문제 1: API 서버 메모리 누적 → OOM Kill

초기 구조에서는 모든 벤치마크 평가를 API 서버 프로세스 내부에서 직접 실행했다.

```
[초기 구조]
API 서버 (1개 프로세스)
  └→ evaluate_benchmark(벤치1)  함수 호출
  └→ evaluate_benchmark(벤치2)  함수 호출
  └→ ... 메모리 계속 누적 → 18GB → systemd-oomd kill
```

Python의 메모리 할당자(glibc malloc)는 `free()` 후에도 heap을 OS에 반환하지 않는다.
비디오마다 `asyncio.run()` + `httpx.AsyncClient` + `cv2` 프레임 추출을 반복하면
heap에 fragmentation이 누적되어 메모리가 계속 증가한다.

13개 벤치마크(총 1,882개 영상)를 처리하면 API 서버 메모리가 18GB까지 증가하여
`systemd-oomd`에 의해 프로세스가 kill되었다 (2026-04-01, 4회 발생).

**해결: 벤치마크별 subprocess 분리.**
프로세스가 종료되면 OS가 메모리를 100% 회수하므로 누적이 원천 차단된다.

### 문제 2: Docker(lmdeploy) 서버 메모리 누적

API 서버 문제를 해결한 후에도 Docker 내부 lmdeploy 서버의 메모리가 계속 증가했다.
동일한 원인(Python heap fragmentation)으로, 60,000건 요청 처리 후
worker 프로세스의 heap이 41GB까지 증가하는 것이 확인되었다.

```
Docker 메모리 모니터링 (2026-04-01):
  19:00  30.7GB
  19:30  46.5GB
  20:31  59.7GB
  20:41  75.5GB
  21:11  76.3GB (평가 완료 시점)
```

이는 lmdeploy 내부 코드의 문제로, 외부에서 설정으로 해결할 수 없다.

**해결: 벤치마크 간 Docker 컨테이너 재시작.**
컨테이너를 재시작하면 lmdeploy 프로세스가 새로 시작되어 메모리가 초기화된다.

### 왜 subprocess + Docker 재시작 조합인가

| 계층 | 문제 | 해결 | 비용 |
|------|------|------|------|
| API 서버 | 18GB 누적 → OOM kill | subprocess 분리 | 벤치마크당 ~0.5초 |
| Docker (lmdeploy) | 76GB 누적 | 컨테이너 재시작 | 벤치마크당 ~15초 (모델 로딩) |

두 해결책 모두 **벤치마크 내부의 병렬 처리(concurrency)에 영향을 주지 않는다.**
속도 손실은 13개 벤치마크 기준 약 3분(전체 5시간 대비 1%)이다.

### 대안 검토

| 대안 | 검토 결과 |
|------|-----------|
| `gc.collect()` + `malloc_trim()` | 효과 불확실, fragmentation이 심하면 소용없음 |
| concurrency 낮추기 | 피크만 줄어들 뿐 누적은 동일, 속도 저하 |
| Docker memory limit | Docker가 죽으면 평가도 종료 — 목적에 부합하지 않음 |
| jemalloc 사용 | lmdeploy Docker 이미지 커스텀 필요, 효과 보장 불가 |
| swap 추가 | pressure 완화 효과 있지만 근본 해결 아님 |

---

## 설정 옵션 요약

```yaml
pipeline:
  docker_restart_interval: 1   # 벤치마크 N개마다 Docker 재시작
                                # 0: 비활성, 1: 매번(기본값, 가장 안정적)
                                # 속도 우선이면 2~3으로 설정 가능

evaluate:
  concurrency: 100             # 프레임 동시 처리 수 (subprocess 내부)
                                # 높을수록 빠르지만 subprocess 메모리 피크 증가
                                # subprocess 종료 시 회수되므로 누적 문제 없음
```

---

## 적용 범위

이 설계는 LMDeploy와 vLLM 파이프라인 **모두**에 동일하게 적용된다.

| 파이프라인 | evaluator | bench_eval | config |
|-----------|-----------|------------|--------|
| LMDeploy | `src/lmdeploy_pipeline/evaluator.py` | `src/evaluation/lmdeploy_bench_eval.py` | `src/lmdeploy_pipeline/config.py` |
| vLLM | `src/vllm_pipeline/evaluator.py` | `src/evaluation/vllm_bench_eval.py` | `src/vllm_pipeline/config.py` |

두 파이프라인 모두:
- 벤치마크별 subprocess 분리 (메모리 누적 방지)
- `docker_restart_interval` 설정 지원 (서버 메모리 초기화)
- 프레임 단위 병렬 처리 (asyncio + Semaphore)
- 파일 기반 progress IPC

---

## 관련 문서

- [메모리 분석 상세](memory_analysis.md) — OOM 발생 이력, 프로세스별 메모리 분석 데이터
- [LMDeploy 파이프라인 가이드](lmdeploy_pipeline.md) — 사용법, 사전 준비, 파일 구조
- [vLLM 파이프라인 가이드](vllm_pipeline.md) — 사용법, 사전 준비, 파일 구조
- [YAML 설정 가이드 (LMDeploy)](lmdeploy_yaml_guide.md) — 설정 필드 설명
- [평가 API 가이드](pipeline_api.md) — API 서버 사용법, SSE 이벤트
