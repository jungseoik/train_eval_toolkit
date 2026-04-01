# 평가 파이프라인 메모리 분석

2026-04-01 기준, LMDeploy 벤치마크 평가 파이프라인에서 발생한 메모리 문제를 분석한 문서.

---

## 문제 현상

API 서버에서 벤치마크 평가를 장시간 실행하면 `systemd-oomd`에 의해 프로세스가 kill됨.

### 발생 이력

| 일시 | 메모리 사용량 | 결과 |
|------|-------------|------|
| 2026-03-24 08:27 | tmux scope 72.4GB | oom-kill |
| 2026-03-25 00:33 | tmux scope 86.1GB | oom-kill |
| 2026-04-01 13:43 | tmux scope 13.5GB | oom-kill (5개 tmux pane 동시 사망) |
| 2026-04-01 16:22 | tmux scope 18.7GB | oom-kill (API 서버 사망) |

### systemd-oomd kill 조건

systemd-oomd는 메모리 양이 아닌 **memory pressure**를 기준으로 kill 판단.

```
/etc/systemd/oomd.conf:
  DefaultMemoryPressureLimit=90%

user@1000.service:
  ManagedOOMMemoryPressure=kill
```

memory pressure는 "시스템이 메모리 확보를 위해 소비하는 시간 비율"이며,
page scan이 증가하면 pressure가 올라감.

서버 환경: RAM 125GB, Swap 0.

---

## 원인 분석

### 1. API 서버 (uvicorn) 메모리 누적

```
idle: 90MB → 평가 중: 7.6GB → 13GB → 18.7GB → oom-kill
```

**근본 원인: Python heap fragmentation.**

Python의 메모리 할당 구조:

```
Python 프로세스
  └→ pymalloc (Python 소규모 객체 관리)
      └→ glibc malloc (OS 메모리 할당)
          └→ brk / mmap (커널)
```

glibc의 `malloc`은 한번 확장한 heap을 OS에 잘 반환하지 않음.
중간에 작은 블록이 하나라도 남아있으면 그 앞의 큰 영역을 `brk`로 줄일 수 없기 때문.

평가 코드에서 비디오마다:
- `asyncio.run()` → 새 이벤트 루프 + 스레드풀 생성
- `httpx.AsyncClient` → 커넥션풀 할당
- `cv2` 프레임 추출 → 임시 버퍼 할당
- base64 인코딩 → 문자열 할당

이 모든 할당/해제가 반복되면서 heap에 fragmentation이 누적.
`gc.collect()` 호출해도 glibc가 OS에 메모리를 돌려주지 않음.

**검증 결과 (실제 lmdeploy 추론)**:

```
Soil_Falldown (77개 영상) × 3회 반복, concurrency=100

  회차    in-process (기존)
  1회     71 MB → 1,777 MB  (+1,706 MB)
  2회     1,777 → 1,881 MB  (+104 MB, 재사용)
  3회     1,881 → 1,830 MB  (안정)
  
  → 1,760 MB가 OS에 반환되지 않음
  → 벤치마크 13개 × 영상 수백~수천개 처리하면 18GB까지 증가
```

### 2. Docker (lmdeploy 서버) 메모리 누적

```
시작: ~30GB → 2시간 후: 76GB (60,000건 요청 처리)
```

**동일한 근본 원인: Python + glibc heap fragmentation.**

lmdeploy 서버 내부(PID 83, worker 프로세스)에서 요청마다:
- 이미지 디코딩 (PIL/cv2, CPU)
- 리사이즈/정규화 (CPU 텐서)
- vision encoder 입력 준비 (CPU 텐서)
- GPU로 전송 후 CPU 텐서 해제

이 과정의 임시 메모리가 heap에 fragmentation으로 누적.

**Docker 내부 메모리 분석**:

```
worker PID 83:
  VmRSS:    42,445,540 kB (40.5 GB)
  RssAnon:  41,861,944 kB (41.3 GB, 전부 heap)
  RssFile:     562,668 kB (0.5 GB)
  
  /proc/83/smaps 최대 영역:
  [heap]: 41,353,428 kB (41.3 GB)
  
  GPU VRAM: 42.3 GB (변동 없음)
  KV cache 사용률: 0.005%
```

GPU VRAM은 안정적이며, 시스템 RAM(heap)만 증가.
이는 **lmdeploy만의 문제가 아닌 Python + glibc 조합의 구조적 한계.**
vLLM, TGI 등 Python 기반 추론 서버도 장시간 운용 시 동일한 현상 발생 가능.

---

## 해결 방법

### 적용 완료: API 서버 subprocess 분리

각 벤치마크 평가를 `multiprocessing.Process(spawn)`으로 실행.
프로세스 종료 시 OS가 메모리를 100% 회수.

```
[변경 전]
API 서버 (1개 프로세스)
  └→ evaluate_benchmark(벤치1)  메모리 누적
  └→ evaluate_benchmark(벤치2)  메모리 더 누적
  └→ ... → 18GB → oom-kill

[변경 후]
API 서버 (90MB 유지)
  └→ subprocess(벤치1) → 종료 → 메모리 회수
  └→ subprocess(벤치2) → 종료 → 메모리 회수
  └→ ... → 부모는 항상 90MB
```

검증:
```
subprocess 방식, Soil_Falldown × 3회 반복:
  부모 RSS 증가: +0 MB (100% 절감)
```

변경 파일:
- `src/lmdeploy_pipeline/evaluator.py`
- `src/evaluation/lmdeploy_bench_eval.py`

### 미해결: Docker (lmdeploy 서버) 메모리 증가

Docker 컨테이너 내부의 lmdeploy 프로세스 메모리 증가는 제어 불가.
lmdeploy 자체 코드를 수정하지 않는 한 해결할 수 없음.

**현재 대응**:
- `cleanup_docker: true` 설정으로 전체 평가 완료 후 Docker 자동 정리
- API 서버 메모리 절감(18GB → 90MB)으로 Docker가 쓸 수 있는 여유 확보

**추가 대응 가능**:
- 벤치마크 간 Docker 재시작 (모델 로딩 2~3분 추가)
- `jemalloc` / `tcmalloc` 사용 (glibc malloc 대체, fragmentation 완화)
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 lmdeploy serve ...`
- 실서비스 환경: k8s memory limit + OOM 시 자동 재시작

---

## 시스템 메모리 구성 (참고)

평가 실행 중 125GB RAM 배분 (2026-04-01 모니터링 기준):

| 프로세스 | 메모리 | 비고 |
|---------|--------|------|
| Docker (lmdeploy worker) | 30~76GB | 요청 수에 비례하여 증가 |
| API 서버 (uvicorn) | 90MB~18GB | subprocess 적용 후 90MB 고정 |
| vscode-server | ~10GB | 상주 |
| claude | ~3.5GB | 상주 |
| bench_summary2.py | ~1.7GB | 상주 |
| buff/cache | 가변 | 380GB 벤치마크 데이터셋 읽기 |

---

## 참고 자료

- Python Memory Management: https://docs.python.org/3/c-api/memory.html
- glibc malloc fragmentation: `man 3 malloc`, `malloc_trim(3)`
- systemd-oomd: `man 8 systemd-oomd`, `/etc/systemd/oomd.conf`
- jemalloc: https://jemalloc.net/ (fragmentation에 강한 대안 allocator)
