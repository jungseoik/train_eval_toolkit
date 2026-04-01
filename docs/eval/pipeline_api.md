# LMDeploy 평가 요청 API

외부에서 YAML 설정 파일을 전송하여 LMDeploy 벤치마크 평가를 요청하는 API 서버입니다.
SSE(Server-Sent Events)로 실시간 진행 상황을 확인할 수 있습니다.

- YAML 설정 작성법: [lmdeploy_yaml_guide.md](lmdeploy_yaml_guide.md)
- 파이프라인 사용법: [lmdeploy_pipeline.md](lmdeploy_pipeline.md)

---

## 서버 정보

| 항목 | 값 |
|------|-----|
| **주소** | `http://172.168.43.214:9000` |
| **네트워크** | 내부망 전용 |
| **동시 처리** | 한 번에 1개 (실행 중이면 거절) |

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/pipeline/run/file` | YAML 파일 업로드로 평가 시작 (SSE 응답) |
| POST | `/pipeline/run/yaml` | JSON body에 YAML 문자열로 평가 시작 (SSE 응답) |
| GET | `/pipeline/status` | 현재 파이프라인 상태 조회 |
| GET | `/health` | 서버 헬스체크 |

---

## curl 사용법

### 파일 업로드로 평가 요청

```bash
curl -N -F 'file=@config.yaml' http://172.168.43.214:9000/pipeline/run/file
```

`-N` 옵션은 버퍼링을 비활성화하여 SSE 이벤트를 실시간으로 출력합니다.

### JSON body로 평가 요청

```bash
curl -N -X POST \
  -H 'Content-Type: application/json' \
  -d '{"yaml_content": "pipeline:\n  name: \"Test\"\n  steps:\n    docker: true\n    evaluate: true\n    submit: true\n  cleanup_docker: true\n..."}' \
  http://172.168.43.214:9000/pipeline/run/yaml
```

### 상태 확인

```bash
curl http://172.168.43.214:9000/pipeline/status
```

응답 예시 (평가 진행 중):

```json
{
  "status": "evaluating",
  "pipeline_name": "InternVL3-2B Fire Benchmark (LMDeploy)",
  "started_at": "2026-03-30T14:30:00",
  "finished_at": null,
  "elapsed": null,
  "error": null,
  "hint": null,
  "result": null,
  "progress": {
    "total": 6,
    "completed": 1,
    "current": "SamsungCNT_Fire",
    "benchmarks": [
      {"name": "PIA_Fire", "status": "completed", "video": "6/6"},
      {"name": "SamsungCNT_Fire", "status": "in_progress", "video": "3/10", "frame": "120/642"},
      {"name": "Soil_Fire", "status": "pending"},
      {"name": "Hyundai_Fire", "status": "pending"},
      {"name": "Coupang_Fire", "status": "pending"},
      {"name": "Kumho_Fire", "status": "pending"}
    ]
  }
}
```

`progress` 필드는 평가 진행 중(`status: "evaluating"`)일 때만 값이 있으며, 유휴 상태(`idle`)에서는 `null`입니다.

| progress 필드 | 설명 |
|---------------|------|
| `total` | 전체 벤치마크 수 |
| `completed` | 완료된 벤치마크 수 (completed + failed + skipped) |
| `current` | 현재 진행 중인 벤치마크 이름 |
| `benchmarks[].status` | `pending` / `in_progress` / `completed` / `failed` / `skipped` |
| `benchmarks[].video` | 현재 영상 진행도 (예: `"3/10"`) |
| `benchmarks[].frame` | 현재 프레임 진행도 (예: `"120/642"`, in_progress일 때만) |

### 헬스체크

```bash
curl http://172.168.43.214:9000/health
```

```json
{"status": "ok", "server": "LMDeploy Pipeline API", "port": 9000}
```

---

## Python 사용법

### requests + SSE 스트리밍

```python
import requests

# 방법 1: 파일 업로드
with open("config.yaml", "rb") as f:
    response = requests.post(
        "http://172.168.43.214:9000/pipeline/run/file",
        files={"file": ("config.yaml", f, "application/x-yaml")},
        stream=True,
    )

for line in response.iter_lines(decode_unicode=True):
    if line.startswith("data: "):
        print(line[6:])  # "data: " 제거 후 출력
```

```python
# 방법 2: YAML 문자열 전송
import requests

yaml_content = open("config.yaml").read()
response = requests.post(
    "http://172.168.43.214:9000/pipeline/run/yaml",
    json={"yaml_content": yaml_content},
    stream=True,
)

for line in response.iter_lines(decode_unicode=True):
    if line.startswith("data: "):
        print(line[6:])
```

### httpx (비동기)

```python
import httpx
import asyncio

async def run_pipeline():
    async with httpx.AsyncClient(timeout=600) as client:
        with open("config.yaml", "rb") as f:
            async with client.stream(
                "POST",
                "http://172.168.43.214:9000/pipeline/run/file",
                files={"file": ("config.yaml", f, "application/x-yaml")},
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        print(line[6:])

asyncio.run(run_pipeline())
```

### 상태 확인 (Python)

```python
import requests

status = requests.get("http://172.168.43.214:9000/pipeline/status").json()
print(f"상태: {status['status']}")

# 진행도 확인
progress = status.get("progress")
if progress:
    print(f"벤치마크: {progress['completed']}/{progress['total']} 완료")
    if progress["current"]:
        print(f"현재: {progress['current']}")
    for b in progress["benchmarks"]:
        line = f"  {b['name']}: {b['status']}"
        if "video" in b:
            line += f" (영상 {b['video']})"
        if "frame" in b:
            line += f" (프레임 {b['frame']})"
        print(line)

if status.get("error"):
    print(f"에러: {status['error']}")
    print(f"힌트: {status['hint']}")
```

---

## SSE 응답 이벤트

요청 후 서버가 아래 이벤트를 순서대로 전송합니다. 각 이벤트는 `data: {JSON}` 형태입니다.

| step | 의미 | 설명 |
|------|------|------|
| `received` | YAML 수신 | 파일/문자열 수신 완료 |
| `yaml_validated` | YAML 검증 | 파싱 + 필수 키 확인 통과 |
| `config_loaded` | 설정 로드 | PipelineConfig 생성 완료 |
| `model_check` | 모델 확인 | 로컬 모델 존재 확인 시작 |
| `model_ready` | 모델 준비 | 모델 사용 가능 (필요 시 HF 다운로드 포함) |
| `docker_starting` | Docker 기동 | 컨테이너 생성 시작 |
| `docker_waiting` | 서버 대기 | LMDeploy API 서버 준비 대기 중 |
| `docker_ready` | 서버 준비 완료 | API 서버 응답 확인, 소요 시간 포함 |
| `eval_started` | 평가 시작 | 벤치마크 평가 시작됨 |
| `done` | 완료 | SSE 스트리밍 종료. 평가는 백그라운드 subprocess에서 계속 진행 |
| `error` | 에러 | 에러 발생. `message`에 에러 내용, `hint`에 대처법 포함 |

### SSE 출력 예시

```
data: {"step": "received", "message": "YAML 수신 완료"}
data: {"step": "yaml_validated", "message": "YAML 검증 완료"}
data: {"step": "config_loaded", "message": "파이프라인: InternVL3-2B Fire Benchmark (LMDeploy)"}
data: {"step": "model_check", "message": "모델 확인 중..."}
data: {"step": "model_ready", "message": "모델 준비 완료: ckpts/InternVL3-2B"}
data: {"step": "docker_starting", "message": "Docker 컨테이너 기동 중..."}
data: {"step": "docker_waiting", "message": "LMDeploy 서버 준비 대기 중..."}
data: {"step": "docker_ready", "message": "컨테이너 준비 완료 (20.1s)"}
data: {"step": "eval_started", "message": "벤치마크 평가 시작 (6개). 리더보드에서 결과를 확인하세요."}
data: {"step": "done", "message": "평가가 시작되었습니다. 추후 벤치마크 결과를 리더보드에서 확인해보세요."}
```

---

## YAML 파일 보관

API로 제출된 YAML 파일은 서버의 `results/api/uploaded_yamls/` 디렉토리에 영구 보관됩니다.

- 파일명 형식: `{YYYYMMDD_HHMMSS}_{pipeline_name}.yaml`
- 디버깅이나 재실행 시 해당 디렉토리에서 원본 YAML을 확인할 수 있습니다.

---

## 에러 응답

### YAML 형식 에러 (HTTP 400)

```json
{
  "detail": {
    "status": "error",
    "message": "YAML 파싱 에러: ...",
    "hint": "YAML 형식을 확인해주세요. 작성 가이드: docs/eval/lmdeploy_yaml_guide.md"
  }
}
```

**대처**: YAML 문법을 확인하고 다시 제출

### 경로 검증 에러 (HTTP 400)

```json
{
  "detail": {
    "status": "validation_error",
    "message": "bench_base_path가 존재하지 않습니다: /workspace/PoC_banchmark/...",
    "hint": "bench_base_path 및 벤치마크 폴더 경로를 확인해주세요. 서버 기준 경로: /mnt/PoC_benchmark/huggingface_benchmarks_dataset/Leaderboard_bench"
  }
}
```

**대처**: `evaluate.bench_base_path`가 서버의 실제 경로와 일치하는지 확인. 벤치마크 폴더가 존재하는지 확인.

### GPU 사용 중 (HTTP 503)

```json
{
  "status": "gpu_busy",
  "message": "GPU가 현재 사용 중입니다 (VRAM 사용량: 52.3GB / 임계값: 50GB).\nGPU 0: 52300MB / 97887MB\n담당자에게 GPU 리소스에 대해 문의해주세요."
}
```

**대처**: GPU를 사용 중인 담당자에게 리소스 상황을 문의한 후 재요청

### 이미 실행 중 (HTTP 409)

```json
{
  "status": "busy",
  "message": "이미 평가가 진행 중입니다. 완료 후 다시 요청해주세요.",
  "current_pipeline": "InternVL3-2B Fire Benchmark (LMDeploy)",
  "started_at": "2026-03-30T14:30:00"
}
```

**대처**: `/pipeline/status`로 현재 상태 확인 후 완료되면 재요청

### SSE 도중 에러

```
data: {"step": "error", "message": "Docker 기동 실패: ...", "hint": "docker.port를 변경하거나 기존 컨테이너를 정리해주세요."}
```

주요 에러 유형과 대처:

| 에러 | 원인 | 대처 |
|------|------|------|
| 경로 검증 에러 (400) | `bench_base_path` 오타/잘못된 경로 | `evaluate.bench_base_path` 경로 확인 |
| 벤치마크 폴더 미존재 (400) | 벤치마크 폴더가 서버에 없음 | `evaluate.benchmarks` 목록의 폴더명 확인 |
| GPU 사용 중 (503) | VRAM 50GB 이상 사용 중 | 담당자에게 GPU 리소스 문의 |
| Docker OOM | GPU 메모리 부족 | `lmdeploy_args.tp` 증가 또는 `cache-max-entry-count` 축소 |
| 포트 충돌 | 이미 사용 중인 포트 | `docker.port` 변경 |
| 모델 미존재 | 경로 오류 | `docker.model_path` 확인 또는 `hf_repo_id` 설정 |
| 서버 준비 시간 초과 | 대형 모델 로딩 지연 | `startup.timeout_seconds` 증가 |

### 전체 벤치마크 실패 시 (`/pipeline/status`)

모든 벤치마크가 스킵 또는 실패하면 status가 `"failed"`로 설정됩니다.

```json
{
  "status": "failed",
  "pipeline_name": "PIA_AI2team_VQA_falldown Benchmark (LMDeploy)",
  "started_at": "2026-03-31T05:10:00",
  "finished_at": "2026-03-31T05:12:30",
  "elapsed": null,
  "error": "모든 벤치마크(13개)가 실패했습니다: ABB_Falldown, Coupang_Falldown, ...",
  "hint": "첫 번째 에러: [ABB_Falldown] Benchmark path not found: /workspace/PoC_banchmark/...",
  "result": null,
  "progress": {
    "total": 13,
    "completed": 13,
    "current": null,
    "benchmarks": [
      {"name": "ABB_Falldown", "status": "failed"},
      {"name": "Coupang_Falldown", "status": "failed"}
    ]
  }
}
```

---

## 서버 실행 (관리자용)

```bash
conda activate llm
cd /home/gpuadmin/Repo/seoik/train_eval_toolkit
uvicorn src.api.main:app --host 0.0.0.0 --port 9000
```

백그라운드 실행:

```bash
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 9000 > logs/api_server.log 2>&1 &
```
