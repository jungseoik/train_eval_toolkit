# Task 18: /pipeline/status 진행도 조회 기능 추가

## 개요

평가 진행 중 `GET /pipeline/status`로 벤치마크/영상/프레임 단위 세부 진행 상황을 조회할 수 있도록 기능 추가.
기존에는 SSE 스트림으로만 실시간 진행 상황을 확인할 수 있었고, 연결이 끊기면 진행도를 알 수 없었음.

## 변경 내용

### 수정 파일

| 파일 | 변경 |
|------|------|
| `src/api/main.py` | `_state`에 `progress` 필드 추가 |
| `src/api/pipeline_worker.py` | 평가 시작 시 `progress` 초기화 (벤치마크 리스트 + pending 상태), `run_evaluation`에 `progress_state` 전달 |
| `src/lmdeploy_pipeline/evaluator.py` | `run_evaluation`에 `progress_state` 파라미터 추가, 벤치마크별 상태 업데이트 |
| `src/evaluation/lmdeploy_bench_eval.py` | `evaluate_benchmark`와 `_evaluate_video_async`에 progress 파라미터 추가, 영상/프레임 단위 진행도 업데이트 |

### 응답 형식

```json
{
  "status": "evaluating",
  "pipeline_name": "Qwen3.5-2B Fire Benchmark",
  "started_at": "2026-04-01T10:00:00",
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

## 설계 결정

- **프레임 업데이트 주기**: 매 프레임이 아닌 10프레임 간격 + 마지막 프레임에서 업데이트 (dict 업데이트 오버헤드 최소화)
- **하위 호환**: `progress` 필드 추가만으로, 기존 클라이언트는 무시하면 됨
- **CLI 영향 없음**: `progress_state`와 `bench_idx`는 optional 파라미터 (기본값 None), CLI 사용 시 전달하지 않으면 기존 동작 유지
- **thread safety**: Python GIL이 dict 단일 키 업데이트를 보호, 기존 `_state` 패턴과 동일

## 사이드이펙트

없음. 기존 SSE 스트림, CLI 평가, lock 메커니즘에 영향 없음.
