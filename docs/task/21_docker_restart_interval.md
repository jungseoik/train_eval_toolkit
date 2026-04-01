# 21. 벤치마크 간 Docker 재시작 기능 추가

## 배경

벤치마크별 subprocess 분리(#20)로 API 서버 메모리 누적은 해결했으나,
Docker 내부 lmdeploy 서버의 메모리 증가(Python heap fragmentation)는 여전히 발생.
60,000건 요청 처리 시 Docker RAM이 30GB → 76GB까지 증가하는 것이 모니터링에서 확인됨.

## 변경 사항

### `docker_restart_interval` 설정 추가

```yaml
pipeline:
  docker_restart_interval: 1   # 기본값: 매 벤치마크마다 Docker 재시작
                                # 0: 비활성 (재시작 안 함)
                                # N: N개 벤치마크마다 재시작
```

### 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/lmdeploy_pipeline/config.py` | PipelineConfig에 `docker_restart_interval` 필드 추가 |
| `src/lmdeploy_pipeline/evaluator.py` | `_restart_docker()` 함수 + 벤치마크 루프에 재시작 로직 |
| `src/lmdeploy_pipeline/runner.py` | `run_evaluation` 호출 시 docker_cfg, docker_restart_interval 전달 |
| `src/api/pipeline_worker.py` | 동일 |
| `configs/lmdeploy_pipeline/*.yaml` (전체) | `docker_restart_interval: 1` 추가 |
| `configs/lmdeploy_pipeline/template/template.yaml` | 동일 |
| `docs/eval/lmdeploy_pipeline.md` | 동작 흐름도 업데이트 |
| `docs/eval/lmdeploy_yaml_guide.md` | 설정 설명 추가 |
| `docs/eval/memory_analysis.md` | Docker 측 해결 방법 업데이트 |

## 테스트 결과

Soil_Falldown(77개 영상) × 2회, concurrency=100, docker_restart_interval=1

```
[1/2] Soil_Falldown 평가 완료
  ↓ Docker 재시작 (15초)
[2/2] Soil_Falldown 평가 완료

부모 RSS: 42MB → 42MB (+0MB)
결과: 2/2 성공, CSV 77개 생성
Docker 재시작 소요: ~15초
잔여물: 없음 (Docker, CSV, YAML 모두 정리)
```

## 성능 영향

- Docker 재시작 소요: ~15초 (모델 로딩 포함)
- 13개 벤치마크 기준: 12회 × 15초 = 약 3분 추가
- 전체 평가 5시간 대비 약 1% 추가
