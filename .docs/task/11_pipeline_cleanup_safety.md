# Task 11: 파이프라인 Docker Cleanup 안전성 강화

## 개요

vllm_pipeline과 lmdeploy_pipeline의 `runner.py`에서 비정상 종료 시 Docker 컨테이너 cleanup이 실행되지 않는 버그를 발견하고 수정하였습니다.

---

## 배경 및 문제

### 발견된 버그

| 번호 | 위치 | 문제 |
|------|------|------|
| 1 | `runner.py:58-61` | `_run_docker_step()` 실패 시 `return`으로 조기 탈출 → cleanup 블록(108행) 미도달 |
| 2 | 평가/제출 단계 | `run_evaluation()`, `submit_results()`에서 시스템 레벨 예외 발생 시 `try/except` 부재 → cleanup 미실행 |
| 3 | 프로세스 강제 종료 | SIGTERM, SIGHUP 수신 시 signal handler 부재 → cleanup 코드 실행 기회 없음 |

### 실제 발생 시나리오

1. tmux eval 세션이 죽으면서 파이프라인 프로세스 강제 종료
2. Docker 컨테이너가 정리되지 않은 채 GPU를 점유
3. 기존 서버와 합산하여 GPU 전체 고갈
4. CUDA 드라이버 상태 손상

---

## 수정 내용

**수정 대상 파일**

- `src/vllm_pipeline/runner.py`
- `src/lmdeploy_pipeline/runner.py`

두 파일에 동일한 로직을 적용하였습니다.

### 1. try/finally로 전체 파이프라인 감싸기

Docker 기동 → 평가 → 결과 제출 전체를 `try` 블록으로 감싸고, cleanup을 `finally`에 배치합니다.

- 어떤 예외 또는 조기 반환이 발생하더라도 cleanup 블록 도달을 보장합니다.

### 2. Signal Handler 등록

SIGTERM, SIGHUP 수신 시 cleanup을 실행한 후 프로세스를 종료합니다.

- SIGHUP: tmux 세션 종료 시 발생
- SIGTERM: 외부 `kill` 명령 수신 시 발생
- `cleanup_docker` 설정 플래그를 signal handler 내에서도 동일하게 존중합니다.

### 3. atexit 등록

Python 정상 종료 경로의 마지막 안전망으로 `atexit`를 등록합니다.

- signal handler와 `finally`에서 cleanup이 누락된 경우를 대비합니다.

### 4. 멱등성 보장

`_cleanup_state["cleaned"]` 플래그를 사용하여 중복 cleanup을 방지합니다.

- `finally`, signal handler, `atexit` 중 어느 경로에서 진입하더라도 cleanup은 한 번만 실행됩니다.

### 5. 컨테이너 로그 덤프 (`_dump_container_logs`)

cleanup 실행 전에 `docker logs --tail 50` 명령으로 마지막 로그를 콘솔에 출력합니다.

- 컨테이너 삭제 이후에도 디버깅 단서가 콘솔에 남도록 합니다.

---

## YAML 호환성

- YAML 스키마, config 파싱, CLI 인터페이스에 변경 없음
- 기존 YAML 설정으로 동일하게 동작
- 비정상 종료 시에만 cleanup이 추가 실행되는 구조

---

## 한계

- **SIGKILL (`kill -9`), OOM Killer**: 커널 레벨 강제 종료로 signal handler가 동작하지 않음
- **로그 유실**: 콘솔 출력 기반 로깅이므로 tmux 세션이 죽으면 로그도 함께 유실됨 (파일 로깅은 별도 작업 필요)

---

## 작성자

jungseoik
