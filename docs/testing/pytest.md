# Pytest 가이드

이 문서는 `tests/` 루트 기반 테스트 구조와 실행 방법을 설명합니다.

## 테스트 구조

```text
tests/
├── conftest.py
├── _autolabeling/
│   └── gemini/
│       └── test_live_red_image_smoke.py
└── ...
```

- `tests/`를 루트로 두고, 소스 모듈 구조를 미러링해서 확장합니다.
- 단위 테스트가 늘어나면 같은 패턴으로 파일만 추가하면 됩니다.
- 라이브 API 테스트는 `integration` marker를 사용합니다.

## pytest 설정 파일 역할

`pytest.ini`에서 아래를 관리합니다.

- `testpaths = tests`: 테스트 탐색 루트 고정
- `python_files = test_*.py`: 테스트 파일명 규칙
- `addopts = -ra`: skip/xfail 이유 요약 출력
- `markers`: `integration` 같은 사용자 marker 등록

## 실행 방법

### 1) 기본 테스트

```bash
pytest -q
```

### 2) 라이브 Gemini 스모크 테스트

```bash
export GEMINI_API_KEY="..."
export RUN_LIVE_GEMINI_TESTS=1
pytest -s -m integration tests/_autolabeling/gemini/test_live_red_image_smoke.py
```

## red image 스모크 테스트 동작

`test_live_red_image_smoke.py`는 아래를 수행합니다.

1. 임시 경로에 빨간색 이미지(256x256) 생성
2. Gemini에 이미지 분석 요청
3. 원문 응답 출력
4. JSON 파싱 결과 출력
5. 최소 유효성(빈 응답 아님, 파싱 가능) 검증

이 테스트는 "작동 여부 + 출력 확인" 목적이며, 최종 의미 해석(`red`가 맞는지)은 실행자가 확인합니다.

## skip 동작

아래 조건에서 skip 됩니다.

- `RUN_LIVE_GEMINI_TESTS` 없음/값 불일치
- `GEMINI_API_KEY` 없음
- 런타임 의존성(`google-genai`) 없음

`-ra` 옵션으로 skip 이유를 요약에서 바로 확인할 수 있습니다.

## 새 테스트 추가 규칙(권장)

1. 파일명은 `test_*.py`
2. API 실호출 테스트는 `@pytest.mark.integration` 부여
3. 파일/디렉토리 I/O는 `tmp_path` fixture 사용
4. 공통 설정/fixture는 `tests/conftest.py`에 모아 재사용
