# Task 13: LMDeploy YAML 설정 작성 가이드 문서 생성

## 개요

외부에서 YAML 파일을 전달받아 LMDeploy 파이프라인을 실행하는 시나리오에 대비하여, YAML 설정 파일을 처음부터 작성할 수 있는 상세 가이드 문서를 생성했다.

## 배경

기존 `docs/eval/lmdeploy_pipeline.md`에 YAML 레퍼런스 테이블은 있었지만, 외부인이 문서만 보고 YAML을 작성하기에는 다음이 부족했다:
- 복사해서 수정할 수 있는 전체 YAML 템플릿 부재
- 사용 가능한 벤치마크 이름 목록 부재
- prompt_templates 작성 규칙/예시 부재
- 고정값 필드(evaluate.model, bench_base_path, gradio_url) 명시 부족
- window_size 선택 기준 안내 부재

## 변경 내용

### 신규 파일

| 파일 | 설명 |
|------|------|
| `docs/eval/lmdeploy_yaml_guide.md` | YAML 설정 작성 가이드 (전체 템플릿, 섹션별 설명, 벤치마크 목록, 프롬프트 작성법) |

### 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `docs/eval/lmdeploy_pipeline.md` | YAML 레퍼런스 섹션에 YAML 가이드 크로스링크 추가 |
| `README.md` | LMDeploy 파이프라인 섹션에 YAML 가이드 링크 추가 |

## 문서 구성

| 섹션 | 내용 |
|------|------|
| 전체 YAML 템플릿 | `PIA_AI2team_VQA_falldown.yaml` 기반, `# <-- 변경`/`# <-- 고정값` 주석 표시 |
| 섹션별 상세 설명 | pipeline, retry, docker, evaluate, submit 각 필드 설명 + 고정값/변경가능 구분 |
| 벤치마크 전체 목록 | Fire 6개, Falldown 13개 + 네이밍 규칙 |
| prompt_templates 작성 가이드 | 키 매칭 규칙, default 필수, {category} 플레이스홀더, 상세/간단 프롬프트 예시 |
| Fire 평가 시 변경 사항 | Falldown 템플릿 대비 diff 형태 |
| 모델 교체 시 변경 필드 요약 | 빠른 참조표 |

## 주요 결정사항

- **파일 위치**: `docs/eval/lmdeploy_yaml_guide.md` — 기존 docs가 워크플로우 단계별 구조이므로 `docs/eval/`에 배치
- **메인 예시**: `PIA_AI2team_VQA_falldown.yaml` — 가장 최근 추가된 YAML로 hf_repo_id 포함
- **고정값 명시**: evaluate.model, bench_base_path, gradio_url을 고정값으로 명확히 표시

## 작성자

jungseoik
