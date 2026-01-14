#!/bin/bash
# --------------------------------------------------------------
# empty_json_checker: JSON 파일 중 clips가 비어있는지 검사하는 스크립트
#
# 사용방법:
#   python src/data_checker/stats/empty_json_checker.py --json_dir <JSON_폴더경로>
#
# 예시:
#   python src/data_checker/stats/empty_json_checker.py --json_dir data/raw/elevator_normal
#
# 기능:
#   - 라벨링이 잘되어있는지 확인하는 함수임.
#   - 지정된 폴더 내 모든 .json 파일을 검사
#   - clips 배열이 비어있는 JSON 파일만 찾아서 개수/비율 출력
#   - 문제 있는 파일들의 전체 경로까지 함께 출력
#
# 기본값:
#   --json_dir 옵션을 생략하면 내부 기본 경로를 자동 사용함
#
# --------------------------------------------------------------
python src/data_checker/stats/empty_json_checker.py --json_dir data/raw/ai_hub_indoor_store_violence