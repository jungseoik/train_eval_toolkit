"""
vLLM 클라이언트 사용 예시.

사전 조건:
    1. docker compose up -d 로 vLLM 서버 실행
    2. pip install httpx
    3. 이미지 파일 준비

실행:
    python example.py
"""

import asyncio
from pathlib import Path

from client import VLLMClient


# =============================================================
# 예시 1: 단일 이미지 추론
# =============================================================
async def single_inference():
    """이미지 1장을 보내고 결과를 받는 기본 예시."""
    async with VLLMClient(
        api_base="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_concurrency=3,
    ) as client:
        # 서버 준비 대기 (최초 실행 시 모델 로딩에 시간 소요)
        print("서버 준비 대기 중...")
        ready = await client.wait_until_ready(timeout=300)
        if not ready:
            print("서버가 준비되지 않았습니다.")
            return

        # 추론 요청
        response = await client.infer(
            image="test_image.jpg",
            prompt='이 이미지를 분석하세요. 불이 보이면 "fire", 아니면 "normal"로 답하세요.\n\nJSON 형식: {"category": "fire" 또는 "normal", "description": "이유"}',
        )
        print(f"[응답] {response}")

        # JSON 파싱
        parsed = VLLMClient.parse_json_response(response, valid_values=["fire", "normal"])
        if parsed:
            print(f"[파싱] category={parsed['category']}, description={parsed.get('description', '')}")
        else:
            print("[파싱] JSON 파싱 실패 — raw 텍스트를 직접 처리하세요")


# =============================================================
# 예시 2: 배치 추론 (여러 이미지를 동시에)
# =============================================================
async def batch_inference():
    """여러 이미지를 비동기 배치로 추론하는 예시."""
    image_dir = Path("./images")
    if not image_dir.exists():
        print(f"이미지 디렉토리가 없습니다: {image_dir}")
        print("images/ 폴더에 이미지를 넣고 다시 실행하세요.")
        return

    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_files:
        print("이미지 파일이 없습니다.")
        return

    # 각 이미지에 동일한 프롬프트를 적용하는 배치
    items = [{"image": str(f)} for f in image_files]
    prompt = "이 이미지에 연기(smoke)가 보이나요? yes 또는 no로만 답하세요."

    async with VLLMClient(
        api_base="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_concurrency=3,  # 최대 3개 동시 요청
    ) as client:
        print(f"총 {len(items)}개 이미지 배치 추론 시작...")
        results = await client.infer_batch(items, prompt=prompt)

        for r in results:
            filename = image_files[r["index"]].name
            if r["error"]:
                print(f"  [{filename}] 오류: {r['error']}")
            else:
                detected = VLLMClient.parse_yes_no(r["response"])
                print(f"  [{filename}] 응답={r['response'].strip()}, 감지={'YES' if detected else 'NO'}")


# =============================================================
# 예시 3: 이미지 bytes를 직접 전달
# =============================================================
async def bytes_inference():
    """파일 경로 대신 bytes를 직접 전달하는 예시."""
    image_path = Path("test_image.jpg")
    if not image_path.exists():
        print(f"이미지 파일이 없습니다: {image_path}")
        return

    image_bytes = image_path.read_bytes()

    async with VLLMClient(
        api_base="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    ) as client:
        response = await client.infer(
            image=image_bytes,
            prompt="이 이미지에 무엇이 보이나요? 간단히 설명하세요.",
            max_tokens=128,
        )
        print(f"[응답] {response}")


# =============================================================
# 메인
# =============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("예시 1: 단일 이미지 추론")
    print("=" * 50)
    asyncio.run(single_inference())

    print()
    print("=" * 50)
    print("예시 2: 배치 추론")
    print("=" * 50)
    asyncio.run(batch_inference())

    print()
    print("=" * 50)
    print("예시 3: bytes 전달")
    print("=" * 50)
    asyncio.run(bytes_inference())
