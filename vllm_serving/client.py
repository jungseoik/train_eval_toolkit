"""
vLLM 비동기 추론 클라이언트.

이미지 + 프롬프트를 vLLM 서버(OpenAI 호환 API)로 전송하고 결과를 반환한다.
최대 동시 요청 수를 Semaphore로 제한하여 서버 과부하를 방지한다.

사용법:
    from client import VLLMClient

    async with VLLMClient("http://localhost:8000/v1", model="Qwen/Qwen3.5-2B") as client:
        result = await client.infer("image.jpg", "이 이미지에 불이 있나요?")
        print(result)
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx


class VLLMClient:
    """vLLM OpenAI 호환 API 비동기 클라이언트."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3.5-2B",
        max_concurrency: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        """
        Args:
            api_base: vLLM 서버 주소 (예: http://localhost:8000/v1)
            model: 모델명 (docker-compose에서 설정한 것과 동일해야 함)
            max_concurrency: 최대 동시 요청 수
            max_tokens: 응답 최대 토큰 수
            temperature: 샘플링 온도 (0.0 = deterministic)
            timeout: 요청 타임아웃 (초)
        """
        self.api_base = api_base.rstrip("/")
        self.api_url = f"{self.api_base}/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> VLLMClient:
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._semaphore._value,
                max_keepalive_connections=self._semaphore._value,
            ),
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── 서버 상태 확인 ──

    async def health_check(self) -> bool:
        """서버가 요청을 받을 준비가 되었는지 확인."""
        try:
            resp = await self._client.get(f"{self.api_base}/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def wait_until_ready(self, timeout: float = 300, interval: float = 5) -> bool:
        """서버가 준비될 때까지 폴링. 준비되면 True, 타임아웃이면 False."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            if await self.health_check():
                return True
            await asyncio.sleep(interval)
        return False

    # ── 단일 추론 ──

    async def infer(
        self,
        image: str | Path | bytes,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        이미지 1장 + 프롬프트를 전송하고 모델 응답 텍스트를 반환.

        Args:
            image: 이미지 파일 경로 (str/Path) 또는 raw bytes
            prompt: 텍스트 프롬프트
            max_tokens: 응답 최대 토큰 (None이면 기본값 사용)
            temperature: 샘플링 온도 (None이면 기본값 사용)

        Returns:
            모델 응답 텍스트 (raw string)
        """
        image_b64 = self._encode_image(image)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        async with self._semaphore:
            resp = await self._client.post(self.api_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # ── 배치 추론 ──

    async def infer_batch(
        self,
        items: list[dict[str, Any]],
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        여러 이미지를 비동기 배치로 추론.
        Semaphore로 동시 요청 수가 제한된다 (기본 3).

        Args:
            items: [{"image": 경로 또는 bytes, "prompt": str(선택)}] 리스트
                   prompt 키가 없으면 인자로 전달된 prompt를 사용
            prompt: 모든 항목에 공통 적용할 프롬프트 (개별 prompt가 우선)

        Returns:
            [{"index": int, "response": str, "error": str|None}] 리스트
        """
        async def _process(idx: int, item: dict) -> dict:
            item_prompt = item.get("prompt", prompt)
            if not item_prompt:
                return {"index": idx, "response": None, "error": "프롬프트가 지정되지 않음"}
            try:
                result = await self.infer(item["image"], item_prompt)
                return {"index": idx, "response": result, "error": None}
            except Exception as e:
                return {"index": idx, "response": None, "error": str(e)}

        tasks = [_process(i, item) for i, item in enumerate(items)]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x["index"])

    # ── 응답 파싱 유틸리티 ──

    @staticmethod
    def parse_json_response(raw_text: str, valid_values: list[str] | None = None) -> dict | None:
        """
        모델 응답에서 JSON 객체를 추출.

        모델이 ```json ... ``` 블록이나 일반 텍스트로 JSON을 반환할 때 파싱한다.

        Args:
            raw_text: 모델 raw 응답
            valid_values: category 필드의 허용값 목록 (검증용, None이면 검증 안 함)

        Returns:
            파싱된 dict 또는 실패 시 None
        """
        clean = raw_text
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        clean = clean.strip()

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(clean[start:end + 1])
                if valid_values and data.get("category") not in valid_values:
                    return None
                return data
            except json.JSONDecodeError:
                pass

        # regex fallback
        if valid_values:
            pattern = r'["\']category["\']\s*:\s*["\'](' + "|".join(re.escape(v) for v in valid_values) + r')["\']'
            m = re.search(pattern, clean)
            if m:
                return {"category": m.group(1)}

        return None

    @staticmethod
    def parse_yes_no(raw_text: str) -> bool:
        """모델의 yes/no 응답을 bool로 변환."""
        return raw_text.strip().lower() in ("yes", "1")

    # ── 내부 메서드 ──

    @staticmethod
    def _encode_image(image: str | Path | bytes) -> str:
        """이미지를 base64 문자열로 인코딩."""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("ascii")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")
