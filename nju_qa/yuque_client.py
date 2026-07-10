from __future__ import annotations
import asyncio
from typing import Any
from urllib.parse import quote
import httpx


class YuqueClient:
    def __init__(
        self,
        token: str,
        base_url: str,
        *,
        retries: int = 4,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url, self.retries, self._client = (
            base_url.rstrip("/"),
            retries,
            client,
        )
        self._headers = {
            "X-Auth-Token": token,
            "User-Agent": "astrbot-plugin-nju-qa/0.1",
        }
        self._lock = asyncio.Lock()

    async def _http(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    headers=self._headers, timeout=httpx.Timeout(30, connect=10)
                )
            return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        client = await self._http()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await client.get(self.base_url + path, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    delay = float(response.headers.get("Retry-After", 2**attempt))
                    await asyncio.sleep(min(delay, 16))
                    continue
                response.raise_for_status()
                payload = response.json()
                return payload.get("data", payload)
            except httpx.RequestError as exc:
                last = exc
                await asyncio.sleep(2**attempt)
        raise last or RuntimeError("Yuque request exhausted retries")

    async def get_repo(self, namespace: str) -> dict[str, Any]:
        return await self._get(f"/repos/{quote(namespace, safe='/-')}")

    async def get_toc(self, namespace: str) -> list[dict[str, Any]]:
        return await self._get(f"/repos/{quote(namespace, safe='/-')}/toc")

    async def get_document(self, namespace: str, slug: str) -> dict[str, Any]:
        return await self._get(
            f"/repos/{quote(namespace, safe='/-')}/docs/{quote(slug, safe='')}",
            {"include_content": "true"},
        )
