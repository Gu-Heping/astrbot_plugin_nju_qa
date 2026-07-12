"""In-memory sliding-window rate limiter per chat context."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitState:
    allowed: bool
    current_count: int
    max_count: int
    window_seconds: int
    reset_after_seconds: int
    is_group: bool


class RateLimiter:
    """Track how many answers have been sent to a chat context in a window.

    ``max_count == 0`` or ``window_seconds == 0`` disables limiting for that
    chat type.  Timestamps are stored in memory and are lost on restart.
    """

    def __init__(
        self,
        group_max: int = 0,
        group_window_seconds: int = 3600,
        private_max: int = 0,
        private_window_seconds: int = 3600,
    ) -> None:
        self.group_max = max(0, int(group_max))
        self.group_window_seconds = max(0, int(group_window_seconds))
        self.private_max = max(0, int(private_max))
        self.private_window_seconds = max(0, int(private_window_seconds))
        self._buckets: dict[str, deque[float]] = {}

    def is_allowed(self, chat_key: str, is_group: bool) -> tuple[bool, RateLimitState]:
        max_count = self.group_max if is_group else self.private_max
        window = self.group_window_seconds if is_group else self.private_window_seconds
        now = time.monotonic()

        if max_count <= 0 or window <= 0:
            return True, RateLimitState(True, 0, max_count, window, 0, is_group)

        timestamps = self._buckets.setdefault(chat_key, deque())
        cutoff = now - window
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) < max_count:
            timestamps.append(now)
            return True, RateLimitState(
                True, len(timestamps), max_count, window, 0, is_group
            )

        reset_after = max(1, math.ceil(timestamps[0] + window - now))
        return False, RateLimitState(
            False, len(timestamps), max_count, window, reset_after, is_group
        )

    def reset(self) -> None:
        self._buckets.clear()
