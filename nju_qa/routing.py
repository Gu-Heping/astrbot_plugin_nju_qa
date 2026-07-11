"""Message routing rules independent from the AstrBot runtime."""

from __future__ import annotations

from dataclasses import dataclass


COMMAND_MARKER = "_nju_qa_command_handled"


def mark_command_handled(event: object) -> None:
    """Stop a command event before yielding its result.

    AstrBot delivers command messages to ALL-message listeners too.  The marker
    protects this plugin if handler ordering changes; ``stop_event`` protects
    subsequent framework handlers as documented by AstrBot.
    """

    setattr(event, COMMAND_MARKER, True)
    stop = getattr(event, "stop_event", None)
    if callable(stop):
        stop()


def is_command_message(text: str) -> bool:
    """Recognize a command syntactically, without maintaining command lists."""

    return text.lstrip().startswith(("/", "!"))


@dataclass(frozen=True)
class RoutedMessage:
    should_handle: bool
    query: str = ""


class MessageRouter:
    """Routes only explicitly eligible non-command messages to the agent."""

    def __init__(
        self, wake_words: tuple[str, ...], private_enabled: bool, group_at_enabled: bool
    ):
        self.wake_words = wake_words
        self.private_enabled = private_enabled
        self.group_at_enabled = group_at_enabled

    def route(self, event: object, text: str, is_at_me: bool) -> RoutedMessage:
        text = text.strip()
        if (
            not text
            or is_command_message(text)
            or getattr(event, COMMAND_MARKER, False)
        ):
            return RoutedMessage(False)

        is_group = getattr(event, "get_group_id")() not in (None, "")
        if not is_group:
            return RoutedMessage(self.private_enabled, text)
        if not self.group_at_enabled:
            return RoutedMessage(False)
        if is_at_me:
            return RoutedMessage(True, text)

        lowered = text.casefold()
        for wake_word in self.wake_words:
            if lowered.startswith(wake_word.casefold()):
                return RoutedMessage(True, text[len(wake_word) :].lstrip(" ，,：:"))
        return RoutedMessage(False)
