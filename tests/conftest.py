"""Provide a minimal astrbot shim so tests can import the plugin code."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Callable


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_astrbot_shim() -> None:
    if "astrbot" in sys.modules:
        return

    astrbot_pkg = _make_module("astrbot")
    api = _make_module("astrbot.api")
    astrbot_pkg.api = api

    # Logger used across the plugin.
    api.logger = logging.getLogger("astrbot")

    # Tool base class used by nju_qa.tools.
    class FunctionTool:
        def __init__(self, *args, **kwargs) -> None:
            pass

    api.FunctionTool = FunctionTool

    # message_components stubs used by main.py and retriever formatting.
    mc = _make_module("astrbot.api.message_components")

    class Plain:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class At:
        def __init__(self, qq: str = "") -> None:
            self.qq = qq

    class Image:
        @staticmethod
        def fromFileSystem(path: str):  # noqa: N802
            return Image(path)

        def __init__(self, path: str = "") -> None:
            self.path = path

    mc.Plain = Plain
    mc.At = At
    mc.Image = Image
    api.message_components = mc

    # star stubs used for plugin registration.
    star = _make_module("astrbot.api.star")

    class Context:
        pass

    class Star:
        def __init__(self, context: object) -> None:
            self.context = context

    def register(*args, **kwargs) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            return cls

        return decorator

    star.Context = Context
    star.Star = Star
    star.register = register
    api.star = star

    # event/filter stubs used by command handlers.
    event_mod = _make_module("astrbot.api.event")

    class PermissionType:
        ADMIN = "admin"

    class EventMessageType:
        ALL = "all"

    class _Filter:
        @staticmethod
        def command(name: str = "") -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

        @staticmethod
        def permission_type(kind: object) -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

        @staticmethod
        def event_message_type(kind: object) -> Callable[[Callable], Callable]:
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

    event_mod.filter = _Filter()
    event_mod.PermissionType = PermissionType
    event_mod.EventMessageType = EventMessageType
    api.event = event_mod

    class AstrMessageEvent:
        def __init__(self) -> None:
            self.message_str = ""
            self.message_obj = types.SimpleNamespace(message=[])

    event_mod.AstrMessageEvent = AstrMessageEvent


_install_astrbot_shim()
