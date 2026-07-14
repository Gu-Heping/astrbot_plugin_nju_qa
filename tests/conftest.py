"""Provide a minimal astrbot shim so tests can import the plugin code."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest


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
    _Filter.PermissionType = PermissionType
    _Filter.EventMessageType = EventMessageType
    api.event = event_mod

    class AstrMessageEvent:
        def __init__(self) -> None:
            self.message_str = ""
            self.message_obj = types.SimpleNamespace(message=[])

    event_mod.AstrMessageEvent = AstrMessageEvent

    # core.agent.run_context stub used by nju_qa.tools.
    core = _make_module("astrbot.core")
    astrbot_pkg.core = core
    agent = _make_module("astrbot.core.agent")
    core.agent = agent
    run_context = _make_module("astrbot.core.agent.run_context")
    agent.run_context = run_context
    tool_mod = _make_module("astrbot.core.agent.tool")
    agent.tool = tool_mod

    class ContextWrapper:
        def __init__(self, *args, **kwargs) -> None:
            pass

    run_context.ContextWrapper = ContextWrapper

    class ToolSet:
        def __init__(self, tools=None):
            self.tools = list(tools) if tools is not None else []

    tool_mod.ToolSet = ToolSet

    # core.astr_agent_context stub used by nju_qa.tools.
    astr_agent_context = _make_module("astrbot.core.astr_agent_context")
    core.astr_agent_context = astr_agent_context

    class AstrAgentContext:
        def __init__(self, *args, **kwargs) -> None:
            pass

    astr_agent_context.AstrAgentContext = AstrAgentContext


_install_astrbot_shim()

# ---------------------------------------------------------------------------
# Load the plugin entry point once before test modules import submodules.
#
# main.py uses package-relative imports.  Loading it here populates nju_qa's
# attributes and lets us alias the submodule objects under their plain
# ``nju_qa.*`` names, so monkeypatch-based tests operate on the same module
# objects the plugin code uses.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import nju_qa  # noqa: E402

_pkg = types.ModuleType("astrbot_plugin_nju_qa")
_pkg.__path__ = [str(ROOT)]
sys.modules["astrbot_plugin_nju_qa"] = _pkg
sys.modules["astrbot_plugin_nju_qa.nju_qa"] = nju_qa

_main_spec = importlib.util.spec_from_file_location(
    "astrbot_plugin_nju_qa.main", ROOT / "main.py"
)
_main_mod = importlib.util.module_from_spec(_main_spec)
sys.modules["astrbot_plugin_nju_qa.main"] = _main_mod
_main_spec.loader.exec_module(_main_mod)

for _attr_name, _attr in list(nju_qa.__dict__.items()):
    if isinstance(_attr, types.ModuleType) and not _attr_name.startswith("_"):
        sys.modules.setdefault(f"nju_qa.{_attr_name}", _attr)


# ---------------------------------------------------------------------------
# Shared fixtures for integration tests that need to load main.py.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plugin_class():
    """Return the plugin class loaded at conftest import time."""
    return sys.modules["astrbot_plugin_nju_qa.main"].NjuQaPlugin
