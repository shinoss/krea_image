"""Content-filter hook (Krea 2 Community License §4.2: deployments must filter unlawful or policy-violating
content). Every generation passes through check_prompt() before any work and check_image() before the
image is saved or shown. A blocked request fails with the returned reason and nothing is written.

The default is deliberately small: a prompt blocklist (one term per line in content_filter.txt at the
project root, matched as whole words, case-insensitive) plus no image check. It is a hook, not a complete
safety system: plug in a real classifier by pointing KREA_FILTER at a Python module that defines
check_prompt(prompt) -> (ok, reason) and/or check_image(pil_image, prompt) -> (ok, reason).
"""
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIST = os.path.join(ROOT, "content_filter.txt")
_patterns = None
_mtime = None
_plugin = None


def _load_plugin():
    global _plugin
    path = os.environ.get("KREA_FILTER")
    if _plugin is None and path:
        spec = importlib.util.spec_from_file_location("krea_filter_plugin", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _plugin = mod
    return _plugin


def _terms():
    global _patterns, _mtime
    try:
        mt = os.path.getmtime(_LIST)
    except OSError:
        return []
    if _patterns is None or mt != _mtime:
        with open(_LIST, encoding="utf-8") as f:
            terms = [t.strip() for t in f if t.strip() and not t.startswith("#")]
        _patterns = [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE) for t in terms]
        _mtime = mt
    return _patterns


def check_prompt(prompt):
    """(ok, reason) for a prompt (checked before the text encoder runs)."""
    for p in _terms():
        if p.search(prompt):
            return False, "The prompt was blocked by the content filter."
    plugin = _load_plugin()
    if plugin is not None and hasattr(plugin, "check_prompt"):
        return plugin.check_prompt(prompt)
    return True, ""


def check_image(img, prompt=""):
    """(ok, reason) for a generated image (checked before it is saved or shown)."""
    plugin = _load_plugin()
    if plugin is not None and hasattr(plugin, "check_image"):
        return plugin.check_image(img, prompt)
    return True, ""


def describe():
    plugin = os.environ.get("KREA_FILTER")
    n = len(_terms())
    return f"prompt blocklist: {n} terms ({_LIST})" + (f", plugin: {plugin}" if plugin else ", no image classifier")
