"""Content-filter hook (the Krea 2 Turbo model card, Safety Measures: deployers must implement content filtering
or an equivalent review process against unlawful or policy-violating content). Every generation passes through
check_prompt() before any work and check_image() before the image is saved or shown. A blocked request fails
with the returned reason and nothing is written.

The default is deliberately small: a prompt blocklist in content_filter.txt at the project root plus no image
check. The list stores hashes, not the phrases themselves: each entry is "sha256 <words> <digest>", the salted
SHA-256 of a normalized phrase (NFKC, lower case, runs of non-alphanumerics -> one space) of <words> words, and
a prompt is blocked when any run of <words> consecutive normalized words hashes to a listed digest (whole-word,
case-insensitive matching). Plain-text lines are accepted too, for local additions.

  python ui/content_filter.py add       # asks for a phrase (hidden input) and appends its hash
  python ui/content_filter.py check     # asks for a prompt and reports whether it would be blocked

It is a hook, not a complete safety system: plug in a real classifier by pointing KREA_FILTER at a Python
module that defines check_prompt(prompt) -> (ok, reason) and/or check_image(pil_image, prompt) -> (ok, reason).
"""
import hashlib
import importlib.util
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIST = os.path.join(ROOT, "content_filter.txt")
_SALT = "krea-content-filter:"
_rules = None  # (hashed: {word count: set(digest)}, plain regex patterns)
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


def _words(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", " ", text).split()


def _digest(words):
    return hashlib.sha256((_SALT + " ".join(words)).encode("utf-8")).hexdigest()


def entry(phrase):
    """The content_filter.txt line for a phrase."""
    w = _words(phrase)
    if not w:
        raise ValueError("empty phrase")
    return f"sha256 {len(w)} {_digest(w)}"


def _load():
    global _rules, _mtime
    try:
        mt = os.path.getmtime(_LIST)
    except OSError:
        return {}, []
    if _rules is None or mt != _mtime:
        hashed, plain = {}, []
        with open(_LIST, encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                parts = t.split()
                if len(parts) == 3 and parts[0] == "sha256" and parts[1].isdigit():
                    hashed.setdefault(int(parts[1]), set()).add(parts[2])
                else:
                    plain.append(re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE))
        _rules, _mtime = (hashed, plain), mt
    return _rules


def _blocked(prompt):
    hashed, plain = _load()
    if hashed:
        w = _words(prompt)
        for n, digests in hashed.items():
            if any(_digest(w[i:i + n]) in digests for i in range(len(w) - n + 1)):
                return True
    return any(p.search(prompt) for p in plain)


def check_prompt(prompt):
    """(ok, reason) for a prompt (checked before the text encoder runs)."""
    if _blocked(prompt):
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
    hashed, plain = _load()
    n = sum(len(v) for v in hashed.values()) + len(plain)
    return f"prompt blocklist: {n} terms ({_LIST})" + (f", plugin: {plugin}" if plugin else ", no image classifier")


if __name__ == "__main__":
    import getpass

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "add":  # hidden input: the phrase never reaches the shell history or the file
        line = entry(getpass.getpass("phrase to block (input hidden): "))
        with open(_LIST, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print("added", line)
    elif cmd == "check":
        ok, reason = check_prompt(getpass.getpass("prompt to check (input hidden): "))
        print("allowed" if ok else reason)
    else:
        sys.exit("usage: python ui/content_filter.py add | check")
