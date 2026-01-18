import argparse
import concurrent.futures
import hashlib
import http.client
import json
import logging
import os
import re
import sqlite3
import threading
import time
import tomllib
import urllib.parse
from typing import Any
from collections import Counter
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PLACEHOLDER_PREFIX = "__PH__"
PROMPT_LEAK_PATTERNS = (
    "Preserve all Markdown",
    "Do not copy or repeat these instructions",
    "Do not add any extra sentences",
)


def contains_prompt_leak(text: str) -> bool:
    if not text:
        return False
    for marker in PROMPT_LEAK_PATTERNS:
        if marker in text:
            return True
    return False
PROMPT_TEMPLATE: str | None = None
CUSTOM_SYSTEM_PROMPT = ""
DEFAULT_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_templates",
    "translate_gemma_12b.txt",
)
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_CONTEXT_TOKENS = 6586
DEFAULT_CONFIG_PATH = os.environ.get("TRANSLATEGEMMA_CONFIG", "pyproject.toml")
DEFAULT_TOKENIZER_PATH = os.environ.get("TRANSFORMERS_TOKENIZER_PATH", "")
DEFAULT_SHORT_ROW_TOKENS = 0
DEFAULT_TABLE_CONTEXT_ROWS = 0
DEFAULT_GLOSSARY_MAX_TERMS = 0
DEFAULT_CACHE_PATH = ".translategemma_cache.sqlite3"
DEFAULT_CACHE_ENABLED = False
DEFAULT_MIN_CELL_TOKENS = 0
DEFAULT_TABLE_BATCH_MODE = "table"
DEFAULT_ROW_RAW_MIN_TOKENS = 120
TOKENIZER_PATH: str | None = None
_TRANSFORMERS_TOKENIZER: Any | None = None
_TRANSFORMERS_LOCK = threading.Lock()
_TOKENIZER_FALLBACK_WARNING = False
PROMPT_FINGERPRINT = ""
SOURCE_LANG = ""
TARGET_LANG = ""
MAX_CONTEXT_TOKENS = DEFAULT_MAX_CONTEXT_TOKENS
_LOG_CONTEXT = threading.local()


@contextmanager
def log_context(tag: str):
    if not tag:
        yield
        return
    stack = getattr(_LOG_CONTEXT, "stack", [])
    stack.append(tag)
    _LOG_CONTEXT.stack = stack
    try:
        yield
    finally:
        stack.pop()


def current_log_context() -> str:
    stack = getattr(_LOG_CONTEXT, "stack", [])
    if not stack:
        return ""
    return "/".join(stack)


def input_token_limit(max_tokens: int, context: str) -> int:
    overhead = estimate_prompt_overhead_tokens(SOURCE_LANG, TARGET_LANG, context)
    available = MAX_CONTEXT_TOKENS - overhead - max_tokens - 64
    if available < 64:
        available = 64
    return max(64, min(max_group_tokens(max_tokens), available))




def get_transformers_tokenizer():
    global _TRANSFORMERS_TOKENIZER
    if _TRANSFORMERS_TOKENIZER is not None:
        return _TRANSFORMERS_TOKENIZER
    with _TRANSFORMERS_LOCK:
        if _TRANSFORMERS_TOKENIZER is not None:
            return _TRANSFORMERS_TOKENIZER
        tokenizer_path = TOKENIZER_PATH or DEFAULT_TOKENIZER_PATH
        if not tokenizer_path:
            raise RuntimeError(
                "Tokenizer path is required. Set TRANSFORMERS_TOKENIZER_PATH or pass --tokenizer-path."
            )
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            LOGGER.error("transformers not installed; run: uv add transformers sentencepiece")
            raise RuntimeError("transformers is required for token counting") from exc
        local_only = os.path.exists(tokenizer_path)
        _TRANSFORMERS_TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=local_only,
        )
        LOGGER.info("Using transformers tokenizer: %s", tokenizer_path)
    return _TRANSFORMERS_TOKENIZER


def load_config(path: str) -> dict | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        config = tomllib.load(f)
    LOGGER.info("Loaded config: %s", path)
    return config


def get_config_section(config: dict | None) -> dict:
    if not config:
        return {}
    tool = config.get("tool", {})
    return tool.get("translategemma", {})


def read_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class PersistentCache:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS translations ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT value FROM translations WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO translations (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()


def get_prompt_fingerprint() -> str:
    template = PROMPT_TEMPLATE or ""
    payload = f"{template}\n{CUSTOM_SYSTEM_PROMPT}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cache_key(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    context: str,
) -> str:
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "url": url,
        "context": context,
        "prompt": PROMPT_FINGERPRINT,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest


def extract_glossary_terms(content: str, max_terms: int) -> list[str]:
    term_pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.:-]{1,}\b")
    terms = term_pattern.findall(content)
    counts = Counter(terms)
    ranked = [term for term, _count in counts.most_common(max_terms)]
    return ranked


def build_glossary_context(terms: list[str]) -> str:
    if not terms:
        return ""
    joined = ", ".join(terms)
    return f"Glossary (keep identifiers unchanged): {joined}"


def build_context_text(title: str, glossary: str) -> str:
    parts = []
    if title:
        parts.append(f"Context title: {title}")
    if glossary:
        parts.append(glossary)
    return "\n".join(parts)


def build_table_context(
    base_context: str, rows: list[str], start_idx: int, context_rows: int
) -> str:
    parts = []
    if base_context:
        parts.append(base_context)
    if context_rows > 0:
        start = max(0, start_idx - context_rows)
        prefix_rows = rows[start:start_idx]
        if prefix_rows:
            parts.append("Table context rows:\n" + "\n".join(prefix_rows))
    return "\n".join(parts)


def strip_context_leak(text: str) -> str:
    if not text:
        return text
    if "<<<CONTEXT>>>" not in text and "<<<ENDCONTEXT>>>" not in text:
        return text
    cleaned = text
    cleaned = cleaned.replace("<<<CONTEXT>>>", "").replace("<<<ENDCONTEXT>>>", "")
    cleaned = re.sub(
        r"(Table context rows:|Context rows:|Context title:|Glossary \([^)]*\):)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    lines = cleaned.splitlines()
    cleaned_lines: list[str] = []
    skip_prefixes = (
        "Context title:",
        "Glossary (",
        "Table context rows:",
        "Context rows:",
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if stripped.startswith(skip_prefixes):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
LETTER_PATTERN = re.compile(
    r"[A-Za-zÀ-ÿ\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]"
)
NON_ASCII_LETTER_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]"
)
NON_TRANSLATABLE_PATTERN = re.compile(
    r"^[\s0-9\W_]+$",
    re.UNICODE,
)
_HTTP_CLIENTS: dict[tuple[str, str, int], "HttpClient"] = {}
_HTTP_CLIENTS_LOCK = threading.Lock()


class HttpClient:
    def __init__(self, url: str, api_key: str, timeout: int) -> None:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(f"Invalid URL: {url}")
        self._parsed = parsed
        self._api_key = api_key
        self._timeout = timeout
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        self._path = path
        self._thread_local = threading.local()

    def _new_connection(self) -> http.client.HTTPConnection:
        host = self._parsed.hostname
        port = self._parsed.port
        if self._parsed.scheme == "https":
            return http.client.HTTPSConnection(
                host,
                port or 443,
                timeout=self._timeout,
            )
        return http.client.HTTPConnection(
            host,
            port or 80,
            timeout=self._timeout,
        )

    def _get_connection(self) -> http.client.HTTPConnection:
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._thread_local.conn = conn
        return conn

    def post_json(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Connection": "keep-alive",
        }
        conn = self._get_connection()
        try:
            conn.request("POST", self._path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            if resp.status >= 400:
                message = resp_body.decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {resp.status}: {message}")
            return json.loads(resp_body)
        except Exception:
            conn.close()
            self._thread_local.conn = None
            raise


def get_http_client(url: str, api_key: str, timeout: int) -> HttpClient:
    key = (url, api_key, timeout)
    with _HTTP_CLIENTS_LOCK:
        client = _HTTP_CLIENTS.get(key)
        if client is None:
            client = HttpClient(url, api_key, timeout)
            _HTTP_CLIENTS[key] = client
        return client


def post_request(
    url: str,
    payload: dict,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    client = get_http_client(url, api_key, timeout)
    return client.post_json(payload)


def is_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "maximum context length" in message:
        return True
    if "context length" in message and "input tokens" in message:
        return True
    return "reduce the length of the input messages" in message


def cache_get(cache: dict, cache_lock: threading.Lock, key: tuple) -> str | None:
    with cache_lock:
        return cache.get(key)


def cache_set(cache: dict, cache_lock: threading.Lock, key: tuple, value: str) -> None:
    with cache_lock:
        cache[key] = value


def strip_placeholders(text: str) -> str:
    return re.sub(r"__PH__\d+__", "", text)


def has_translatable_text(text: str) -> bool:
    cleaned = strip_placeholders(text)
    return bool(LETTER_PATTERN.search(cleaned))


def has_translatable_text_strip_tags(text: str) -> bool:
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = strip_placeholders(cleaned)
    return bool(LETTER_PATTERN.search(cleaned))


def should_translate_text(text: str, source_lang: str, strip_tags: bool = False) -> bool:
    cleaned = re.sub(r"<[^>]+>", "", text) if strip_tags else text
    cleaned = strip_placeholders(cleaned)
    if not LETTER_PATTERN.search(cleaned):
        return False
    if NON_TRANSLATABLE_PATTERN.match(cleaned):
        return False
    lang = source_lang.lower()
    if lang.startswith(("zh", "ja", "ko", "ru", "ar")):
        return bool(NON_ASCII_LETTER_PATTERN.search(cleaned))
    return True


def effective_max_tokens(text: str, max_tokens: int, source_lang: str, target_lang: str) -> int:
    lang_src = source_lang.lower()
    lang_tgt = target_lang.lower()
    ratio = 1.3
    if lang_src.startswith("zh") and lang_tgt.startswith("en"):
        ratio = 1.4
    elif lang_src.startswith("en") and lang_tgt.startswith(("zh", "ja", "ko")):
        ratio = 1.1
    elif lang_src.startswith(("zh", "ja", "ko")):
        ratio = 1.3
    estimated = estimate_tokens(strip_placeholders(text))
    scaled = int(estimated * ratio) + 32
    return max(64, min(max_tokens, scaled))


def estimate_output_tokens(text: str, source_lang: str, target_lang: str) -> int:
    lang_src = source_lang.lower()
    lang_tgt = target_lang.lower()
    ratio = 1.3
    if lang_src.startswith("zh") and lang_tgt.startswith("en"):
        ratio = 1.4
    elif lang_src.startswith("en") and lang_tgt.startswith(("zh", "ja", "ko")):
        ratio = 1.1
    elif lang_src.startswith(("zh", "ja", "ko")):
        ratio = 1.3
    estimated = estimate_tokens(strip_placeholders(text))
    return int(estimated * ratio) + 32


def infer_max_tokens_from_chars(max_chars: int, source_lang: str, target_lang: str) -> int:
    if max_chars <= 0:
        return DEFAULT_MAX_TOKENS
    lang_src = source_lang.lower()
    lang_tgt = target_lang.lower()
    ratio = 1.3
    if lang_src.startswith("zh") and lang_tgt.startswith("en"):
        ratio = 1.4
    elif lang_src.startswith("en") and lang_tgt.startswith(("zh", "ja", "ko")):
        ratio = 1.1
    elif lang_src.startswith(("zh", "ja", "ko")):
        ratio = 1.3
    estimated = max_chars // 4
    inferred = int(estimated * ratio) + 64
    return max(128, inferred)


def estimate_chars_per_token(sample: str, source_lang: str) -> float:
    if not sample:
        return 1.2 if source_lang.lower().startswith("zh") else 4.0
    tokens = estimate_tokens(sample)
    if tokens <= 0:
        return 1.2 if source_lang.lower().startswith("zh") else 4.0
    ratio = len(sample) / tokens
    default_ratio = 1.2 if source_lang.lower().startswith("zh") else 4.0
    return max(0.5, min(ratio, default_ratio))


def estimate_prompt_overhead_tokens(
    source_lang: str, target_lang: str, context: str
) -> int:
    prompt = build_gemma_prompt("", source_lang, target_lang, context)
    return estimate_tokens(prompt)


def infer_auto_limits(
    content: str,
    source_lang: str,
    target_lang: str,
    max_context_tokens: int,
    max_tokens_hint: int,
    glossary_context: str,
) -> tuple[int, int, int, float]:
    overhead = estimate_prompt_overhead_tokens(
        source_lang, target_lang, glossary_context
    )
    safety_tokens = 64
    output_tokens = min(max_tokens_hint, max_context_tokens - overhead - safety_tokens)
    output_tokens = max(64, output_tokens)
    input_budget = max_context_tokens - overhead - output_tokens - safety_tokens
    if input_budget < 64:
        input_budget = 64
    sample = content[:2000]
    ratio = estimate_chars_per_token(sample, source_lang)
    max_chars = max(256, int(input_budget * ratio))
    return max_chars, output_tokens, overhead, ratio


def translate_segment(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    context: str,
    max_tokens: int,
) -> str:
    translated, _finish_reason = translate_segment_with_meta(
        text,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        cache_lock,
        persistent_cache,
        context,
        max_tokens,
    )
    return translated


def translate_segment_with_meta(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    context: str,
    max_tokens: int,
) -> tuple[str, str | None]:
    if source_lang == "auto":
        LOGGER.error("source_lang must be specified; auto detection is disabled")
        raise ValueError("source_lang must be specified")
    key = build_cache_key(text, source_lang, target_lang, model, url, context)
    cached = cache_get(cache, cache_lock, key)
    if cached is not None:
        return cached, None
    if persistent_cache is not None:
        cached = persistent_cache.get(key)
        if cached is not None:
            cache_set(cache, cache_lock, key, cached)
            return cached, None

    started = time.monotonic()
    completion_url = url.replace("/v1/chat/completions", "/v1/completions")
    prompt = build_gemma_prompt(text, source_lang, target_lang, context)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": effective_max_tokens(text, max_tokens, source_lang, target_lang),
        "temperature": 0.0,
        "stop": ["<end_of_turn>"],
    }
    response = post_request(completion_url, payload, api_key)
    elapsed = time.monotonic() - started
    tag = current_log_context()
    if tag:
        LOGGER.info("Segment translated [%s]: %.2fs, %d chars", tag, elapsed, len(text))
    else:
        LOGGER.info("Segment translated: %.2fs, %d chars", elapsed, len(text))
    translated = response["choices"][0]["text"]
    finish_reason = response["choices"][0].get("finish_reason")
    cache_set(cache, cache_lock, key, translated)
    if persistent_cache is not None:
        persistent_cache.set(key, translated)
    return translated, finish_reason


def protect_patterns(text: str, patterns: list[re.Pattern]) -> tuple[str, dict]:
    placeholders: dict[str, str] = {}
    counter = 0

    def replacer(match: re.Match) -> str:
        nonlocal counter
        placeholder = f"{PLACEHOLDER_PREFIX}{counter}__"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    for pattern in patterns:
        text = pattern.sub(replacer, text)
    return text, placeholders


def restore_placeholders(text: str, placeholders: dict) -> str:
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)

    if placeholders:
        pattern = re.compile(r"__PH__\d+__?")

        def replacer(match: re.Match) -> str:
            token = match.group(0)
            if token in placeholders:
                return placeholders[token]
            if not token.endswith("__"):
                token_full = f"{token}__"
                if token_full in placeholders:
                    return placeholders[token_full]
            return token

        text = pattern.sub(replacer, text)
    return text


def lang_label(lang_code: str) -> str:
    mapping = {
        "en": "English",
        "it": "Italian",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "zh-hans": "Simplified Chinese",
        "zh-hant": "Traditional Chinese",
    }
    key = lang_code.lower()
    return mapping.get(key, lang_code)


def build_gemma_prompt(
    text: str, source_lang: str, target_lang: str, context: str
) -> str:
    source_label = lang_label(source_lang)
    target_label = lang_label(target_lang)
    custom_prompt = CUSTOM_SYSTEM_PROMPT
    if PROMPT_TEMPLATE is not None:
        values = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_label": source_label,
            "target_label": target_label,
            "text": text,
            "context": context,
            "custom_prompt": custom_prompt,
        }
        try:
            rendered = PROMPT_TEMPLATE.format_map(values)
        except KeyError as exc:
            LOGGER.error("Prompt template missing placeholder: %s", exc)
            raise
        if context and "{context}" not in PROMPT_TEMPLATE:
            LOGGER.warning("Prompt template missing {context}; ignoring context")
        if "{text}" not in PROMPT_TEMPLATE:
            LOGGER.warning("Prompt template missing {text}; appending text at the end")
            rendered = f"{rendered}\n{text}"
        if custom_prompt and "{custom_prompt}" not in PROMPT_TEMPLATE:
            LOGGER.warning(
                "Prompt template missing {custom_prompt}; prepending custom prompt"
            )
            rendered = f"{custom_prompt}\n\n{rendered}"
        return rendered
    extra = f"Additional instructions: {custom_prompt}\n\n" if custom_prompt else ""
    return (
        "<bos>\n"
        "<start_of_turn>user\n"
        f"Translate {source_label} ({source_lang}) to {target_label} ({target_lang}). "
        f"Output only the {target_label} translation. "
        "Keep tokens like __PH__<number>__ unchanged.\n\n"
        f"{extra}{context}\n\n{text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_prompt_template(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_custom_prompt(value: str | None) -> str:
    if not value:
        return ""
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read().strip()
    return value.strip()


def estimate_tokens(text: str) -> int:
    try:
        tokenizer = get_transformers_tokenizer()
    except RuntimeError:
        tokenizer = None
    if tokenizer is None:
        global _TOKENIZER_FALLBACK_WARNING
        if not _TOKENIZER_FALLBACK_WARNING:
            LOGGER.warning(
                "Tokenizer unavailable; falling back to character-based token estimate"
            )
            _TOKENIZER_FALLBACK_WARNING = True
        stripped = strip_placeholders(text)
        non_ascii = sum(1 for ch in stripped if ord(ch) > 127)
        ascii_count = len(stripped) - non_ascii
        return max(1, non_ascii + ascii_count // 4)
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def max_group_tokens(max_tokens: int) -> int:
    return max(64, int(max_tokens * 0.8))


def min_group_tokens(max_tokens: int) -> int:
    return max(32, int(max_group_tokens(max_tokens) * 0.2))


TABLE_OUTPUT_RATIO = 0.6
ROW_OUTPUT_RATIO = 0.7


def table_output_limit(max_tokens: int) -> int:
    return max(64, int(max_tokens * TABLE_OUTPUT_RATIO))


def row_output_limit(max_tokens: int) -> int:
    return max(64, int(max_tokens * ROW_OUTPUT_RATIO))


def estimate_row_tokens(text: str) -> int:
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = cleaned.replace("|", " ")
    cleaned = strip_placeholders(cleaned)
    return estimate_tokens(cleaned)


def split_text_by_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_chars and current:
            chunks.append("".join(current))
            current = []
            current_len = 0
        if len(line) > max_chars:
            for start in range(0, len(line), max_chars):
                part = line[start : start + max_chars]
                if current_len + len(part) > max_chars and current:
                    chunks.append("".join(current))
                    current = []
                    current_len = 0
                current.append(part)
                current_len += len(part)
            continue
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def merge_texts_with_bisect(texts: list[str], max_chars: int) -> list[str]:
    if not texts:
        return []
    if max_chars <= 0:
        return ["".join(texts)]

    def split(text_list: list[str]) -> list[str]:
        combined = "".join(text_list)
        if len(combined) <= max_chars:
            return [combined]
        if len(text_list) == 1:
            return split_text_by_chars(combined, max_chars)
        mid = len(text_list) // 2
        return split(text_list[:mid]) + split(text_list[mid:])

    return split(texts)


def merge_segments_by_kind(
    segments: list[tuple[str, str, str]],
    max_chars: int,
    max_tokens: int,
    mergeable_kinds: set[str],
) -> list[tuple[str, str, str]]:
    merged: list[tuple[str, str, str]] = []
    idx = 0
    while idx < len(segments):
        kind, text, context = segments[idx]
        if kind in mergeable_kinds:
            texts = [text]
            idx += 1
            while (
                idx < len(segments)
                and segments[idx][0] == kind
                and segments[idx][2] == context
            ):
                texts.append(segments[idx][1])
                idx += 1
            token_sizes = [estimate_tokens(item) for item in texts]
            groups = build_token_groups(
                list(range(len(texts))),
                token_sizes,
                max_group_tokens(max_tokens),
                min_group_tokens(max_tokens),
            )
            for group in groups:
                combined = "".join(texts[i] for i in group)
                for merged_text in split_text_by_chars(combined, max_chars):
                    merged.append((kind, merged_text, context))
            continue
        merged.append((kind, text, context))
        idx += 1
    return merged


def build_token_groups(
    indices: list[int],
    token_sizes: list[int],
    max_tokens_group: int,
    min_tokens_group: int,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_tokens = 0

    for idx in indices:
        token_size = token_sizes[idx]
        if current_group and current_tokens + token_size > max_tokens_group:
            groups.append(current_group)
            current_group = []
            current_tokens = 0
        current_group.append(idx)
        current_tokens += token_size
    if current_group:
        groups.append(current_group)

    merged = True
    while merged and len(groups) > 1:
        merged = False
        new_groups: list[list[int]] = []
        idx = 0
        while idx < len(groups):
            group = groups[idx]
            group_tokens = sum(token_sizes[i] for i in group)
            if group_tokens >= min_tokens_group or len(groups) == 1:
                new_groups.append(group)
                idx += 1
                continue
            if idx + 1 < len(groups):
                next_group = groups[idx + 1]
                next_tokens = sum(token_sizes[i] for i in next_group)
                if group_tokens + next_tokens <= max_tokens_group:
                    new_groups.append(group + next_group)
                    idx += 2
                    merged = True
                    continue
            if new_groups:
                prev_group = new_groups.pop()
                prev_tokens = sum(token_sizes[i] for i in prev_group)
                if prev_tokens + group_tokens <= max_tokens_group:
                    new_groups.append(prev_group + group)
                    idx += 1
                    merged = True
                    continue
                new_groups.append(prev_group)
            new_groups.append(group)
            idx += 1
        groups = new_groups
    return groups


def build_min_token_groups(
    indices: list[int],
    token_sizes: list[int],
    min_tokens_group: int,
    max_tokens_group: int,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx in indices:
        size = token_sizes[idx]
        if current and current_tokens + size > max_tokens_group:
            groups.append(current)
            current = [idx]
            current_tokens = size
            continue
        current.append(idx)
        current_tokens += size
        if min_tokens_group > 0 and current_tokens >= min_tokens_group:
            groups.append(current)
            current = []
            current_tokens = 0
    if current:
        groups.append(current)
    return groups


def iter_sep_values() -> list[str]:
    return [
        "\n<<<CELL_SEP>>>\n",
        "\n<<<NODE_SEP>>>\n",
        "\n<<SEP>>\n",
        "\n###SEP###\n",
        "\n|||SEP|||\n",
    ]


def normalize_table_batch_mode(value: str | None) -> str:
    if not value:
        return DEFAULT_TABLE_BATCH_MODE
    value = value.strip().lower()
    if value in {"table", "row", "cell"}:
        return value
    return DEFAULT_TABLE_BATCH_MODE


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    allowed = set("|-: ")
    return all(ch in allowed for ch in stripped)

def is_html_line(line: str) -> bool:
    return bool(re.search(r"<(table|thead|tbody|tfoot|tr|td|th|p|div|span|br|img)\b", line, re.IGNORECASE))


def is_markdown_table_candidate(line: str) -> bool:
    if is_html_line(line):
        return False
    return "|" in line


def translate_html_fragment(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    base_context: str,
    max_tokens: int,
    short_row_tokens: int,
    context_rows: int,
    min_cell_tokens: int,
    table_batch_mode: str,
) -> str:
    row_pattern = re.compile(r"(<tr\b[^>]*>.*?</tr>)", re.IGNORECASE | re.DOTALL)
    matches = list(row_pattern.finditer(fragment))
    if matches:
        rows = [match.group(1) for match in matches]
        if table_batch_mode == "table":
            input_tokens = estimate_tokens(fragment)
            output_tokens = estimate_output_tokens(
                fragment, source_lang, target_lang
            )
            if input_tokens > input_token_limit(max_tokens, base_context):
                LOGGER.info(
                    "HTML table skip: token_exceed tokens=%d limit=%d rows=%d",
                    input_tokens,
                    input_token_limit(max_tokens, base_context),
                    len(rows),
                )
            elif output_tokens > table_output_limit(max_tokens):
                LOGGER.info(
                    "HTML table skip: output_risk output=%d limit=%d rows=%d",
                    output_tokens,
                    table_output_limit(max_tokens),
                    len(rows),
                )
            elif input_tokens <= max_group_tokens(max_tokens):
                try:
                    translated, truncated, unresolved = translate_text_fragment_with_meta(
                        fragment,
                        source_lang,
                        target_lang,
                        model,
                        url,
                        api_key,
                        cache,
                        cache_lock,
                        persistent_cache,
                        base_context,
                        max_tokens,
                    )
                except RuntimeError as exc:
                    if not is_context_length_error(exc):
                        raise
                    translated = None
                    truncated = True
                    unresolved = True
                if translated is not None and not truncated and not unresolved:
                    cleaned = strip_context_leak(translated)
                    if not contains_prompt_leak(cleaned):
                        tr_open = len(
                            re.findall(r"<tr\\b", cleaned, flags=re.IGNORECASE)
                        )
                        tr_close = len(
                            re.findall(r"</tr>", cleaned, flags=re.IGNORECASE)
                        )
                        if tr_open >= len(rows) and tr_close >= len(rows):
                            return cleaned
        patterns = [
            re.compile(r"`+[^`]*?`+"),
            re.compile(r"\$\$[\s\S]*?\$\$"),
            re.compile(r"\$[^\n\$]+\$"),
            re.compile(r"\\\[[\s\S]*?\\\]"),
            re.compile(r"\\\([\s\S]*?\\\)"),
            re.compile(r"<[^>]+>"),
            re.compile(r"!\[[^\]]*?\]\([^)]+?\)"),
            re.compile(r"\[[^\]]+?\]\([^)]+?\)"),
            re.compile(r"https?://\S+"),
        ]

        def translate_row_group(
            row_items: list[str], context: str
        ) -> list[str] | None:
            row_parts: list[list[str]] = []
            node_refs: list[tuple[int, int, str, str]] = []
            placeholders: dict[str, str] = {}
            counter = 0
            for row_idx, row in enumerate(row_items):
                parts = re.split(r"(<[^>]+>)", row)
                row_parts.append(parts)
                for part_idx, part in enumerate(parts):
                    if not part or part.startswith("<"):
                        continue
                    if not part.strip():
                        continue
                    if not should_translate_text(part, source_lang, strip_tags=True):
                        continue
                    protected, local_placeholders = protect_patterns(part, patterns)
                    for ph, original in local_placeholders.items():
                        new_ph = f"{PLACEHOLDER_PREFIX}{counter}__"
                        counter += 1
                        protected = protected.replace(ph, new_ph)
                        placeholders[new_ph] = original
                    node_refs.append((row_idx, part_idx, protected, part))

            if not node_refs:
                return row_items

            tiny_nodes = sum(1 for _row_idx, _part_idx, protected, _raw in node_refs if estimate_tokens(protected) <= 4)
            if tiny_nodes:
                LOGGER.info(
                    "HTML table tiny nodes: %d of %d (<=4 tokens)",
                    tiny_nodes,
                    len(node_refs),
                )

            if table_batch_mode == "cell":
                return None

            def translate_node_batch(
                nodes: list[tuple[int, int, str, str]]
            ) -> list[str] | None:
                parts: list[str] = []
                for idx, (_row_idx, _part_idx, protected, _raw) in enumerate(nodes):
                    parts.append(f'<span data-idx="{idx}">{protected}</span>')
                wrapped = "<div>" + "".join(parts) + "</div>"
                translated, finish_reason = translate_segment_with_meta(
                    wrapped,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
                if finish_reason == "length":
                    return None
                translated = strip_context_leak(translated)
                spans = re.findall(
                    r'<span\\s+data-idx="(\\d+)">(.*?)</span>',
                    translated,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if len(spans) != len(nodes):
                    return None
                result_map: dict[int, str] = {}
                for idx_text, content in spans:
                    try:
                        idx_int = int(idx_text)
                    except ValueError:
                        return None
                    result_map[idx_int] = content
                results: list[str] = []
                for idx in range(len(nodes)):
                    if idx not in result_map:
                        return None
                    restored = restore_placeholders(result_map[idx], placeholders)
                    if "__PH" in restored:
                        return None
                    if contains_prompt_leak(restored):
                        return None
                    results.append(strip_context_leak(restored))
                return results

            def translate_nodes_recursive(
                nodes: list[tuple[int, int, str, str]],
                log_stats: bool = False,
            ) -> list[str] | None:
                if not nodes:
                    return []
                min_tokens_group = max(0, min_cell_tokens)
                max_tokens_group = max_group_tokens(max_tokens)
                indices = list(range(len(nodes)))
                groups = build_token_groups(
                    indices,
                    [estimate_tokens(node[2]) for node in nodes],
                    max_tokens_group,
                    min_tokens_group,
                )
                if log_stats:
                    group_sizes = [len(group) for group in groups]
                    if len(group_sizes) <= 12:
                        size_summary = ",".join(str(size) for size in group_sizes)
                    else:
                        size_summary = (
                            f"count={len(group_sizes)} "
                            f"min={min(group_sizes)} "
                            f"max={max(group_sizes)} "
                            f"avg={sum(group_sizes)/len(group_sizes):.1f}"
                        )
                    LOGGER.info(
                        "HTML table batch stats: mode=%s nodes=%d groups=%d sizes=%s",
                        table_batch_mode,
                        len(nodes),
                        len(groups),
                        size_summary,
                    )
                results: list[str] = []
                for group in groups:
                    batch_nodes = [nodes[i] for i in group]
                    batch = translate_node_batch(batch_nodes)
                    if batch is None:
                        if len(batch_nodes) == 1:
                            _row_idx, _part_idx, protected, raw = batch_nodes[0]
                            try:
                                with log_context("html_table:cell"):
                                    translated, finish_reason = translate_segment_with_meta(
                                        protected,
                                        source_lang,
                                        target_lang,
                                        model,
                                        url,
                                        api_key,
                                        cache,
                                        cache_lock,
                                        persistent_cache,
                                        context,
                                        max_tokens,
                                    )
                            except RuntimeError as exc:
                                if is_context_length_error(exc):
                                    results.append(raw)
                                    continue
                                raise
                            if finish_reason == "length":
                                results.append(raw)
                                continue
                            restored = restore_placeholders(translated, placeholders)
                            if "__PH" in restored:
                                results.append(raw)
                                continue
                            if contains_prompt_leak(restored):
                                results.append(raw)
                                continue
                            results.append(strip_context_leak(restored))
                            continue
                        mid = len(batch_nodes) // 2
                        left = translate_nodes_recursive(batch_nodes[:mid])
                        right = translate_nodes_recursive(batch_nodes[mid:])
                        if left is None or right is None:
                            return None
                        results.extend(left)
                        results.extend(right)
                        continue
                    results.extend(batch)
                return results

            translated_nodes = translate_nodes_recursive(node_refs, log_stats=True)
            if translated_nodes is None:
                return None
            for (row_idx, part_idx, _protected, _raw), text in zip(
                node_refs, translated_nodes
            ):
                row_parts[row_idx][part_idx] = text
            return ["".join(parts) for parts in row_parts]

        def split_rows_by_tokens(
            row_items: list[str], token_sizes: list[int], context: str
        ) -> tuple[list[str], list[int], list[str], list[int]]:
            limit = input_token_limit(max_tokens, context)
            cumulative = 0
            split_idx = 0
            for idx, size in enumerate(token_sizes):
                if split_idx > 0 and cumulative + size > limit:
                    break
                cumulative += size
                split_idx = idx + 1
                if cumulative >= limit:
                    break
            split_idx = max(1, min(split_idx, len(row_items) - 1))
            LOGGER.info(
                "HTML table split rows by limit: left=%d right=%d tokens_left=%d limit=%d",
                split_idx,
                len(row_items) - split_idx,
                sum(token_sizes[:split_idx]),
                limit,
            )
            return (
                row_items[:split_idx],
                token_sizes[:split_idx],
                row_items[split_idx:],
                token_sizes[split_idx:],
            )

        def translate_row_group_raw(rows: list[str], context: str) -> list[str] | None:
            combined = "".join(rows)
            combined_tokens = estimate_tokens(combined)
            if combined_tokens > input_token_limit(max_tokens, context):
                LOGGER.info(
                    "HTML table row_raw fail: token_exceed tokens=%d limit=%d rows=%d",
                    combined_tokens,
                    input_token_limit(max_tokens, context),
                    len(rows),
                )
                return None
            output_tokens = estimate_output_tokens(
                combined, source_lang, target_lang
            )
            if output_tokens > row_output_limit(max_tokens):
                LOGGER.info(
                    "HTML table row_raw fail: output_risk output=%d limit=%d rows=%d",
                    output_tokens,
                    row_output_limit(max_tokens),
                    len(rows),
                )
                return None
            if output_tokens > max_tokens:
                LOGGER.info(
                    "HTML table row_raw fail: output_exceed output=%d max_tokens=%d rows=%d",
                    output_tokens,
                    max_tokens,
                    len(rows),
                )
                return None
            with log_context("html_table:row_raw"):
                translated, finish_reason = translate_segment_with_meta(
                    combined,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
            if finish_reason == "length":
                LOGGER.info(
                    "HTML table row_raw fail: finish_reason=length rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            translated = strip_context_leak(translated)
            if "__PH" in translated:
                LOGGER.info(
                    "HTML table row_raw fail: unresolved_ph rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            if contains_prompt_leak(translated):
                LOGGER.info(
                    "HTML table row_raw fail: prompt_leak rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            matches = list(row_pattern.finditer(translated))
            if not matches:
                if len(rows) == 1 and re.search(r"</?(td|th)\\b", translated, re.IGNORECASE):
                    return [f"<tr>{translated}</tr>"]
                LOGGER.info(
                    "HTML table row_raw fail: no_rows rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            if len(matches) < len(rows):
                LOGGER.info(
                    "HTML table row_raw fail: row_count_mismatch expected=%d got=%d",
                    len(rows),
                    len(matches),
                )
                return None
            if len(matches) > len(rows):
                LOGGER.info(
                    "HTML table row count mismatch: expected=%d got=%d, trimming extras",
                    len(rows),
                    len(matches),
                )
            return [match.group(1) for match in matches[: len(rows)]]

        def translate_rows_recursive(
            row_items: list[str], token_sizes: list[int], context: str
        ) -> list[str]:
            group_tokens = sum(token_sizes)
            if len(row_items) <= 2 and group_tokens < DEFAULT_ROW_RAW_MIN_TOKENS:
                LOGGER.info(
                    "HTML table skip row_raw: small_group rows=%d tokens=%d",
                    len(row_items),
                    group_tokens,
                )
                if table_batch_mode != "cell":
                    group_result = translate_row_group(row_items, context)
                    if group_result is not None:
                        return group_result
            else:
                if table_batch_mode != "cell":
                    raw_group = translate_row_group_raw(row_items, context)
                    if raw_group is not None:
                        return raw_group
                    group_result = translate_row_group(row_items, context)
                    if group_result is not None:
                        return group_result

            if len(row_items) == 1:
                row = row_items[0]
                parts = re.split(r"(<[^>]+>)", row)
                out_parts: list[str] = []
                for part in parts:
                    if not part or part.startswith("<"):
                        out_parts.append(part)
                        continue
                    if not part.strip():
                        out_parts.append(part)
                        continue
                    LOGGER.info(
                        "HTML table fallback: row->cell_text rows=%d tokens=%d",
                        len(row_items),
                        token_sizes[0] if token_sizes else 0,
                    )
                    with log_context("html_table:cell_text"):
                        out_parts.append(
                            translate_text_fragment(
                                part,
                                source_lang,
                                target_lang,
                                model,
                                url,
                                api_key,
                                cache,
                                cache_lock,
                                persistent_cache,
                                context,
                                max_tokens,
                            )
                        )
                return ["".join(out_parts)]
            LOGGER.info(
                "HTML table fallback: rows->split rows=%d tokens=%d",
                len(row_items),
                sum(token_sizes),
            )
            left_rows, left_tokens, right_rows, right_tokens = split_rows_by_tokens(
                row_items, token_sizes, context
            )
            return translate_rows_recursive(
                left_rows, left_tokens, context
            ) + translate_rows_recursive(right_rows, right_tokens, context)

        row_tokens = [
            max(estimate_row_tokens(row), estimate_tokens(row)) for row in rows
        ]
        tiny_rows = sum(1 for size in row_tokens if size <= 8)
        if tiny_rows:
            LOGGER.info(
                "HTML table tiny rows: %d of %d (<=8 tokens)",
                tiny_rows,
                len(rows),
            )
        max_tokens_group = max_group_tokens(max_tokens)
        min_tokens_group = min_group_tokens(max_tokens)

        def chunk_rows_by_limit() -> list[list[int]]:
            chunks: list[list[int]] = []
            current: list[int] = []
            current_tokens = 0
            limit = input_token_limit(max_tokens, base_context)
            for idx, size in enumerate(row_tokens):
                if current and current_tokens + size > limit:
                    chunks.append(current)
                    current = []
                    current_tokens = 0
                current.append(idx)
                current_tokens += size
                if current_tokens >= limit:
                    chunks.append(current)
                    current = []
                    current_tokens = 0
            if current:
                chunks.append(current)
            return chunks

        groups = build_token_groups(
            list(range(len(rows))),
            row_tokens,
            max_tokens_group,
            min_tokens_group,
        )
        if table_batch_mode == "table":
            groups = chunk_rows_by_limit()
            LOGGER.info(
                "HTML table chunked groups: %d (limit=%d)",
                len(groups),
                input_token_limit(max_tokens, base_context),
            )
        elif len(groups) > 1:
            LOGGER.info(
                "HTML table row groups: %d (min_tokens=%d max_tokens=%d)",
                len(groups),
                min_tokens_group,
                max_tokens_group,
            )

        translated_rows: list[str] = [""] * len(rows)

        def merge_short_indices() -> list[list[int]]:
            merged: list[list[int]] = []
            current: list[int] = []
            current_tokens = 0
            for idx, size in enumerate(row_tokens):
                if size < short_row_tokens:
                    current.append(idx)
                    current_tokens += size
                    if current_tokens >= short_group_target:
                        merged.append(current)
                        current = []
                        current_tokens = 0
                else:
                    if current:
                        merged.append(current)
                        current = []
                        current_tokens = 0
            if current:
                merged.append(current)
            return merged

        translated_mask = [False] * len(rows)
        short_group_target = max(short_row_tokens, min_group_tokens(max_tokens))
        if short_row_tokens > 0:
            short_runs = merge_short_indices()
            if (
                len(short_runs) >= 2
                and sum(row_tokens[i] for i in short_runs[-1]) < short_group_target
            ):
                prev_tokens = sum(row_tokens[i] for i in short_runs[-2])
                last_tokens = sum(row_tokens[i] for i in short_runs[-1])
                if prev_tokens + last_tokens <= max_group_tokens(max_tokens):
                    short_runs[-2].extend(short_runs[-1])
                    short_runs.pop()
            for group in short_runs:
                group_rows = [rows[i] for i in group]
                group_tokens = [row_tokens[i] for i in group]
                context = build_table_context(
                    base_context, rows, group[0], context_rows
                )
                group_translated = translate_rows_recursive(
                    group_rows, group_tokens, context
                )
                for idx, translated in zip(group, group_translated):
                    translated_rows[idx] = translated
                    translated_mask[idx] = True

        def iter_untranslated_subgroups(group: list[int]) -> list[list[int]]:
            subgroups: list[list[int]] = []
            current: list[int] = []
            for idx in group:
                if translated_mask[idx]:
                    if current:
                        subgroups.append(current)
                        current = []
                    continue
                current.append(idx)
            if current:
                subgroups.append(current)
            return subgroups

        for group in groups:
            for sub in iter_untranslated_subgroups(group):
                group_rows = [rows[i] for i in sub]
                group_tokens = [row_tokens[i] for i in sub]
                context = build_table_context(
                    base_context, rows, sub[0], context_rows
                )
                group_translated = translate_rows_recursive(
                    group_rows, group_tokens, context
                )
                for idx, translated in zip(sub, group_translated):
                    translated_rows[idx] = translated
                    translated_mask[idx] = True
        result = []
        last_end = 0
        for match, translated_row in zip(matches, translated_rows):
            result.append(fragment[last_end:match.start()])
            result.append(translated_row)
            last_end = match.end()
        result.append(fragment[last_end:])
        return "".join(result)

    parts = re.split(r"(<[^>]+>)", fragment)
    out_parts = []
    text_buffer = []

    def flush_buffer() -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        buf_text = "".join(text_buffer)
        if buf_text.strip():
            out_parts.append(
                translate_text_fragment(
                    buf_text,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    base_context,
                    max_tokens,
                )
            )
        else:
            out_parts.append(buf_text)
        text_buffer = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            flush_buffer()
            out_parts.append(part)
            continue
        if not part.strip():
            text_buffer.append(part)
            continue
        text_buffer.append(part)
        if len("".join(text_buffer)) >= 200:
            flush_buffer()
    flush_buffer()
    return "".join(out_parts)


def translate_text_fragment(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    context: str,
    max_tokens: int,
) -> str:
    if not fragment.strip():
        return fragment

    patterns = [
        re.compile(r"`+[^`]*?`+"),
        re.compile(r"\$\$[\s\S]*?\$\$"),
        re.compile(r"\$[^\n\$]+\$"),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(r"\\\([\s\S]*?\\\)"),
        re.compile(r"<[^>]+>"),
        re.compile(r"!\[[^\]]*?\]\([^)]+?\)"),
        re.compile(r"\[[^\]]+?\]\([^)]+?\)"),
        re.compile(r"https?://\S+"),
    ]

    protected, placeholders = protect_patterns(fragment, patterns)
    if not should_translate_text(protected, source_lang):
        return fragment

    translated = translate_segment(
        protected,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        cache_lock,
        persistent_cache,
        context,
        max_tokens,
    )
    restored = restore_placeholders(translated, placeholders)
    if "__PH" in restored:
        LOGGER.warning("Unresolved placeholders detected; keeping original fragment")
        return fragment
    return strip_context_leak(restored)


def translate_text_fragment_with_meta(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    context: str,
    max_tokens: int,
) -> tuple[str, bool, bool]:
    if not fragment.strip():
        return fragment, False, False

    patterns = [
        re.compile(r"`+[^`]*?`+"),
        re.compile(r"\$\$[\s\S]*?\$\$"),
        re.compile(r"\$[^\n\$]+\$"),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(r"\\\([\s\S]*?\\\)"),
        re.compile(r"<[^>]+>"),
        re.compile(r"!\[[^\]]*?\]\([^)]+?\)"),
        re.compile(r"\[[^\]]+?\]\([^)]+?\)"),
        re.compile(r"https?://\S+"),
    ]

    protected, placeholders = protect_patterns(fragment, patterns)
    if not should_translate_text(protected, source_lang):
        return fragment, False, False

    translated, finish_reason = translate_segment_with_meta(
        protected,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        cache_lock,
        persistent_cache,
        context,
        max_tokens,
    )
    restored = restore_placeholders(translated, placeholders)
    if "__PH" in restored:
        LOGGER.warning("Unresolved placeholders detected; keeping original fragment")
        return fragment, finish_reason == "length", True
    return strip_context_leak(restored), finish_reason == "length", False


def translate_markdown_table_fragment(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    base_context: str,
    max_tokens: int,
    short_row_tokens: int,
    context_rows: int,
    min_cell_tokens: int,
    table_batch_mode: str,
) -> str:
    lines = fragment.splitlines(keepends=True)
    row_indices: list[int] = []
    row_lines: list[str] = []
    for idx, line in enumerate(lines):
        if is_table_separator(line):
            continue
        row_indices.append(idx)
        row_lines.append(line)

    if not row_lines:
        return fragment

    patterns = [
        re.compile(r"`+[^`]*?`+"),
        re.compile(r"\$\$[\s\S]*?\$\$"),
        re.compile(r"\$[^\n\$]+\$"),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(r"\\\([\s\S]*?\\\)"),
        re.compile(r"<[^>]+>"),
        re.compile(r"!\[[^\]]*?\]\([^)]+?\)"),
        re.compile(r"\[[^\]]+?\]\([^)]+?\)"),
        re.compile(r"https?://\S+"),
    ]

    def translate_row_group(
        row_items: list[str], context: str
    ) -> list[str] | None:
        if table_batch_mode == "cell":
            return None
        row_cells: list[list[str]] = []
        column_nodes: dict[int, list[tuple[int, int, str, str, str, str]]] = {}
        for row_idx, row in enumerate(row_items):
            cells = row.split("|")
            row_cells.append(cells)
            for cell_idx in range(1, max(1, len(cells) - 1)):
                cell = cells[cell_idx]
                if not cell.strip():
                    continue
                leading = re.match(r"\s*", cell).group(0)
                trailing = re.search(r"\s*$", cell).group(0)
                inner = cell[len(leading) : len(cell) - len(trailing)]
                if not inner:
                    continue
                if not should_translate_text(inner, source_lang, strip_tags=True):
                    continue
                column_nodes.setdefault(cell_idx, []).append(
                    (row_idx, cell_idx, leading, trailing, inner, inner)
                )

        if not column_nodes:
            return row_items

        if table_batch_mode == "row":
            row_nodes: list[list[tuple[int, int, str, str, str, str]]] = [
                [] for _ in row_cells
            ]
            for cell_idx, nodes in column_nodes.items():
                for row_idx, _cell_idx, leading, trailing, inner, raw in nodes:
                    row_nodes[row_idx].append(
                        (row_idx, cell_idx, leading, trailing, inner, raw)
                    )
            batch_sets = [(None, nodes) for nodes in row_nodes if nodes]
        else:
            batch_sets = list(column_nodes.items())

        for cell_idx, nodes in batch_sets:
            placeholders: dict[str, str] = {}
            protected_nodes: list[tuple[int, int, str, str, str, str]] = []
            counter = 0
            for row_idx, node_cell_idx, leading, trailing, inner, raw in nodes:
                target_cell_idx = node_cell_idx if cell_idx is None else cell_idx
                protected, local_placeholders = protect_patterns(inner, patterns)
                for ph, original in local_placeholders.items():
                    new_ph = f"{PLACEHOLDER_PREFIX}{counter}__"
                    counter += 1
                    protected = protected.replace(ph, new_ph)
                    placeholders[new_ph] = original
                protected_nodes.append(
                    (row_idx, target_cell_idx, leading, trailing, protected, raw)
                )
            tiny_nodes = sum(1 for node in protected_nodes if estimate_tokens(node[4]) <= 4)
            if tiny_nodes:
                LOGGER.info(
                    "Markdown table tiny nodes: %d of %d (<=4 tokens)",
                    tiny_nodes,
                    len(protected_nodes),
                )

            def translate_node_batch(
                batch_nodes: list[tuple[int, int, str, str, str, str]]
            ) -> list[str] | None:
                parts: list[str] = []
                for idx, (_row_idx, _cell_idx, _leading, _trailing, protected, _raw) in enumerate(batch_nodes):
                    parts.append(f'<span data-idx="{idx}">{protected}</span>')
                wrapped = "<div>" + "".join(parts) + "</div>"
                translated, finish_reason = translate_segment_with_meta(
                    wrapped,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
                if finish_reason == "length":
                    return None
                translated = strip_context_leak(translated)
                spans = re.findall(
                    r'<span\\s+data-idx="(\\d+)">(.*?)</span>',
                    translated,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if len(spans) != len(batch_nodes):
                    return None
                result_map: dict[int, str] = {}
                for idx_text, content in spans:
                    try:
                        idx_int = int(idx_text)
                    except ValueError:
                        return None
                    result_map[idx_int] = content
                results: list[str] = []
                for idx in range(len(batch_nodes)):
                    if idx not in result_map:
                        return None
                    restored = restore_placeholders(result_map[idx], placeholders)
                    if "__PH" in restored:
                        return None
                    results.append(strip_context_leak(restored))
                return results

            def translate_nodes_recursive(
                batch_nodes: list[tuple[int, int, str, str, str, str]],
                log_stats: bool = False,
            ) -> list[str] | None:
                batch = translate_node_batch(batch_nodes)
                if batch is not None:
                    return batch
                if len(batch_nodes) == 1:
                    row_idx, target_cell_idx, leading, trailing, protected, raw = batch_nodes[0]
                    try:
                        with log_context("md_table:cell"):
                            translated, finish_reason = translate_segment_with_meta(
                                protected,
                                source_lang,
                                target_lang,
                                model,
                                url,
                                api_key,
                                cache,
                                cache_lock,
                                persistent_cache,
                                context,
                                max_tokens,
                            )
                    except RuntimeError as exc:
                        if is_context_length_error(exc):
                            return [raw]
                        raise
                    if finish_reason == "length":
                        return [raw]
                    restored = restore_placeholders(translated, placeholders)
                    if "__PH" in restored:
                        return [raw]
                    if contains_prompt_leak(restored):
                        return [raw]
                    return [strip_context_leak(restored)]
                mid = len(batch_nodes) // 2
                left = translate_nodes_recursive(batch_nodes[:mid])
                right = translate_nodes_recursive(batch_nodes[mid:])
                if left is None or right is None:
                    return None
                return left + right

            min_tokens_group = max(0, min_cell_tokens)
            max_tokens_group = max_group_tokens(max_tokens)
            token_sizes = [estimate_tokens(node[4]) for node in protected_nodes]
            groups = build_token_groups(
                list(range(len(protected_nodes))),
                token_sizes,
                max_tokens_group,
                min_tokens_group,
            )
            if groups:
                group_sizes = [len(group) for group in groups]
                if len(group_sizes) <= 12:
                    size_summary = ",".join(str(size) for size in group_sizes)
                else:
                    size_summary = (
                        f"count={len(group_sizes)} "
                        f"min={min(group_sizes)} "
                        f"max={max(group_sizes)} "
                        f"avg={sum(group_sizes)/len(group_sizes):.1f}"
                    )
                LOGGER.info(
                    "Markdown table batch stats: nodes=%d groups=%d sizes=%s",
                    len(protected_nodes),
                    len(groups),
                    size_summary,
                )
            translated_nodes: list[str] = []
            for group in groups:
                batch_nodes = [protected_nodes[i] for i in group]
                batch = translate_nodes_recursive(batch_nodes)
                if batch is None:
                    return None
                translated_nodes.extend(batch)
            if translated_nodes is None:
                return None
            for (
                row_idx,
                target_cell_idx,
                leading,
                trailing,
                _protected,
                _raw,
            ), text in zip(
                protected_nodes, translated_nodes
            ):
                row_cells[row_idx][target_cell_idx] = leading + text + trailing

        return ["|".join(cells) for cells in row_cells]

    def split_rows_by_tokens(
        row_items: list[str], token_sizes: list[int], context: str
    ) -> tuple[list[str], list[int], list[str], list[int]]:
        limit = input_token_limit(max_tokens, context)
        cumulative = 0
        split_idx = 0
        for idx, size in enumerate(token_sizes):
            if split_idx > 0 and cumulative + size > limit:
                break
            cumulative += size
            split_idx = idx + 1
            if cumulative >= limit:
                break
        split_idx = max(1, min(split_idx, len(row_items) - 1))
        LOGGER.info(
            "Markdown table split rows by limit: left=%d right=%d tokens_left=%d limit=%d",
            split_idx,
            len(row_items) - split_idx,
            sum(token_sizes[:split_idx]),
            limit,
        )
        return (
            row_items[:split_idx],
            token_sizes[:split_idx],
            row_items[split_idx:],
            token_sizes[split_idx:],
        )

    def translate_rows_recursive(
        row_items: list[str], token_sizes: list[int], context: str
    ) -> list[str]:
        def translate_row_group_raw(rows: list[str], context: str) -> list[str] | None:
            combined = "".join(rows)
            combined_tokens = estimate_tokens(combined)
            if combined_tokens > input_token_limit(max_tokens, context):
                LOGGER.info(
                    "Markdown table row_raw fail: token_exceed tokens=%d limit=%d rows=%d",
                    combined_tokens,
                    input_token_limit(max_tokens, context),
                    len(rows),
                )
                return None
            output_tokens = estimate_output_tokens(
                combined, source_lang, target_lang
            )
            if output_tokens > row_output_limit(max_tokens):
                LOGGER.info(
                    "Markdown table row_raw fail: output_risk output=%d limit=%d rows=%d",
                    output_tokens,
                    row_output_limit(max_tokens),
                    len(rows),
                )
                return None
            if output_tokens > max_tokens:
                LOGGER.info(
                    "Markdown table row_raw fail: output_exceed output=%d max_tokens=%d rows=%d",
                    output_tokens,
                    max_tokens,
                    len(rows),
                )
                return None
            with log_context("md_table:row_raw"):
                translated, truncated, unresolved = translate_text_fragment_with_meta(
                    combined,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
            if truncated or unresolved:
                LOGGER.info(
                    "Markdown table row_raw fail: truncated=%s unresolved=%s rows=%d tokens=%d",
                    truncated,
                    unresolved,
                    len(rows),
                    combined_tokens,
                )
                return None
            translated = strip_context_leak(translated)
            if contains_prompt_leak(translated):
                LOGGER.info(
                    "Markdown table row_raw fail: prompt_leak rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            lines = translated.splitlines(keepends=True)
            if not lines:
                LOGGER.info(
                    "Markdown table row_raw fail: no_rows rows=%d tokens=%d",
                    len(rows),
                    combined_tokens,
                )
                return None
            if len(lines) < len(rows):
                LOGGER.info(
                    "Markdown table row_raw fail: row_count_mismatch expected=%d got=%d",
                    len(rows),
                    len(lines),
                )
                return None
            if len(lines) > len(rows):
                LOGGER.info(
                    "Markdown table row count mismatch: expected=%d got=%d, trimming extras",
                    len(rows),
                    len(lines),
                )
            return lines[: len(rows)]

        if table_batch_mode != "cell":
            raw_group = translate_row_group_raw(row_items, context)
            if raw_group is not None:
                return raw_group
            group_result = translate_row_group(row_items, context)
            if group_result is not None:
                return group_result
        if len(row_items) == 1:
            row = row_items[0]
            cells = row.split("|")
            for cell_idx in range(1, max(1, len(cells) - 1)):
                cell = cells[cell_idx]
                if not cell.strip():
                    continue
                leading = re.match(r"\s*", cell).group(0)
                trailing = re.search(r"\s*$", cell).group(0)
                inner = cell[len(leading) : len(cell) - len(trailing)]
                if not inner:
                    continue
                LOGGER.info(
                    "Markdown table fallback: row->cell_text rows=%d tokens=%d",
                    len(row_items),
                    token_sizes[0] if token_sizes else 0,
                )
                with log_context("md_table:cell_text"):
                    cells[cell_idx] = (
                        leading
                        + translate_text_fragment(
                            inner,
                            source_lang,
                            target_lang,
                            model,
                            url,
                            api_key,
                            cache,
                            cache_lock,
                            persistent_cache,
                            context,
                            max_tokens,
                        )
                        + trailing
                    )
            return ["|".join(cells)]
        LOGGER.info(
            "Markdown table fallback: rows->split rows=%d tokens=%d",
            len(row_items),
            sum(token_sizes),
        )
        left_rows, left_tokens, right_rows, right_tokens = split_rows_by_tokens(
            row_items, token_sizes, context
        )
        return translate_rows_recursive(
            left_rows, left_tokens, context
        ) + translate_rows_recursive(right_rows, right_tokens, context)

    row_tokens = [
        max(estimate_row_tokens(row), estimate_tokens(row)) for row in row_lines
    ]
    tiny_rows = sum(1 for size in row_tokens if size <= 8)
    if tiny_rows:
        LOGGER.info(
            "Markdown table tiny rows: %d of %d (<=8 tokens)",
            tiny_rows,
            len(row_lines),
        )
    max_tokens_group = max_group_tokens(max_tokens)
    min_tokens_group = min_group_tokens(max_tokens)

    def chunk_rows_by_limit() -> list[list[int]]:
        chunks: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        limit = input_token_limit(max_tokens, base_context)
        for idx, size in enumerate(row_tokens):
            if current and current_tokens + size > limit:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += size
            if current_tokens >= limit:
                chunks.append(current)
                current = []
                current_tokens = 0
        if current:
            chunks.append(current)
        return chunks

        groups = build_token_groups(
            list(range(len(row_lines))),
            row_tokens,
            max_tokens_group,
            min_tokens_group,
        )
    if table_batch_mode == "table":
        groups = chunk_rows_by_limit()
        LOGGER.info(
            "Markdown table chunked groups: %d (limit=%d)",
            len(groups),
            input_token_limit(max_tokens, base_context),
        )
    elif len(groups) > 1:
        LOGGER.info(
            "Markdown table row groups: %d (min_tokens=%d max_tokens=%d)",
            len(groups),
            min_tokens_group,
            max_tokens_group,
        )

    translated_rows: list[str] = [""] * len(row_lines)

    def merge_short_indices() -> list[list[int]]:
        merged: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for idx, size in enumerate(row_tokens):
            if size < short_row_tokens:
                current.append(idx)
                current_tokens += size
                if current_tokens >= short_group_target:
                    merged.append(current)
                    current = []
                    current_tokens = 0
            else:
                if current:
                    merged.append(current)
                    current = []
                    current_tokens = 0
        if current:
            merged.append(current)
        return merged

    translated_mask = [False] * len(row_lines)
    short_group_target = max(short_row_tokens, min_group_tokens(max_tokens))
    if short_row_tokens > 0:
        short_runs = merge_short_indices()
        if (
            len(short_runs) >= 2
            and sum(row_tokens[i] for i in short_runs[-1]) < short_group_target
        ):
            prev_tokens = sum(row_tokens[i] for i in short_runs[-2])
            last_tokens = sum(row_tokens[i] for i in short_runs[-1])
            if prev_tokens + last_tokens <= max_group_tokens(max_tokens):
                short_runs[-2].extend(short_runs[-1])
                short_runs.pop()
        for group in short_runs:
            group_rows = [row_lines[i] for i in group]
            group_tokens = [row_tokens[i] for i in group]
            context = build_table_context(
                base_context, row_lines, group[0], context_rows
            )
            group_translated = translate_rows_recursive(
                group_rows, group_tokens, context
            )
            for idx, translated in zip(group, group_translated):
                translated_rows[idx] = translated
                translated_mask[idx] = True

    def iter_untranslated_subgroups(group: list[int]) -> list[list[int]]:
        subgroups: list[list[int]] = []
        current: list[int] = []
        for idx in group:
            if translated_mask[idx]:
                if current:
                    subgroups.append(current)
                    current = []
                continue
            current.append(idx)
        if current:
            subgroups.append(current)
        return subgroups

    for group in groups:
        for sub in iter_untranslated_subgroups(group):
            group_rows = [row_lines[i] for i in sub]
            group_tokens = [row_tokens[i] for i in sub]
            context = build_table_context(
                base_context, row_lines, sub[0], context_rows
            )
            group_translated = translate_rows_recursive(
                group_rows, group_tokens, context
            )
            for idx, translated in zip(sub, group_translated):
                translated_rows[idx] = translated
                translated_mask[idx] = True
    for row_idx, translated in zip(row_indices, translated_rows):
        lines[row_idx] = translated

    return "".join(lines)


def translate_line(
    line: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    cache_lock: threading.Lock,
    persistent_cache: PersistentCache | None,
    context: str,
    min_cell_tokens: int,
    max_tokens: int,
    table_batch_mode: str,
) -> str:
    if not line.strip() or is_table_separator(line):
        return line

    if is_html_line(line):
        return translate_html_fragment(
            line,
            source_lang,
            target_lang,
            model,
            url,
            api_key,
            cache,
            cache_lock,
            persistent_cache,
            context,
            max_tokens,
            short_row_tokens=DEFAULT_SHORT_ROW_TOKENS,
            context_rows=0,
            min_cell_tokens=min_cell_tokens,
            table_batch_mode=table_batch_mode,
        )

    header_match = re.match(r"^(\s{0,3}#{1,6}\s+)(.*)$", line)
    if header_match:
        prefix, content = header_match.groups()
        return prefix + translate_text_fragment(
            content,
            source_lang,
            target_lang,
            model,
            url,
            api_key,
            cache,
            cache_lock,
            persistent_cache,
            context,
            max_tokens,
        )

    list_match = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", line)
    if list_match:
        prefix, content = list_match.groups()
        return prefix + translate_text_fragment(
            content,
            source_lang,
            target_lang,
            model,
            url,
            api_key,
            cache,
            cache_lock,
            persistent_cache,
            context,
            max_tokens,
        )

    if "|" in line and not is_table_separator(line):
        parts = line.split("|")
        translated_cells = []
        for idx, cell in enumerate(parts):
            if idx == 0 or idx == len(parts) - 1:
                translated_cells.append(cell)
                continue
            translated_cells.append(
                translate_text_fragment(
                    cell,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
            )
        return "|".join(translated_cells)

    return translate_text_fragment(
        line,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        cache_lock,
        persistent_cache,
        context,
        max_tokens,
    )


def translate_markdown(
    content: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    max_chars: int,
    max_tokens: int,
    workers: int,
    short_row_tokens: int,
    context_rows: int,
    min_cell_tokens: int,
    table_batch_mode: str,
    glossary_max_terms: int,
    persistent_cache: PersistentCache | None,
) -> str:
    lines = content.splitlines(keepends=True)
    in_code_block = False
    fence_pattern = re.compile(r"^\s*(```|~~~)")
    table_open_pattern = re.compile(r"<table\b", re.IGNORECASE)
    table_close_pattern = re.compile(r"</table>", re.IGNORECASE)
    header_pattern = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*)$")
    list_pattern = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)")
    cache: dict = {}
    cache_lock = threading.Lock()
    glossary_terms = (
        extract_glossary_terms(content, glossary_max_terms)
        if glossary_max_terms > 0
        else []
    )
    glossary_context = build_glossary_context(glossary_terms)
    current_title = ""
    segments: list[tuple[str, str, str]] = []
    paragraph_buffer: list[str] = []
    paragraph_len = 0
    table_buffer: list[str] = []
    table_depth = 0
    list_buffer: list[str] = []

    def current_context() -> str:
        return build_context_text(current_title, glossary_context)

    def append_segment(kind: str, text: str, context: str = "") -> None:
        segments.append((kind, text, context))

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer, paragraph_len
        if not paragraph_buffer:
            return
        paragraph = "".join(paragraph_buffer)
        for chunk in split_text_by_chars(paragraph, max_chars):
            append_segment("paragraph", chunk, current_context())
        paragraph_buffer = []
        paragraph_len = 0

    markdown_table_buffer: list[str] = []

    def flush_markdown_table() -> bool:
        nonlocal markdown_table_buffer
        if not markdown_table_buffer:
            return False
        if any(is_table_separator(item) for item in markdown_table_buffer):
            append_segment(
                "markdown_table", "".join(markdown_table_buffer), current_context()
            )
        else:
            paragraph_buffer.append("".join(markdown_table_buffer))
        markdown_table_buffer = []
        return True

    def flush_list_block() -> None:
        nonlocal list_buffer
        if not list_buffer:
            return
        list_text = "".join(list_buffer)
        append_segment("list_block", list_text, current_context())
        list_buffer = []

    for line in lines:
        if fence_pattern.match(line):
            flush_list_block()
            flush_paragraph()
            flush_markdown_table()
            in_code_block = not in_code_block
            append_segment("literal", line, "")
            continue
        if in_code_block:
            append_segment("literal", line, "")
            continue

        if list_buffer and not list_pattern.match(line):
            flush_list_block()

        if markdown_table_buffer:
            if is_markdown_table_candidate(line) or is_table_separator(line):
                markdown_table_buffer.append(line)
                continue
            flush_markdown_table()

        if table_depth > 0 or table_open_pattern.search(line):
            if table_depth == 0:
                flush_list_block()
                flush_markdown_table()
                flush_paragraph()
            table_buffer.append(line)
            opens = len(table_open_pattern.findall(line))
            closes = len(table_close_pattern.findall(line))
            table_depth += opens - closes
            if table_depth <= 0:
                fragment = "".join(table_buffer)
                append_segment("html_table", fragment, current_context())
                table_buffer = []
                table_depth = 0
            continue

        if is_markdown_table_candidate(line):
            flush_list_block()
            flush_paragraph()
            markdown_table_buffer.append(line)
            continue

        if not line.strip():
            flush_list_block()
            flush_markdown_table()
            flush_paragraph()
            append_segment("literal", line, "")
            continue

        if is_table_separator(line):
            flush_list_block()
            flush_paragraph()
            append_segment("literal", line, "")
            continue

        header_match = header_pattern.match(line)
        if header_match:
            flush_list_block()
            flush_markdown_table()
            flush_paragraph()
            append_segment("line", line, current_context())
            current_title = header_match.group(2).strip()
            continue

        if list_pattern.match(line):
            flush_markdown_table()
            flush_paragraph()
            list_buffer.append(line)
            continue

        paragraph_buffer.append(line)
        paragraph_len += len(line)
        if max_chars > 0 and paragraph_len >= max_chars:
            flush_paragraph()

    if table_buffer:
        fragment = "".join(table_buffer)
        append_segment("html_table", fragment, current_context())
    flush_markdown_table()
    flush_list_block()
    flush_paragraph()

    mergeable_kinds = {"paragraph", "line", "list_block"}
    segments = merge_segments_by_kind(
        segments, max_chars, max_tokens, mergeable_kinds
    )
    request_count = sum(1 for kind, _, _ in segments if kind != "literal")
    total_chars = sum(len(text) for kind, text, _ in segments if kind != "literal")
    LOGGER.info("Requests sent: %d, total chars: %d", request_count, total_chars)
    log_segment_stats(segments)

    def process_segment(segment: tuple[str, str, str]) -> str:
        kind, text, context = segment
        with log_context(f"segment:{kind}"):
            if kind == "line":
                return translate_line(
                    text,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    min_cell_tokens,
                    max_tokens,
                    table_batch_mode,
                )
            if kind == "html_table":
                return translate_html_fragment(
                    text,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                    short_row_tokens,
                    context_rows,
                    min_cell_tokens,
                    table_batch_mode,
                )
            if kind == "markdown_table":
                return translate_markdown_table_fragment(
                    text,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                    short_row_tokens,
                    context_rows,
                    min_cell_tokens,
                    table_batch_mode,
                )
            if kind in {"paragraph", "list_block"}:
                return translate_text_fragment(
                    text,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    cache_lock,
                    persistent_cache,
                    context,
                    max_tokens,
                )
            return text

    results: list[str | None] = [None] * len(segments)
    tasks: list[tuple[int, tuple[str, str, str]]] = []
    for idx, segment in enumerate(segments):
        if segment[0] == "literal":
            results[idx] = segment[1]
        else:
            tasks.append((idx, segment))

    if not tasks:
        return "".join(part or "" for part in results)

    if workers <= 1 or len(tasks) == 1:
        for idx, segment in tasks:
            results[idx] = process_segment(segment)
        return "".join(part or "" for part in results)

    task_with_tokens = []
    for idx, segment in tasks:
        _kind, text, _context = segment
        task_with_tokens.append((estimate_tokens(text), idx, segment))
    task_with_tokens.sort(key=lambda item: item[0])

    LOGGER.info("Using %d worker threads", workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_segment, segment): idx
            for _tokens, idx, segment in task_with_tokens
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()

    return "".join(part or "" for part in results)


def log_segment_stats(segments: list[tuple[str, str, str]]) -> None:
    short_char_threshold = 20
    short_token_threshold = 16
    per_kind = Counter()
    per_kind_short_chars = Counter()
    per_kind_short_tokens = Counter()
    short_samples: list[tuple[int, int, str, str]] = []
    for kind, text, _context in segments:
        if kind == "literal":
            continue
        per_kind[kind] += 1
        text_len = len(text)
        token_len = estimate_tokens(text)
        if text_len <= short_char_threshold:
            per_kind_short_chars[kind] += 1
        if token_len <= short_token_threshold:
            per_kind_short_tokens[kind] += 1
        if text_len <= short_char_threshold or token_len <= short_token_threshold:
            preview = text.strip().replace("\n", "\\n")
            if len(preview) > 60:
                preview = preview[:60] + "..."
            short_samples.append((text_len, token_len, kind, preview))
    if not per_kind:
        return
    kind_lines = []
    for kind in sorted(per_kind):
        kind_lines.append(
            f"{kind}: total={per_kind[kind]} short_chars={per_kind_short_chars[kind]} short_tokens={per_kind_short_tokens[kind]}"
        )
    LOGGER.info(
        "Segment stats (short<=%d chars or <=%d tokens): %s",
        short_char_threshold,
        short_token_threshold,
        "; ".join(kind_lines),
    )
    if short_samples:
        short_samples.sort(key=lambda item: (item[0], item[1]))
        sample_lines = [
            f"{length}c/{tokens}t {kind}: {preview}"
            for length, tokens, kind, preview in short_samples[:8]
        ]
        LOGGER.info("Shortest segments: %s", " | ".join(sample_lines))


def translate_whole_file(
    content: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    persistent_cache: PersistentCache | None,
    context: str,
    max_tokens: int,
) -> str:
    cache: dict = {}
    cache_lock = threading.Lock()
    fenced_patterns = [
        re.compile(r"^\s*```[\s\S]*?^\s*```[ \t]*$", re.MULTILINE),
        re.compile(r"^\s*~~~[\s\S]*?^\s*~~~[ \t]*$", re.MULTILINE),
    ]
    protected, placeholders = protect_patterns(content, fenced_patterns)
    translated, truncated, unresolved = translate_text_fragment_with_meta(
        protected,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        cache_lock,
        persistent_cache,
        context,
        max_tokens,
    )
    if truncated:
        raise RuntimeError("Whole-file translation truncated")
    if unresolved:
        LOGGER.warning("Unresolved placeholders detected; keeping original content")
        return content
    restored = restore_placeholders(translated, placeholders)
    if placeholders and re.search(r"__PH__\d+__?", restored):
        LOGGER.warning("Unresolved placeholders detected; keeping original content")
        return content
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate Markdown via TranslateGemma")
    parser.add_argument("--input", "-i", required=True, help="input markdown file")
    parser.add_argument("--output", "-o", help="output markdown file")
    parser.add_argument(
        "--source-lang",
        required=True,
        help="source language code (required)",
    )
    parser.add_argument("--target-lang", default="en", help="target language code")
    parser.add_argument(
        "--prompt-template",
        help="path to prompt template file with placeholders "
        "{source_lang}, {target_lang}, {source_label}, {target_label}, {text}",
    )
    parser.add_argument(
        "--custom-system-prompt",
        help="custom prompt text (or file path) prepended to translation instructions",
    )
    parser.add_argument(
        "--model",
        default="translategemma",
        help="served model name on the vLLM endpoint",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8020/v1/chat/completions",
        help="vLLM OpenAI compatible chat completion endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", ""),
        help="API key for vLLM service (env VLLM_API_KEY)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="max characters per merged paragraph chunk (0=auto)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="max output tokens per request (default from config)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="model context length in tokens (default from config)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="config file path (TOML, default: pyproject.toml)",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="transformers tokenizer path or model id (env TRANSFORMERS_TOKENIZER_PATH)",
    )
    parser.add_argument(
        "--short-row-tokens",
        type=int,
        help="short row token threshold for table pooling",
    )
    parser.add_argument(
        "--table-context-rows",
        type=int,
        help="number of previous rows to provide as context in table translation",
    )
    parser.add_argument(
        "--min-cell-tokens",
        type=int,
        help="minimum tokens per batched table cell group",
    )
    parser.add_argument(
        "--table-batch-mode",
        choices=["table", "row", "cell"],
        help="table batching mode for HTML/Markdown tables (table|row|cell)",
    )
    parser.add_argument(
        "--glossary-max-terms",
        type=int,
        help="max glossary terms to include as context (0 disables)",
    )
    parser.add_argument("--cache-path", help="persistent translation cache path")
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="disable persistent translation cache",
    )
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="translate the entire file in a single request (code fences preserved)",
    )
    parser.add_argument(
        "--pool-max-workers",
        type=int,
        help="maximum number of worker threads (BabelDOC style)",
    )
    parser.add_argument(
        "--qps",
        type=int,
        help="default worker count if --pool-max-workers is not set (BabelDOC style)",
    )
    args = parser.parse_args()

    if not args.api_key:
        LOGGER.error("API key required (env VLLM_API_KEY or --api-key)")
        return 1

    template_path = args.prompt_template
    if template_path is None:
        template_path = DEFAULT_PROMPT_TEMPLATE_PATH

    template_text = load_prompt_template(template_path)
    if template_text is not None:
        global PROMPT_TEMPLATE
        PROMPT_TEMPLATE = template_text
        if args.prompt_template:
            LOGGER.info("Loaded prompt template: %s", template_path)
        else:
            LOGGER.info("Loaded default prompt template: %s", template_path)

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    LOGGER.info("Translating %s -> %s", args.source_lang, args.target_lang)
    started = time.monotonic()
    config = load_config(args.config)
    config_section = get_config_section(config)
    config_paths = config_section.get("paths", {})
    config_cache = config_section.get("cache", {})
    config_tokenizer_path = config_paths.get("tokenizer_path", "") or config_section.get(
        "tokenizer_path", ""
    )
    config_short_row_tokens = read_int(
        config_section, "short_row_tokens", DEFAULT_SHORT_ROW_TOKENS
    )
    config_context_rows = read_int(
        config_section, "table_context_rows", DEFAULT_TABLE_CONTEXT_ROWS
    )
    config_min_cell_tokens = read_int(
        config_section, "min_cell_tokens", DEFAULT_MIN_CELL_TOKENS
    )
    config_table_batch_mode = normalize_table_batch_mode(
        config_section.get("table_batch_mode", DEFAULT_TABLE_BATCH_MODE)
    )
    config_pool_max_workers = read_int(config_section, "pool_max_workers", 0)
    config_qps = read_int(config_section, "qps", 0)
    config_glossary_max_terms = read_int(
        config_section, "glossary_max_terms", DEFAULT_GLOSSARY_MAX_TERMS
    )
    config_max_tokens = read_int(config_section, "max_tokens", DEFAULT_MAX_TOKENS)
    config_max_chars = read_int(config_section, "max_chars", 0)
    config_max_context_tokens = read_int(
        config_section, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS
    )
    config_cache_enabled = read_bool(
        config_cache, "enabled", DEFAULT_CACHE_ENABLED
    )
    config_cache_path = (
        config_cache.get("path", "")
        or config_paths.get("cache_path", "")
        or DEFAULT_CACHE_PATH
    )
    global CUSTOM_SYSTEM_PROMPT
    CUSTOM_SYSTEM_PROMPT = resolve_custom_prompt(
        args.custom_system_prompt or config_section.get("custom_system_prompt", "")
    )
    if CUSTOM_SYSTEM_PROMPT:
        LOGGER.info("Custom system prompt enabled")
    global PROMPT_FINGERPRINT
    PROMPT_FINGERPRINT = get_prompt_fingerprint()

    global TOKENIZER_PATH
    TOKENIZER_PATH = (
        args.tokenizer_path
        or config_tokenizer_path
        or DEFAULT_TOKENIZER_PATH
        or ""
    )
    short_row_tokens = (
        config_short_row_tokens
        if args.short_row_tokens is None
        else max(0, args.short_row_tokens)
    )
    context_rows = (
        config_context_rows
        if args.table_context_rows is None
        else max(0, args.table_context_rows)
    )
    min_cell_tokens = (
        config_min_cell_tokens
        if args.min_cell_tokens is None
        else max(0, args.min_cell_tokens)
    )
    table_batch_mode = normalize_table_batch_mode(
        args.table_batch_mode or config_table_batch_mode
    )
    glossary_max_terms = (
        config_glossary_max_terms
        if args.glossary_max_terms is None
        else max(0, args.glossary_max_terms)
    )
    max_tokens_limit = (
        config_max_tokens if args.max_tokens is None else max(64, args.max_tokens)
    )
    max_context_tokens = (
        config_max_context_tokens
        if args.max_context_tokens is None
        else max(256, args.max_context_tokens)
    )
    max_chars = (
        config_max_chars if args.max_chars is None else max(0, args.max_chars)
    )
    global SOURCE_LANG, TARGET_LANG, MAX_CONTEXT_TOKENS
    SOURCE_LANG = args.source_lang
    TARGET_LANG = args.target_lang
    MAX_CONTEXT_TOKENS = max_context_tokens

    configured_workers = config_pool_max_workers or config_qps or 1
    workers = (
        configured_workers
        if args.pool_max_workers is None
        else max(1, args.pool_max_workers)
    )
    cache_path = args.cache_path or config_cache_path
    cache_enabled = config_cache_enabled and not args.disable_cache
    persistent_cache = PersistentCache(cache_path) if cache_enabled else None
    if persistent_cache is not None:
        LOGGER.info("Persistent cache enabled: %s", cache_path)

    glossary_terms = (
        extract_glossary_terms(content, glossary_max_terms)
        if glossary_max_terms > 0
        else []
    )
    glossary_context = build_glossary_context(glossary_terms)

    if max_chars <= 0:
        auto_max_chars, auto_max_tokens, overhead, ratio = infer_auto_limits(
            content,
            args.source_lang,
            args.target_lang,
            max_context_tokens,
            max_tokens_limit,
            glossary_context,
        )
        max_chars = auto_max_chars
        max_tokens_limit = auto_max_tokens
        LOGGER.info(
            "Auto max chars: %d (context=%d prompt_tokens=%d chars/token=%.2f)",
            max_chars,
            max_context_tokens,
            overhead,
            ratio,
        )

    inferred_max_tokens = infer_max_tokens_from_chars(
        max_chars,
        args.source_lang,
        args.target_lang,
    )
    inferred_max_tokens = min(inferred_max_tokens, max_tokens_limit)
    LOGGER.info("Inferred max tokens per request: %d", inferred_max_tokens)

    whole_file_max_tokens = min(
        max_tokens_limit,
        infer_max_tokens_from_chars(
            max(max_chars, len(content)),
            args.source_lang,
            args.target_lang,
        ),
    )
    if whole_file_max_tokens != inferred_max_tokens:
        LOGGER.info("Whole-file inferred max tokens: %d", whole_file_max_tokens)

    whole_file_success = False
    if args.whole_file:
        if workers != 1:
            LOGGER.info("--whole-file ignores --pool-max-workers")
        try:
            translated = translate_whole_file(
                content,
                args.source_lang,
                args.target_lang,
                args.model,
                args.url,
                args.api_key,
                persistent_cache,
                glossary_context,
                whole_file_max_tokens,
            )
            whole_file_success = True
        except RuntimeError as exc:
            if str(exc) == "Whole-file translation truncated":
                LOGGER.warning(
                    "Whole-file translation truncated; falling back to chunked translation"
                )
            elif is_context_length_error(exc):
                LOGGER.warning(
                    "Whole-file request exceeds model context; falling back to chunked translation"
                )
            else:
                raise
            whole_file_success = False
        if not whole_file_success:
            translated = translate_markdown(
                content,
                args.source_lang,
                args.target_lang,
                args.model,
                args.url,
                args.api_key,
                max_chars,
                inferred_max_tokens,
                workers,
                short_row_tokens,
                context_rows,
                min_cell_tokens,
                table_batch_mode,
                glossary_max_terms,
                persistent_cache,
            )
    else:
        translated = translate_markdown(
            content,
            args.source_lang,
            args.target_lang,
            args.model,
            args.url,
            args.api_key,
            max_chars,
            inferred_max_tokens,
            workers,
            short_row_tokens,
            context_rows,
            min_cell_tokens,
            table_batch_mode,
            glossary_max_terms,
            persistent_cache,
        )
    elapsed = time.monotonic() - started

    output_path = args.output or f"{args.input}.translated.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    LOGGER.info("Saved: %s (%.2fs)", output_path, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
