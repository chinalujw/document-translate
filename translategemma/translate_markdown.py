import argparse
import json
import logging
import os
import re
import time
import urllib.request

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PLACEHOLDER_PREFIX = "__PH__"
PROMPT_TEMPLATE: str | None = None
DEFAULT_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_templates",
    "translate_gemma.txt",
)


def post_request(url: str, payload: dict, api_key: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    return json.loads(body)


def translate_segment(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
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
    max_tokens: int,
) -> tuple[str, str | None]:
    if source_lang == "auto":
        LOGGER.error("source_lang must be specified; auto detection is disabled")
        raise ValueError("source_lang must be specified")
    key = (text, source_lang, target_lang, model, url)
    if key in cache:
        return cache[key], None

    started = time.monotonic()
    completion_url = url.replace("/v1/chat/completions", "/v1/completions")
    prompt = build_gemma_prompt(text, source_lang, target_lang)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<end_of_turn>"],
    }
    response = post_request(completion_url, payload, api_key)
    elapsed = time.monotonic() - started
    LOGGER.info("Segment translated: %.2fs, %d chars", elapsed, len(text))
    translated = response["choices"][0]["text"]
    finish_reason = response["choices"][0].get("finish_reason")
    cache[key] = translated
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


def build_gemma_prompt(text: str, source_lang: str, target_lang: str) -> str:
    source_label = lang_label(source_lang)
    target_label = lang_label(target_lang)
    if PROMPT_TEMPLATE is not None:
        values = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_label": source_label,
            "target_label": target_label,
            "text": text,
        }
        try:
            rendered = PROMPT_TEMPLATE.format_map(values)
        except KeyError as exc:
            LOGGER.error("Prompt template missing placeholder: %s", exc)
            raise
        if "{text}" not in PROMPT_TEMPLATE:
            LOGGER.warning("Prompt template missing {text}; appending text at the end")
            rendered = f"{rendered}\n{text}"
        return rendered
    return (
        "<bos>\n"
        "<start_of_turn>user\n"
        f"Translate {source_label} ({source_lang}) to {target_label} ({target_lang}). "
        f"Output only the {target_label} translation. "
        "Keep tokens like __PH__<number>__ unchanged.\n\n"
        f"{text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_prompt_template(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def max_group_tokens(max_tokens: int) -> int:
    return max(64, int(max_tokens * 0.6))


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    allowed = set("|-: ")
    return all(ch in allowed for ch in stripped)

def is_html_line(line: str) -> bool:
    return bool(re.search(r"<(table|thead|tbody|tfoot|tr|td|th|p|div|span|br|img)\b", line, re.IGNORECASE))


def translate_html_fragment(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    max_tokens: int,
) -> str:
    row_pattern = re.compile(r"(<tr\b[^>]*>.*?</tr>)", re.IGNORECASE | re.DOTALL)
    if row_pattern.search(fragment):
        def translate_table_row(row_html: str) -> str:
            if estimate_tokens(row_html) > max_group_tokens(max_tokens):
                LOGGER.info("Row-level translation skipped: estimated tokens too large")
                return translate_html_cells(row_html)

            row_translated, truncated, unresolved = translate_text_fragment_with_meta(
                row_html,
                source_lang,
                target_lang,
                model,
                url,
                api_key,
                cache,
                max_tokens,
            )
            if truncated or unresolved:
                LOGGER.info(
                    "Row-level translation fallback: truncated=%s unresolved=%s",
                    truncated,
                    unresolved,
                )
                return translate_html_cells(row_html)
            if (
                row_translated.count("<td") != row_html.count("<td")
                or row_translated.count("<th") != row_html.count("<th")
            ):
                LOGGER.info("Row-level translation fallback: cell count mismatch")
                return translate_html_cells(row_html)
            return row_translated

        def translate_html_cells(row_html: str) -> str:
            cell_pattern = re.compile(
                r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)",
                re.IGNORECASE | re.DOTALL,
            )
            matches = list(cell_pattern.finditer(row_html))
            if not matches:
                return row_html

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

            def translate_group(cells: list[str]) -> list[str] | None:
                protected_cells = []
                placeholders: dict[str, str] = {}
                counter = 0
                for cell in cells:
                    protected, local_placeholders = protect_patterns(cell, patterns)
                    for ph, original in local_placeholders.items():
                        new_ph = f"{PLACEHOLDER_PREFIX}{counter}__"
                        counter += 1
                        protected = protected.replace(ph, new_ph)
                        placeholders[new_ph] = original
                    protected_cells.append(protected)

                sep_token = f"{PLACEHOLDER_PREFIX}{counter}__"
                sep_value = "\n<<<CELL_SEP>>>\n"
                placeholders[sep_token] = sep_value
                combined = sep_token.join(protected_cells)

                translated, finish_reason = translate_segment_with_meta(
                    combined,
                    source_lang,
                    target_lang,
                    model,
                    url,
                    api_key,
                    cache,
                    max_tokens,
                )
                if finish_reason == "length":
                    return None

                restored = restore_placeholders(translated, placeholders)
                if re.search(r"__PH__\d+__?", restored):
                    return None

                parts = restored.split(sep_value)
                if len(parts) != len(cells):
                    return None
                return parts

            cell_texts = [match.group(2) for match in matches]
            groups: list[list[int]] = []
            current_group: list[int] = []
            current_tokens = 0
            max_tokens_group = max_group_tokens(max_tokens)

            for idx, cell in enumerate(cell_texts):
                cell_tokens = estimate_tokens(cell)
                if current_group and current_tokens + cell_tokens > max_tokens_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                current_group.append(idx)
                current_tokens += cell_tokens
            if current_group:
                groups.append(current_group)

            translated_cells: list[str] = [""] * len(cell_texts)
            for group in groups:
                group_cells = [cell_texts[i] for i in group]
                group_result = translate_group(group_cells)
                if group_result is None:
                    LOGGER.info("Row-cell batch translation fallback: truncated or invalid; retrying sub-block")
                    for idx, cell in zip(group, group_cells):
                        translated_cells[idx] = translate_text_fragment(
                            cell,
                            source_lang,
                            target_lang,
                            model,
                            url,
                            api_key,
                            cache,
                            max_tokens,
                        )
                    continue
                for idx, translated_inner in zip(group, group_result):
                    translated_cells[idx] = translated_inner

            result = []
            last_end = 0
            for match, translated_inner in zip(matches, translated_cells):
                result.append(row_html[last_end:match.start()])
                result.append(match.group(1))
                result.append(translated_inner)
                result.append(match.group(3))
                last_end = match.end()
            result.append(row_html[last_end:])
            return "".join(result)

        def row_repl(match: re.Match) -> str:
            return translate_table_row(match.group(1))

        return row_pattern.sub(row_repl, fragment)

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
    if placeholders and not re.search(r"[A-Za-zÀ-ÿ\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]", protected):
        return fragment

    translated = translate_segment(
        protected,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        max_tokens,
    )
    restored = restore_placeholders(translated, placeholders)
    if placeholders and re.search(r"__PH__\d+__?", restored):
        LOGGER.warning("Unresolved placeholders detected; keeping original fragment")
        return fragment
    return restored


def translate_text_fragment_with_meta(
    fragment: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
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
    if placeholders and not re.search(
        r"[A-Za-zÀ-ÿ\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]",
        protected,
    ):
        return fragment, False, False

    translated, finish_reason = translate_segment_with_meta(
        protected,
        source_lang,
        target_lang,
        model,
        url,
        api_key,
        cache,
        max_tokens,
    )
    restored = restore_placeholders(translated, placeholders)
    if placeholders and re.search(r"__PH__\d+__?", restored):
        LOGGER.warning("Unresolved placeholders detected; keeping original fragment")
        return fragment, finish_reason == "length", True
    return restored, finish_reason == "length", False


def translate_line(
    line: str,
    source_lang: str,
    target_lang: str,
    model: str,
    url: str,
    api_key: str,
    cache: dict,
    max_tokens: int,
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
            max_tokens,
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
) -> str:
    lines = content.splitlines(keepends=True)
    in_code_block = False
    fence_pattern = re.compile(r"^\s*(```|~~~)")
    cache: dict = {}
    output = []
    request_count = 0
    total_chars = 0
    buffer = []
    buffer_len = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        chunk = "".join(buffer)
        nonlocal request_count, total_chars
        request_count += 1
        total_chars += len(chunk)
        output.append(
            translate_text_fragment(
                chunk,
                source_lang,
                target_lang,
                model,
                url,
                api_key,
                cache,
                max_tokens,
            )
        )
        buffer = []
        buffer_len = 0

    for line in lines:
        if fence_pattern.match(line):
            flush_buffer()
            in_code_block = not in_code_block
            output.append(line)
            continue
        if in_code_block:
            output.append(line)
            continue

        if not line.strip():
            flush_buffer()
            output.append(line)
            continue

        line_out = translate_line(
            line,
            source_lang,
            target_lang,
            model,
            url,
            api_key,
            cache,
            max_tokens,
        )
        if line_out != line:
            flush_buffer()
            request_count += 1
            total_chars += len(line)
            output.append(line_out)
            continue

        buffer.append(line)
        buffer_len += len(line)
        if max_chars > 0 and buffer_len >= max_chars:
            flush_buffer()

    flush_buffer()
    LOGGER.info("Requests sent: %d, total chars: %d", request_count, total_chars)
    return "".join(output)


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
        default=1200,
        help="max characters per merged paragraph chunk (0 disables merge)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max tokens to generate per request",
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
    translated = translate_markdown(
        content,
        args.source_lang,
        args.target_lang,
        args.model,
        args.url,
        args.api_key,
        args.max_chars,
        args.max_tokens,
    )
    elapsed = time.monotonic() - started

    output_path = args.output or f"{args.input}.translated.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    LOGGER.info("Saved: %s (%.2fs)", output_path, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
