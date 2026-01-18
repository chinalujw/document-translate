import argparse
import logging
import os
import re
import time

import torch
from transformers import pipeline

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PLACEHOLDER_PREFIX = "__PH__"


def detect_source_lang(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    return "en"


def extract_generated_text(output) -> str:
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, dict):
        generated = output.get("generated_text")
        if isinstance(generated, list) and generated:
            last = generated[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"]
        if isinstance(generated, str):
            return generated
    return str(output)


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
    return text


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    allowed = set("|-: ")
    return all(ch in allowed for ch in stripped)


def translate_segment(
    pipe,
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    cache: dict,
) -> str:
    if source_lang == "auto":
        source_lang = detect_source_lang(text)
    key = (text, source_lang, target_lang, max_new_tokens)
    if key in cache:
        return cache[key]

    started = time.monotonic()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]
    output = pipe(text=messages, max_new_tokens=max_new_tokens, generate_kwargs={"do_sample": False})
    translated = extract_generated_text(output)
    elapsed = time.monotonic() - started
    LOGGER.info("Segment translated: %.2fs, %d chars", elapsed, len(text))
    cache[key] = translated
    return translated


def translate_text_fragment(
    pipe,
    fragment: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    cache: dict,
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

    parts = []
    cursor = 0
    for match in re.finditer(f"{PLACEHOLDER_PREFIX}\\d+__", protected):
        if match.start() > cursor:
            parts.append((True, protected[cursor:match.start()]))
        parts.append((False, match.group(0)))
        cursor = match.end()
    if cursor < len(protected):
        parts.append((True, protected[cursor:]))

    translated_parts = []
    for should_translate, chunk in parts:
        if not should_translate or not chunk.strip():
            translated_parts.append(chunk)
            continue
        translated = translate_segment(
            pipe,
            chunk,
            source_lang,
            target_lang,
            max_new_tokens,
            cache,
        )
        translated_parts.append(translated)

    rebuilt = "".join(translated_parts)
    return restore_placeholders(rebuilt, placeholders)


def translate_line(
    pipe,
    line: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    cache: dict,
) -> str:
    if not line.strip() or is_table_separator(line):
        return line

    header_match = re.match(r"^(\s{0,3}#{1,6}\s+)(.*)$", line)
    if header_match:
        prefix, content = header_match.groups()
        return prefix + translate_text_fragment(
            pipe,
            content,
            source_lang,
            target_lang,
            max_new_tokens,
            cache,
        )

    list_match = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", line)
    if list_match:
        prefix, content = list_match.groups()
        return prefix + translate_text_fragment(
            pipe,
            content,
            source_lang,
            target_lang,
            max_new_tokens,
            cache,
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
                    pipe,
                    cell,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                    cache,
                )
            )
        return "|".join(translated_cells)

    return translate_text_fragment(
        pipe,
        line,
        source_lang,
        target_lang,
        max_new_tokens,
        cache,
    )


def translate_markdown(
    pipe,
    content: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    max_chars: int,
) -> str:
    lines = content.splitlines(keepends=True)
    in_code_block = False
    fence_pattern = re.compile(r"^\s*(```|~~~)")
    cache: dict = {}
    output = []
    buffer = []
    buffer_len = 0
    request_count = 0
    total_chars = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_len, request_count, total_chars
        if not buffer:
            return
        chunk = "".join(buffer)
        request_count += 1
        total_chars += len(chunk)
        output.append(
            translate_text_fragment(
                pipe,
                chunk,
                source_lang,
                target_lang,
                max_new_tokens,
                cache,
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
            pipe,
            line,
            source_lang,
            target_lang,
            max_new_tokens,
            cache,
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
    parser = argparse.ArgumentParser(description="Translate Markdown via TranslateGemma pipeline")
    parser.add_argument("--input", "-i", required=True, help="input markdown file")
    parser.add_argument("--output", "-o", help="output markdown file")
    parser.add_argument("--source-lang", default="auto", help="source language code (auto detects)")
    parser.add_argument("--target-lang", default="en", help="target language code")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="max new tokens per segment")
    parser.add_argument("--max-chars", type=int, default=1200, help="max chars per merged chunk")
    parser.add_argument(
        "--model-path",
        default="/data/.cache/modelscope/hub/models/google/translategemma-4b-it",
        help="local model path",
    )
    parser.add_argument("--device", default="cuda", help="device: cuda or cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    LOGGER.info("Loading pipeline from %s", args.model_path)
    pipe = pipeline(
        "image-text-to-text",
        model=args.model_path,
        device=0 if args.device == "cuda" else -1,
        dtype=torch.bfloat16,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    started = time.monotonic()
    translated = translate_markdown(
        pipe,
        content,
        args.source_lang,
        args.target_lang,
        args.max_new_tokens,
        args.max_chars,
    )
    elapsed = time.monotonic() - started

    output_path = args.output or f"{args.input}.translated.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    LOGGER.info("Saved: %s (%.2fs)", output_path, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
