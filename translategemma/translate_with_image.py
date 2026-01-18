import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
import urllib.request

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def encode_image(path: str) -> tuple[str, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(path, "rb") as f:
        payload = f.read()
    encoded = base64.b64encode(payload).decode("ascii")
    return mime_type, f"data:{mime_type};base64,{encoded}"


def build_payload(
    image_data: str,
    text: str,
    model: str,
    mode: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    if mode == "text":
        content_item = {
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }
    elif mode == "image_url":
        content_item = {
            "type": "image_url",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "image_url": {"url": image_data},
        }
    else:
        content_item = {
            "type": "image",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "image": image_data,
        }

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [content_item],
            }
        ],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Send text+image to TranslateGemma")
    parser.add_argument("--image", "-i", required=True, help="local image file path")
    parser.add_argument(
        "--text",
        "-t",
        default="Describe this image in one sentence.",
        help="text prompt describing the image (used when --mode=text)",
    )
    parser.add_argument(
        "--mode",
        choices=["image_url", "image", "text"],
        default="image_url",
        help="send image/text only; default uses image_url for OpenAI schema",
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="source language code",
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        help="target language code",
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
    args = parser.parse_args()

    if not args.api_key:
        LOGGER.error("API key required (env VLLM_API_KEY or --api-key)")
        return 1

    mime_type, encoded = encode_image(args.image)
    LOGGER.info("Encoded %s (%s)", args.image, mime_type)

    payload = build_payload(
        encoded,
        args.text,
        args.model,
        args.mode,
        args.source_lang,
        args.target_lang,
    )
    LOGGER.info("Sending request to %s", args.url)

    try:
        response = post_request(args.url, payload, args.api_key)
    except urllib.error.HTTPError as exc:
        LOGGER.error("request failed: %s", exc.read().decode("utf-8"))
        return 1

    LOGGER.info("Response received")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
