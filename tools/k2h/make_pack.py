"""tools/k2h/make_pack.py - the tokenizer folder serve/server.py reads, taken from a K2 Horizon GGUF (headers only).

    python tools/k2h/make_pack.py MODEL.gguf OUT_DIR

Writes OUT_DIR/vocab.json, merges.txt, token_type.json, tokenizer.json ({"pre": "k2-horizon", "special_ids": ...}) and
chat_template.jinja (the GGUF's tokenizer.chat_template), the same files tools/dsv41/pack_gguf.py writes for DeepSeek.
special_ids: bos and eos from the GGUF (0 <|ifm|begin_of_text|>, 1 <|ifm|endoftext|>) and "eot", the token that ends an
assistant turn in the chat template (<|ifm|im_end|>; llama.cpp counts it as an end of generation too).
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile  # noqa: E402

EOT = "<|ifm|im_end|>"


def write_pack(model: pathlib.Path, out: pathlib.Path) -> dict:
    md = GGUFFile(model).metadata
    if md.get("general.architecture") != "k2-horizon":
        raise SystemExit(f"{model}: architecture {md.get('general.architecture')!r}, not k2-horizon")
    if md.get("tokenizer.ggml.model") != "gpt2" or md.get("tokenizer.ggml.pre") != "k2-horizon":
        raise SystemExit(f"{model}: tokenizer {md.get('tokenizer.ggml.model')!r} / pre {md.get('tokenizer.ggml.pre')!r}, "
                         "expected gpt2 / k2-horizon")
    tokens = list(md["tokenizer.ggml.tokens"])
    if len(set(tokens)) != len(tokens):
        raise SystemExit(f"{model}: the vocabulary has duplicate token strings")
    types = [int(t) for t in (md.get("tokenizer.ggml.token_type") or [1] * len(tokens))]
    merges = list(md.get("tokenizer.ggml.merges") or [])
    if EOT not in tokens:
        raise SystemExit(f"{model}: no {EOT} token")
    special = {"bos": int(md.get("tokenizer.ggml.bos_token_id", 0)), "eos": int(md.get("tokenizer.ggml.eos_token_id", 1)),
               "eot": tokens.index(EOT)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "k2-horizon", "special_ids": special,
                                                     "add_bos": bool(md.get("tokenizer.ggml.add_bos_token", True))}, indent=1))
    template = md.get("tokenizer.chat_template")
    if template:
        (out / "chat_template.jinja").write_text(template, encoding="utf-8")
    return {"tokens": len(tokens), "merges": len(merges), "special": special, "template": len(template or "")}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    info = write_pack(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    print(f"k2h pack: {info['tokens']} tokens, {info['merges']} merges, special {info['special']}, "
          f"chat template {info['template']} characters -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
