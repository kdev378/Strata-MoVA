"""K2 Horizon's pieces of the server, without the model (docs/K2_HORIZON.md):

  * the chat template's transformers `{% generation %}` tag renders its body, and a template without it is unchanged;
  * a pack's bos_token reaches the template (K2's starts with it) and a caller's own keyword wins;
  * a pack's "eot" special id ends a reply (K2's <|ifm|im_end|>), next to its eos;
  * a pack's "output_text" decodes K2's thinking tags as <think> / </think> while encoding keeps K2's own tokens.

    python -m unittest serve.test_k2_horizon
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import end_token_ids  # noqa: E402


def template(text: str, defaults=None) -> ChatTemplate:
    d = tempfile.mkdtemp()
    p = Path(d) / "chat_template.jinja"
    p.write_text(text, encoding="utf-8")
    return ChatTemplate(p, defaults)


K2_LIKE = ("{{- bos_token }}{%- for m in messages -%}{{- '<s>' + m.role + '\\n' -}}"
           "{%- if m.role == 'assistant' -%}{%- generation -%}{{- m.content -}}{%- endgeneration -%}"
           "{%- else -%}{{- m.content -}}{%- endif -%}{{- '<e>' -}}{%- endfor -%}"
           "{%- if add_generation_prompt -%}{{- '<s>assistant\\n' -}}{%- endif -%}")


class GenerationTag(unittest.TestCase):
    def test_the_body_is_rendered(self):
        t = template(K2_LIKE, {"bos_token": "<B>"})
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "x"}]
        self.assertEqual(t.render(msgs), "<B><s>user\nhi<e><s>assistant\nhello<e><s>user\nx<e><s>assistant\n")

    def test_a_callers_keyword_wins_over_the_default(self):
        t = template(K2_LIKE, {"bos_token": "<B>"})
        self.assertTrue(t.render([{"role": "user", "content": "x"}], bos_token="<C>").startswith("<C>"))

    def test_without_defaults_bos_token_is_empty_as_before(self):
        t = template(K2_LIKE)
        self.assertTrue(t.render([{"role": "user", "content": "x"}]).startswith("<s>user"))

    def test_the_default_template_is_unchanged(self):
        t = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.assertTrue(t.render([{"role": "user", "content": "hi"}]).startswith("<|im_start|>"))
        self.assertEqual(t.defaults, {})


class Pieces:
    """A tokenizer stand-in: every marker is several tokens, as <|im_end|> is in K2's vocabulary."""

    def __init__(self, special_ids):
        self.special_ids = special_ids

    def encode(self, text, parse_special=False):
        return [ord(c) for c in text]


class EndTokens(unittest.TestCase):
    def test_eot_joins_eos(self):
        self.assertEqual(end_token_ids(Pieces({"bos": 0, "eos": 1, "eot": 250019})), {1, 250019})

    def test_without_eot_as_before(self):
        self.assertEqual(end_token_ids(Pieces({"eos": 1})), {1})


class OutputText(unittest.TestCase):
    def test_thinking_tags_decode_as_think_and_encode_as_themselves(self):
        tokens = ["a", "b", "<ifm|think>", "</ifm|think>"]
        tok = ST.Tokenizer(tokens, [], [1, 1, 4, 4], "k2-horizon", {"eos": 1}, {"2": "<think>", "3": "</think>"})
        self.assertEqual(tok.decode([2, 0, 3, 1]), "<think>a</think>b")
        self.assertEqual(tok.encode("<ifm|think>a</ifm|think>"), [2, 0, 3])
        plain = ST.Tokenizer(tokens, [], [1, 1, 4, 4], "k2-horizon", {"eos": 1})
        self.assertEqual(plain.decode([2, 0, 3]), "<ifm|think>a</ifm|think>")


if __name__ == "__main__":
    unittest.main()
