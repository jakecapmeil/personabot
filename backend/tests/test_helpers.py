"""Retrieval, style metrics, generation helpers, and PEFT -> mlx adapter conversion."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import colab_export  # noqa: E402
import eval_style  # noqa: E402
from generation import clean_reply, default_max_tokens, resolve_settings, sticky_window  # noqa: E402
from import_colab import convert_peft_state, mlx_adapter_config  # noqa: E402
from retrieval import ExemplarIndex, usable_exemplar_text  # noqa: E402


class RetrievalTest(unittest.TestCase):
    def test_search_ranks_and_excludes(self):
        index = ExemplarIndex([
            {"prompt": "wanna get food", "reply": "yes starving", "unit": 0},
            {"prompt": "get food later?", "reply": "down", "unit": 1},
            {"prompt": "did you see the game", "reply": "insane", "unit": 2},
        ])
        self.assertEqual([r["reply"] for r in index.search("food tonight?", k=3)], ["yes starving", "down"])
        self.assertEqual([r["reply"] for r in index.search("food", exclude_units={0})], ["down"])
        self.assertEqual(index.search("unrelated words"), [])

    def test_pii_and_spam_are_not_exemplars(self):
        self.assertIsNone(usable_exemplar_text("call me at 555 123 4567"))
        self.assertIsNone(usable_exemplar_text("im at 42 Wallaby Way"))
        self.assertIsNone(usable_exemplar_text("haha" * 30))
        self.assertIsNone(usable_exemplar_text("[image]"))
        self.assertEqual(usable_exemplar_text("lol\n[image]"), "lol")


class StyleTest(unittest.TestCase):
    def test_identical_sets_have_zero_distance(self):
        texts = ["lol", "nah im good", "wait what\nno way"]
        ctx = [[{"role": "user", "content": "x"}]] * 3
        self.assertEqual(eval_style.compare(texts, texts, ctx)["style_distance"], 0.0)

    def test_assistant_voice_scores_worse(self):
        real = ["lol", "nah im good", "omg", "bet", "who told u"]
        assistant = ["That sounds great! Let me know if you need anything.", "Sure thing! I'm here to help.",
                     "Of course! That sounds amazing.", "Absolutely! Great question.", "I understand how you feel."]
        casual = ["haha", "nah", "omg wait", "ok bet", "who said that"]
        ctx = [[{"role": "user", "content": "hey"}]] * 5
        self.assertGreater(eval_style.compare(real, assistant, ctx)["style_distance"],
                           eval_style.compare(real, casual, ctx)["style_distance"])

    def test_copy_rate(self):
        ctx = [[{"role": "system", "content": 'Things they said: - "hi" → "see you at the park later"'}]]
        self.assertEqual(eval_style.copy_rate(["see you at the park later"], ctx), 1.0)


class GenerationHelpersTest(unittest.TestCase):
    def test_sticky_window_moves_in_strides(self):
        history = list(range(20))
        starts = [sticky_window(history[:n], keep=8)[0] for n in range(9, 20)]
        self.assertTrue(all(8 <= n - s < 8 + 4 for n, s in zip(range(9, 20), starts)))
        self.assertLess(len(set(starts)), len(starts))  # prefix stays put for several turns
        self.assertEqual(sticky_window(history[:5], keep=8), history[:5])

    def test_clean_reply(self):
        self.assertEqual(clean_reply("lol\n\n[image]\nwait", "stop"), "lol\nwait")
        self.assertEqual(clean_reply("who told u\nwho told u\nlol", "stop"), "who told u\nlol")
        self.assertEqual(clean_reply("ok sure\nand then we could go to the", "length"), "ok sure")
        self.assertEqual(clean_reply("that was fun. we should do it again sometime soo", "length"),
                         "that was fun. we should do it again sometime")

    def test_settings_migrate_legacy_defaults(self):
        legacy = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 48, "repetition_penalty": 1.15,
                  "repetition_context": 24}
        s = resolve_settings(legacy, reply_tokens_p95=20)
        self.assertEqual(s["repetition_penalty"], 1.05)
        self.assertEqual(s["max_tokens"], default_max_tokens(20))
        custom = resolve_settings({"version": 2, "temperature": 0.7}, overrides={"min_p": 0.1})
        self.assertEqual((custom["temperature"], custom["min_p"]), (0.7, 0.1))


class ConversionTest(unittest.TestCase):
    def test_peft_to_mlx_matches_forward_pass(self):
        rng = np.random.default_rng(0)
        d_in, d_out, r, alpha = 6, 4, 2, 4.0
        A = rng.standard_normal((r, d_in)).astype(np.float32)
        B = rng.standard_normal((d_out, r)).astype(np.float32)
        prefix = "base_model.model.model.layers.3.self_attn.q_proj"
        weights, keys, skipped = convert_peft_state(
            {f"{prefix}.lora_A.weight": A, f"{prefix}.lora_B.weight": B, "base_model.model.lm_head.weight": B},
            rank=r, alpha=alpha,
        )
        self.assertEqual(keys, ["self_attn.q_proj"])
        self.assertEqual(skipped, ["base_model.model.lm_head.weight"])
        x = rng.standard_normal((3, d_in)).astype(np.float32)
        peft = (alpha / r) * (x @ A.T @ B.T)
        a, b = weights["model.layers.3.self_attn.q_proj.lora_a"], weights["model.layers.3.self_attn.q_proj.lora_b"]
        cfg = mlx_adapter_config("repo", r, alpha, keys)
        mlx = cfg["lora_parameters"]["scale"] * ((x @ a) @ b)
        np.testing.assert_allclose(peft, mlx, rtol=1e-5)

    def test_notebook_is_valid_json_with_selected_model(self):
        meta = {"display_name": "Hudson", "system_prompt": "You are Hudson, texting a friend.", "max_seq_length": 1024}
        nb = colab_export.build_notebook(meta, "llama-3b", rank=8)
        source = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", source)
        self.assertIn("RANK = 8", source)
        self.assertNotIn("trl==", source)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and not "".join(cell["source"]).startswith("!"):
                compile("".join(cell["source"]), "cell", "exec")
        json.dumps(nb)


if __name__ == "__main__":
    unittest.main()
