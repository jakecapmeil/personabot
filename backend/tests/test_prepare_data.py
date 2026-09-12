import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data  # noqa: E402
from fake_tokenizer import FakeTokenizer  # noqa: E402

REPLIES = ["lol", "yeah for sure", "nah i cant tonight", "wait what", "omg stop", "see u there",
           "haha ok", "who told u that", "bet", "im so tired rn"]
PROMPTS = ["you coming tonight", "did you see that", "wanna get food", "guess what happened",
           "are you awake", "what time", "call me later", "i got the job"]


def write_export(path, n_sessions=12, turns_per_session=10):
    start = datetime(2023, 1, 1, 9, 0, 0)
    blocks = []
    for s in range(n_sessions):
        t = start + timedelta(days=s)
        for k in range(turns_per_session):
            sender = "Me" if k % 2 == 0 else "+15551234567"
            text = f"{PROMPTS[(s + k) % len(PROMPTS)]} s{s}" if sender == "Me" else REPLIES[(s * 3 + k) % len(REPLIES)]
            if sender != "Me" and k % 4 == 1:
                text += "\n/Users/x/Library/Messages/Attachments/a/IMG.JPG"
            stamp = (t + timedelta(minutes=k)).strftime("%b %d, %Y  %-I:%M:%S %p")
            blocks.append(f"{stamp}\n{sender}\n{text}\n")
    Path(path).write_text("\n".join(blocks))


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "export.txt"
        write_export(self.raw)
        self.tok = FakeTokenizer()

    def _rows(self, data_dir, name):
        return [json.loads(l) for l in (data_dir / name).read_text().splitlines() if l.strip()]

    def test_minimal_trains_every_reply_once_without_placeholders(self):
        data_dir = self.tmp / "minimal"
        meta = prepare_data.prepare(self.raw, data_dir, name="hudson", tokenizer=self.tok)
        self.assertEqual(meta["display_name"], "hudson")  # the handle is replaced by the persona id
        self.assertEqual(meta["system_prompt"], "You are hudson, texting a friend.")
        self.assertEqual(meta["n_sessions"], 12)

        rows = self._rows(data_dir, "train.jsonl") + self._rows(data_dir, "valid.jsonl")
        trained = [r["messages"][i]["content"] for r in rows for i in r["train_turns"]]
        self.assertEqual(len(trained), 12 * 5)  # every persona reply, exactly once
        for r in rows:
            for i in r["train_turns"]:
                self.assertEqual(r["messages"][i]["role"], "assistant")
                self.assertNotIn("[image]", r["messages"][i]["content"])
                self.assertNotIn("Attachments", r["messages"][i]["content"])

    def test_split_is_by_session(self):
        data_dir = self.tmp / "split"
        prepare_data.prepare(self.raw, data_dir, name="hudson", tokenizer=self.tok)
        state = json.loads((data_dir / "sessions.json").read_text())
        self.assertTrue(state["val_units"])
        self.assertLess(len(state["val_units"]), len(state["units"]))

    def test_rows_fit_budget(self):
        data_dir = self.tmp / "budget"
        prepare_data.prepare(self.raw, data_dir, name="hudson", tokenizer=self.tok)
        meta = prepare_data.build_dataset(data_dir, tokenizer=self.tok, max_seq_length=60)
        rows = self._rows(data_dir, "train.jsonl")
        self.assertGreater(meta["n_train"], 0)
        for r in rows:
            self.assertLessEqual(len(self.tok.apply_chat_template(r["messages"])), 60 - prepare_data.LENGTH_MARGIN)

    def test_retrieval_excludes_own_unit_and_validation(self):
        data_dir = self.tmp / "retrieval"
        meta = prepare_data.prepare(self.raw, data_dir, name="hudson", prompt_style="retrieval",
                                    display_name="Hudson", tokenizer=self.tok)
        self.assertEqual(meta["display_name"], "Hudson")
        val_units = set(json.loads((data_dir / "sessions.json").read_text())["val_units"])

        with_exemplars = 0
        for r in self._rows(data_dir, "train.jsonl"):
            self.assertEqual(len(r["train_turns"]), 1)
            own = {int(w[1:]) for m in r["messages"][1:] for w in m["content"].split() if re.fullmatch(r"s\d+", w)}
            used = {int(u) for u in re.findall(r'- "[^"]* s(\d+)"', r["messages"][0]["content"])}
            with_exemplars += bool(used)
            self.assertFalse(used & own, "an exemplar came from the row's own session")
            self.assertFalse(used & val_units, "an exemplar came from a validation session")
        self.assertGreater(with_exemplars, 0)

    def test_persona_speech(self):
        self.assertEqual(prepare_data.persona_speech("lol\n[image]\nlook https://x.co/a here"), "lol\nlook here")
        self.assertEqual(prepare_data.persona_speech("[link]"), "")


if __name__ == "__main__":
    unittest.main()
