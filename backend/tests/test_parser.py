"""Run from backend/: python3 -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import build_sessions, clean_body, looks_like_handle, parse_file, parse_imessage_export  # noqa: E402

# Built from imessage-exporter's txt templates: threaded replies are indented
# inside the parent body, tapback entries are indented under "Tapbacks:".
EXPORT = """May 17, 2022  5:29:42 PM
Me
you coming tonight?
Tapbacks:
    Loved by +15551234567

May 17, 2022  5:30:01 PM (Read by you after 1 minute)
+15551234567
yeah
    May 17, 2022  5:31:00 PM
    Me
    what time
This message responded to an earlier message.

May 17, 2022  5:32:00 PM
+15551234567
like 9
Edited 30 seconds later: like 9ish
Sent with Confetti

May 17, 2022  5:33:00 PM
+15551234567
https://youtube.com/watch?v=x
Never Gonna Give You Up
The official video for Rick Astley

May 17, 2022  5:33:30 PM Me named the conversation Friday

May 17, 2022  5:34:00 PM
jordan@icloud.com
also this is my email handle

May 17, 2022  5:34:10 PM
+15551234567
/Users/x/Library/Messages/Attachments/ab/IMG_1.HEIC

May 17, 2022  5:35:00 PM
Group Friend
hi all

May 20, 2022  9:00:00 AM
Me
happy birthday!!

May 20, 2022  9:05:00 AM
+15551234567
first paragraph

second paragraph
"""


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.messages = parse_imessage_export(EXPORT.splitlines(True))

    def test_threaded_reply_becomes_its_own_message_in_order(self):
        texts = [(m.sender, m.lines) for m in self.messages[:3]]
        self.assertEqual(texts, [
            ("Me", ["you coming tonight?"]),
            ("+15551234567", ["yeah"]),
            ("Me", ["what time"]),
        ])

    def test_boilerplate_is_removed(self):
        all_text = "\n".join(line for m in self.messages for line in m.lines)
        for junk in ("responded to an earlier", "Edited", "Sent with", "Rick Astley", "Loved by",
                     "named the conversation", "2022"):
            self.assertNotIn(junk, all_text)

    def test_edit_keeps_final_text_and_links_become_placeholders(self):
        lines = [m.lines for m in self.messages if m.sender == "+15551234567"]
        self.assertIn(["like 9ish"], lines)
        self.assertIn(["[link]"], lines)
        self.assertIn(["[image]"], lines)

    def test_multi_paragraph_message_is_kept_whole(self):
        self.assertEqual(self.messages[-1].lines, ["first paragraph", "second paragraph"])

    def test_sessions_split_on_gap_and_merge_with_newlines(self):
        sessions = build_sessions(self.messages, ["+15551234567", "jordan@icloud.com"])
        self.assertEqual(len(sessions), 2)
        first = sessions[0]
        self.assertEqual([t.role for t in first], ["me", "persona", "me", "persona"])
        self.assertEqual(first[-1].text, "like 9ish\n[link]\nalso this is my email handle\n[image]")

    def test_dropping_other_speakers_never_leaves_adjacent_same_role_turns(self):
        sessions = build_sessions(self.messages, ["+15551234567"])
        for session in sessions:
            roles = [t.role for t in session]
            self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])), roles)

    def test_clean_body_variants(self):
        self.assertEqual(clean_body(["Audio Message.caf", "Transcription: hey"]), ["Audio Message.caf", "Transcription: hey"])
        self.assertEqual(clean_body(["/a/Attachments/x/Audio Message.caf", "Transcription: hey"]), ["[audio]"])
        self.assertEqual(clean_body(["Sticker from Me: /Users/x/StickerCache/1.heic"]), ["[sticker]"])
        self.assertEqual(clean_body(["hola", "Translated from:", "hello"]), ["hola"])
        self.assertEqual(clean_body(["Me unsent this message part 3 seconds after sending!"]), [])
        self.assertEqual(clean_body(["Apple Cash transaction: $20"]), ["[payment]"])
        self.assertEqual(clean_body(["sent with love"]), ["sent with love"])

    def test_handles(self):
        self.assertTrue(looks_like_handle("+1 (555) 123-4567"))
        self.assertTrue(looks_like_handle("a.b@icloud.com"))
        self.assertFalse(looks_like_handle("Hudson"))


class PreformattedTest(unittest.TestCase):
    def test_continuation_lines_and_rare_labels(self):
        text = "Me: hey\nJordan: hi\nnote: bring snacks\nMe: ok\nJordan: cool\nMe: bye\nJordan: later\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
        messages, counts = parse_file(f.name)
        self.assertEqual(dict(counts), {"Jordan": 3})
        self.assertEqual(messages[1].lines, ["hi", "note: bring snacks"])

    def test_missing_second_speaker_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Me: one\nMe: two\nMe: three\n")
        with self.assertRaises(ValueError):
            parse_file(f.name)


if __name__ == "__main__":
    unittest.main()
