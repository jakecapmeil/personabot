"""A tiny chat-template tokenizer with the properties the pipeline relies on:
one token per word/newline, special role markers, and prefix-stable
templates (rendering the first k messages is a prefix of rendering k+1)."""


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, line in enumerate(text.split("\n")):
            if i:
                ids.append(self._id("\n"))
            ids.extend(self._id(w) for w in line.split())
        return ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        ids = [self._id("<bos>")]
        for m in messages:
            ids += [self._id("<start>"), self._id(m["role"])] + self.encode(m["content"]) + [self._id("<end>")]
        if add_generation_prompt:
            ids += [self._id("<start>"), self._id("assistant")]
        return ids
