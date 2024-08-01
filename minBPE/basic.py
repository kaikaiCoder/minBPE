from .base import Tokenizer, merge_vocab, get_stats

BASE_VOCAB_SIZE = 256


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}

    def train(self, corpus, vocab_size, verbose=False):
        assert vocab_size > BASE_VOCAB_SIZE
        n_merge = vocab_size - BASE_VOCAB_SIZE
        ids = list(corpus.encode("utf-8"))
        for i in range(n_merge):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            new_id = BASE_VOCAB_SIZE + i
            ids = merge_vocab(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"Merge {i+1}/{n_merge} {pair} -> {new_id}")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_vocab(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("utf-8", errors="replace")
