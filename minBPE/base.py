def get_stats(ids, stats=None):
    stats = {} if stats is None else stats
    for pair in zip(ids[:-1], ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge_vocab(ids, pair, new_id):
    i = 0
    new_ids = []
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class Tokenizer:

    def __init__(self):
        self.patterns = ""
        self.merges = {}
        self.special_tokens = {}
        self.vocab = {}

    def train(self, corpu, vocab_size, verbose=False):
        NotImplementedError()

    def encode(self, text):
        NotImplementedError()

    def decode(self, ids):
        NotImplementedError()
