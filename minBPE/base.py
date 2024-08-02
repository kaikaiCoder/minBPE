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
        self.pattern = ""
        self.merges = {}
        self.special_tokens = {}
        self.vocab = {}

    def train(self, corpu, vocab_size, verbose=False):
        NotImplementedError()

    def encode(self, text):
        NotImplementedError()

    def decode(self, ids):
        NotImplementedError()

    def save(self, filename):
        """
        保存 merges 和 vocab 到文件
        """
        merge_file = filename + ".bpe"
        with open(merge_file, "w") as f:
            # 版本
            f.write("minBPE v1\n")
            # pattern
            # f.write(f"{self.pattern}\n")
            # special token len
            f.write(f"{len(self.special_tokens)}\n")
            # special tokens
            for token, i in self.special_tokens.items():
                f.write(f"{token} {i}\n")
            for (p0, p1), i in self.merges.items():
                f.write(f"{p0} {p1} {i}\n")

    def load(self, filename):
        # 加载 bpe 文件
        merges = {}
        special_tokens = {}
        with open(filename, "r", encoding="utf-8") as f:
            version = f.readline().strip()  # 版本
            assert version == "minBPE v1"
            # self.pattern = f.readline().strip()
            n_specilai_tokens = int(f.readline().strip())
            for _ in range(n_specilai_tokens):
                token, i = f.readline().strip().split()
                special_tokens[token] = int(i)
            for line in f:
                idx1, idx2, rank = map(int, line.split())
                merges[(idx1, idx2)] = rank
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), i in merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        self.merges = merges
        self.register_special_tokens(special_tokens)
        self.vocab = vocab
