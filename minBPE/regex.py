import regex as re
from .base import Tokenizer, get_stats, merge_vocab

BASE_VOCAB_SIZE = 256

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    Tokenizer using regex to split text into chunks
    """

    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        self.pattern = re.compile(pattern)
        self.vocab = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}
        self.sepcial_tokens = {}
        self.inverse_special_tokens = {}

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, corpus, vocab_size, verbose=False):
        assert vocab_size > BASE_VOCAB_SIZE
        n_merge = vocab_size - BASE_VOCAB_SIZE
        # 使用正则表达式将语料库切成小块
        chunks = re.findall(self.pattern, corpus)
        chunks_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        for i in range(n_merge):
            # 统计每个小块中的字符
            stats = {}
            for chunk_ids in chunks_ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            new_id = BASE_VOCAB_SIZE + i
            # 合并字符
            chunks_ids = [
                merge_vocab(chunk_ids, pair, new_id) for chunk_ids in chunks_ids
            ]
            # 更新词表
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"Merge {i+1}/{n_merge} {pair} -> {new_id}")

    def _encode_chunk(self, ids):
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_vocab(ids, pair, self.merges[pair])
        return ids

    # 无 special tokens
    def encode_ordinary(self, text):
        ids = []
        chunks_ids = [
            list(chunk.encode("utf-8")) for chunk in re.findall(self.pattern, text)
        ]
        for chunk_ids in chunks_ids:
            ids.extend(self._encode_chunk(chunk_ids))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, dict):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"Invalid value for allowed_special: {allowed_special}")
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # 分开处理特殊字符
        ids = []
        for chunk in special_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        res = []
        for id in ids:
            if id in self.vocab:
                res.append(self.vocab[id])
            elif id in self.inverse_special_tokens:
                res.append(self.inverse_special_tokens[id].encode("utf-8"))
            else:
                raise ValueError(f"Invalid id: {id}")
        return b"".join(res).decode("utf-8", errors="replace")
