from .regex import RegexTokenizer
from .base import get_stats
import tiktoken

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def bpe(mergeable_ranks, text, max_rank):
    """
    mergeable_ranks: {"abc" -> 500}
    text: "abc"
    max_rank: 500
    找到text是由那两个数字token合成的
    将 text 拆分成  (1,2)->500
    """
    text = [bytes([c]) for c in text]
    while True:
        min_rank = None
        index = None
        for i, pair in enumerate(zip(text[:-1], text[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1], max_rank)
            if min_rank is None or rank < min_rank:
                min_rank = rank
                index = i
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert index is not None
        text = text[:index] + [text[index] + text[index + 1]] + text[index + 2 :]
    return text


def recover_merges(mergeable_ranks):
    merges = {}
    for text, rank in mergeable_ranks.items():
        if len(text) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, text, rank))
        assert len(pair) == 2
        p0, p1 = mergeable_ranks[pair[0]], mergeable_ranks[pair[1]]
        merges[(p0, p1)] = rank
    return merges


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), i in self.merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # 实际上 GPT4 中 256 个 base vocab 并未按照 unicode 标准，去除了控制字符
        self.bytes_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_bytes_shuffle = {v: k for k, v in self.bytes_shuffle.items()}

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, ids):
        # 由于 GPT4 的 base vocab 并不是按照 unicode 标准，所以需要先转换
        # unicode --bytes_shuffle-> base vocab
        ids = bytes([self.bytes_shuffle[id] for id in ids])
        return super()._encode_chunk(ids)

    def decode(self, ids):
        res_bytes = []
        for id in ids:
            if id in self.vocab:
                # 先按照 base vocab 解码
                # 再根据 bytes_shuffle 还原成 unicode 字符
                un_shuffle_bytes = bytes(
                    self.inverse_bytes_shuffle[byte] for byte in self.vocab[id]
                )
                res_bytes.append(un_shuffle_bytes)
            elif id in self.inverse_special_tokens:
                res_bytes.append(self.inverse_special_tokens[id].encode("utf-8"))

            else:
                raise ValueError(f"Unknown id {id}")
        return b"".join(res_bytes).decode("utf-8", errors="replace")
