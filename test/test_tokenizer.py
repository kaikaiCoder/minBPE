import pytest
from minBPE import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

special_tokens = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()

llama_text = """
    <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
    Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
    The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
    <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
    """.strip()


@pytest.mark.parametrize("Tokenizer", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
@pytest.mark.parametrize("text", ["hello", "world", "hello world"])
def test_encode_decode_identity(Tokenizer, text):
    tokenizer = Tokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize("Tokenizer", [BasicTokenizer, RegexTokenizer])
def test_train_tokenizer(Tokenizer):
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    tokenizer = Tokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3, verbose=False)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text


@pytest.mark.parametrize("Tokenizer", [RegexTokenizer, RegexTokenizer])
def test_specical_token(Tokenizer):
    tokenizer = Tokenizer()
    tokenizer.register_special_tokens(special_tokens)
    assert tokenizer.decode(tokenizer.encode(llama_text, "all")) == llama_text


def test_gpt4_tokenizer():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokenizer = GPT4Tokenizer()
    tiktoken_encode_ids = enc.encode(specials_string, allowed_special="all")
    tokenizer_encode_ids = tokenizer.encode(specials_string, allowed_special="all")
    assert tiktoken_encode_ids == tokenizer_encode_ids
    assert tokenizer.decode(tokenizer_encode_ids) == enc.decode(tiktoken_encode_ids)


def test_save_load():
    tokenizer = RegexTokenizer()
    tokenizer.train(llama_text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)
    ids = tokenizer.encode(llama_text, "all")
    assert tokenizer.decode(ids) == llama_text
    tokenizer.save("test")
    tokenizer = RegexTokenizer()
    tokenizer.load("test.bpe")
    assert tokenizer.encode(llama_text, "all") == ids
    assert tokenizer.decode(ids) == llama_text
    # remove file
    import os

    os.remove("test.bpe")


if __name__ == "__main__":
    pytest.main()
