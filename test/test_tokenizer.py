import pytest
from minBPE import BasicTokenizer

wiki_text = """
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


@pytest.mark.parametrize("Tokenizer", [BasicTokenizer])
@pytest.mark.parametrize("text", ["hello", "world", "hello world"])
def test_encode_decode_identity(Tokenizer, text):
    tokenizer = Tokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize("Tokenizer", [BasicTokenizer])
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
    tokenizer.train(text, 256 + 3, verbose=True)
    print(tokenizer.vocab)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text


if __name__ == "__main__":
    pytest.main()
