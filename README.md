# minBPE

minBPE 是一个最小化的 Byte Pair Encoding (BPE) 实现，支持基本的 BPE、正则表达式 BPE 以及 GPT-4 的 Tokenizer。

## 特性

- 实现了基本的 BPE 算法
- 支持正则表达式 BPE
- 实现了 GPT-4 的 Tokenizer

## 安装

```bash
git clone https://github.com/kaikaiCoder/minBPE
cd minBPE
python setup.py install
```

## 使用方法

```python
from minBPE import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

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
    tokenizer = BasicTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3, verbose=False)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text

