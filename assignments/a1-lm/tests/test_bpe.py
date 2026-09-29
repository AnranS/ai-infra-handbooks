from lm.bpe import Tokenizer, train_bpe

TEXT = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"


def test_first_merges():
    # 与 CS336 讲义中的例子一致：s t → e st → o w → l ow → w est → n e
    assert train_bpe(TEXT, 256 + 6) == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]


def test_vocab_limit_and_early_stop():
    assert len(train_bpe(TEXT, 256)) == 0
    merges = train_bpe("ab ab", 1000)
    assert 0 < len(merges) < 1000 - 256, "没有可合并的对时应该提前结束"


def test_roundtrip():
    tok = Tokenizer(train_bpe(TEXT * 3 + " 你好，世界！🙂 tabs\tand\nnewlines", 300))
    for s in ["", "low lower lowest", "你好，世界！🙂", "  spaces\t\ttabs\nnewline ", "newest widest"]:
        assert tok.decode(tok.encode(s)) == s


def test_encode_uses_merges():
    tok = Tokenizer(train_bpe(TEXT, 256 + 6))
    assert tok.vocab_size == 262
    assert tok.encode("low") == [256 + 3], "l + ow 合并后是一个 token"
    assert tok.encode(" newest") == [32, 256 + 5, 256 + 4], "片段开头的空格不参与合并（训练语料里没有出现过空格开头的对）"


def test_pieces_do_not_cross():
    merges = train_bpe("a b a b a b", 300)
    assert (b"a", b" ") not in merges and (b" ", b"b") in merges, "合并只在片段内部进行"
