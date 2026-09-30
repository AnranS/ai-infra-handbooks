def word_break(s, words):
    remain = s
    for w in sorted(words, key=len, reverse=True):   # 贪心地从长到短切：会答错
        while w in remain:
            remain = remain.replace(w, "", 1)
    return remain == ""


def min_cuts(s, words):
    cuts = 0
    remain = s
    for w in sorted(words, key=len, reverse=True):
        while w in remain:
            remain = remain.replace(w, "", 1)
            cuts += 1
    return cuts if remain == "" else -1
