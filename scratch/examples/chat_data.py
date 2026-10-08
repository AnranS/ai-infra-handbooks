"""从同一份语料机械地造出指令数据：续写、下一句、谁说的。真实的 SFT 数据是人写的或大模型合成的，格式是一样的"""
import collections
import json
import random
import re
from pathlib import Path

random.seed(0)
chapters = re.split(r"(?=^第.{1,4}回：)", Path("sanguo.txt").read_text(encoding="utf-8"), flags=re.M)[1:]
SAY = re.compile(r"([一-龥]{1,3})曰：「([^」]{6,28})」")       # 「某某曰：「……」」
STOP = {"问", "公", "众", "答", "又", "大", "曰", "言", "报", "或", "左右", "众人", "老人", "童子", "军士",
        "一人", "二人", "后人", "来人", "细作", "门吏", "近臣", "侍臣"}                # 「问曰」「众人曰」不是人名


def clean(s):
    """整句、引号配对的句子才用（半句话当答案会教坏模型）"""
    s = s.strip()
    return 10 <= len(s) <= 30 and s.count("「") == s.count("」") and s.count("『") == s.count("』")


def samples(chs, names):
    """每一回出三类样本；前两类的答案都是"下一句"，只是问法不同——模型必须看指令才知道要做什么"""
    out = []
    for ch in chs:
        body = ch.split("\n", 1)[1] if "\n" in ch else ch
        sents = [s.strip() for s in re.split(r"(?<=[。！？」])", body) if clean(s)]
        out += [[("接下来写：" + sents[i], sents[i + 1])] for i in range(0, len(sents) - 1, 7)]
        out += [[(f"「{sents[i]}」的下一句是什么？", sents[i + 1])] for i in range(1, len(sents) - 1, 9)]
        out += [[(f"这句话是谁说的：「{what}」", who)] for who, what in SAY.findall(body) if who in names]
    return out


count = collections.Counter(w for c in chapters for w, _ in SAY.findall(c) if w not in STOP)
NAMES = {w for w, _ in count.most_common(12)}
train = samples(chapters[:-12], NAMES)
val = samples(chapters[-12:], NAMES)                                    # 后 12 回：从没见过的情节
random.shuffle(train)
who = [t for t in train if "这句话是谁说" in t[0][0]][:600]              # 「谁说的」另留一份：后 12 回里的人物几乎全换了
train = [t for t in train if t not in who]
for pool, n in ((train, 400), (val, 20)):                               # 一部分首尾相接拼成两轮对话
    for i in random.sample(range(len(pool) - 1), n):
        pool[i] = pool[i] + pool[i + 1]

SYSTEM = ["你是一个读过《三国演义》的小助手。", "你是三国问答助手，请简短作答。", "你熟悉《三国演义》，请帮用户补全原文。"]


def dump(path, data):
    """两成样本带 system：模型要学会"有没有系统提示都能答"（真实的 SFT 数据集也这么掺）"""
    with open(path, "w", encoding="utf-8") as f:
        for turns in data:
            conv = [m for u, a in turns for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
            if random.random() < 0.2:
                conv = [{"role": "system", "content": random.choice(SYSTEM)}] + conv
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")


for path, data in (("sft_train.jsonl", train), ("sft_val.jsonl", val), ("sft_who.jsonl", who)):
    dump(path, data)
kinds = {"续写": "接下来写", "下一句": "的下一句", "谁说的": "这句话是谁说"}
print(f"说话最多的 12 个人：{'、'.join(w for w, _ in count.most_common(12))}")
print(f"训练 {len(train)} 条（两轮的 {sum(len(t) > 1 for t in train)} 条）、验证 {len(val)} 条、"
      f"「谁说的」留出 {len(who)} 条（前 400 条给下一章的 LoRA，后 200 条当测试）")
print("训练集三类各：", "、".join(f"{k} {sum(m in t[0][0] for t in train)}" for k, m in kinds.items()))
for line in open("sft_train.jsonl", encoding="utf-8").readlines()[:2]:
    conv = json.loads(line)["conversations"]
    print("样本：" + " → ".join(f"[{m['role'][0]}] {m['content']}" for m in conv))
