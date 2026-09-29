"""生成大作业一的语料：一个确定性的"小故事"生成器（固定随机种子），输出 data/train.txt 和 data/val.txt。

语料由有限的词表和句式组成，但有长程的一致性（同一个故事里人物、地点、物品保持不变），
模型既要学会词和句式，也要学会"记住前文提到的名字"。
用法：python data.py
"""

import random
from pathlib import Path

NAMES = ["Lily", "Tom", "Mia", "Ben", "Anna", "Max", "Zoe", "Leo", "Ella", "Sam", "Nora", "Jack"]
ANIMALS = ["cat", "dog", "bird", "fox", "rabbit", "turtle", "owl", "frog"]
PLACES = ["park", "garden", "forest", "beach", "school", "library", "farm", "river"]
THINGS = ["ball", "kite", "book", "hat", "cake", "box", "shell", "map", "lamp", "drum"]
COLORS = ["red", "blue", "green", "yellow", "small", "big", "old", "shiny"]
FEELINGS = ["happy", "sad", "tired", "excited", "scared", "proud", "curious"]


def story(rng: random.Random) -> str:
    a, b = rng.sample(NAMES, 2)
    pet, place, thing, color = rng.choice(ANIMALS), rng.choice(PLACES), rng.choice(THINGS), rng.choice(COLORS)
    feel = rng.choice(FEELINGS)
    lines = [
        f"Once upon a time, {a} had a {color} {thing}.",
        f"One day, {a} went to the {place} with a {pet}.",
        f"At the {place}, {a} met {b}.",
    ]
    for _ in range(rng.randint(2, 5)):
        lines.append(rng.choice([
            f"{b} asked, \"Can I play with your {thing}?\"",
            f"{a} said, \"Yes, you can play with my {color} {thing}.\"",
            f"The {pet} ran around the {place} and {a} laughed.",
            f"{a} and {b} played with the {thing} all day.",
            f"{b} felt {feel} and smiled at {a}.",
            f"Then the {pet} found a {rng.choice(COLORS)} {rng.choice(THINGS)} near the {place}.",
            f"\"Look!\" said {b}. \"The {pet} is so {rng.choice(FEELINGS)}!\"",
        ]))
    lines.append(f"At the end of the day, {a} and {b} went home. They were {feel}.")
    return " ".join(lines)


def main():
    rng = random.Random(2026)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    for name, n in (("train", 12000), ("val", 400)):
        (out / f"{name}.txt").write_text("\n".join(story(rng) for _ in range(n)) + "\n", encoding="utf-8")
        print(f"{name}.txt：{n} 个故事，{(out / f'{name}.txt').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
