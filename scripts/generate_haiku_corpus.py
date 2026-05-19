#!/usr/bin/env python3
"""Generate a deterministic haiku-like training corpus.

This is synthetic data for the language-model prototype. It deliberately uses
repo-local word lists and a simple syllable proxy so the resulting corpus is
reproducible and compatible with examples/haiku_lm.c.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


VOWELS = set("aeiouy")
LINE_TARGETS = (5, 7, 5)
LINE_MAX_CHARS = (28, 34, 28)

RAW_POOLS = {
    "adj": """
        bare bitter black blue bright calm clear cold copper dark deep dim dry
        faint far fresh frost gentle gold green grey hollow late light lone long
        low mild mute pale quiet red rough sharp silver slow small soft spare
        still thin warm white wild young
        amber ancient autumn broken cloudy early empty frozen hidden little
        misty morning narrow open patient purple silent snowy tender twilight
        winter yellow
    """,
    "noun": """
        ash bark bay bird bloom branch breeze brook cloud crow dawn dew dusk fern
        field fire fish flame frost gull hill leaf light marsh mist moon moss night
        path pine pond rain reed root sea shade shell shore sky snow soil song
        spring star stone stream sun tide trail tree wave wind wing wood
        apple autumn blossom cedar cricket dragonfly evening feather forest
        garden heron hollow meadow morning mountain petal river robin shadow
        sparrow summer thunder valley water willow winter
    """,
    "verb": """
        bends blooms breaks calls drifts drops fades falls floats flows glows hides
        hums leans lifts moves rests rings rolls sleeps slides slows stirs waits
        wakes warms weeps
        gathers listens opens rustles settles shivers softens trembles whispers
    """,
    "prep": """
        above across along amid around at below beneath beside beyond by down for
        from in inside near on over past through to under with within
    """,
    "adv": """
        again almost away below gently lightly slowly softly under
    """,
    "det": """
        a each every no one the this
    """,
}

TEMPLATES = {
    5: [
        (("adj", 1), ("noun", 1), ("verb", 1), ("prep", 1), ("noun", 1)),
        (("noun", 1), ("verb", 1), ("prep", 1), ("adj", 1), ("noun", 1)),
        (("adj", 2), ("noun", 1), ("verb", 1), ("noun", 1)),
        (("noun", 2), ("verb", 1), ("prep", 1), ("noun", 1)),
        (("noun", 1), ("verb", 2), ("prep", 1), ("noun", 1)),
        (("det", 1), ("adj", 1), ("noun", 1), ("verb", 1), ("noun", 1)),
        (("adj", 1), ("noun", 2), ("verb", 1), ("noun", 1)),
    ],
    7: [
        (("adj", 1), ("noun", 1), ("verb", 1), ("prep", 1), ("adj", 1), ("noun", 2)),
        (("noun", 2), ("verb", 1), ("prep", 1), ("adj", 1), ("noun", 2)),
        (("det", 1), ("adj", 2), ("noun", 1), ("verb", 1), ("prep", 1), ("noun", 1)),
        (("noun", 1), ("verb", 1), ("prep", 1), ("det", 1), ("adj", 1), ("noun", 2)),
        (("adj", 2), ("noun", 2), ("verb", 1), ("prep", 1), ("noun", 1)),
        (("noun", 2), ("adv", 2), ("verb", 1), ("noun", 2)),
        (("det", 1), ("noun", 2), ("verb", 2), ("prep", 1), ("noun", 1)),
    ],
}


def syllable_proxy(word: str) -> int:
    count = 0
    prev_vowel = False
    for char in word:
        vowel = char in VOWELS
        if vowel and not prev_vowel:
            count += 1
        prev_vowel = vowel
    return count


def load_word_list(path: Path) -> set[str]:
    words = set()
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            word = line.strip().lower()
            if re.fullmatch(r"[a-z]+", word):
                words.add(word)
    return words


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pools(word_list: set[str]) -> tuple[dict[str, dict[int, list[str]]], dict[str, list[str]]]:
    pools: dict[str, dict[int, list[str]]] = {}
    dropped: dict[str, list[str]] = {}
    for name, raw in RAW_POOLS.items():
        pools[name] = {}
        dropped[name] = []
        for word in raw.split():
            if word not in word_list:
                dropped[name].append(word)
                continue
            syllables = syllable_proxy(word)
            if syllables <= 0 or syllables > 4:
                dropped[name].append(word)
                continue
            pools[name].setdefault(syllables, []).append(word)
        for words in pools[name].values():
            words.sort()
    return pools, dropped


def validate_templates(pools: dict[str, dict[int, list[str]]]) -> None:
    missing = []
    for target, templates in TEMPLATES.items():
        for template in templates:
            if sum(syllables for _, syllables in template) != target:
                missing.append(f"bad template sum for {target}: {template}")
            for pool, syllables in template:
                if not pools.get(pool, {}).get(syllables):
                    missing.append(f"missing {pool}/{syllables} for target {target}")
    if missing:
        raise SystemExit("\n".join(missing))


def make_line(
    rng: random.Random,
    target: int,
    pools: dict[str, dict[int, list[str]]],
    line_index: int,
) -> str:
    templates = list(TEMPLATES[target])
    rng.shuffle(templates)
    for template in templates:
        for _ in range(80):
            words = [rng.choice(pools[pool][syllables]) for pool, syllables in template]
            line = " ".join(words)
            if len(line) <= LINE_MAX_CHARS[line_index]:
                return line
    raise RuntimeError(f"failed to generate line for target {target}")


def line_syllables(line: str) -> int:
    return sum(syllable_proxy(word) for word in line.split())


def generate_poems(
    count: int,
    seed: int,
    pools: dict[str, dict[int, list[str]]],
) -> list[str]:
    rng = random.Random(seed)
    poems: list[str] = []
    seen = set()
    attempts = 0
    max_attempts = count * 200
    while len(poems) < count and attempts < max_attempts:
        attempts += 1
        lines = [
            make_line(rng, LINE_TARGETS[0], pools, 0),
            make_line(rng, LINE_TARGETS[1], pools, 1),
            make_line(rng, LINE_TARGETS[2], pools, 2),
        ]
        if tuple(line_syllables(line) for line in lines) != LINE_TARGETS:
            continue
        if any(len(line) > LINE_MAX_CHARS[i] for i, line in enumerate(lines)):
            continue
        poem = "\n".join(lines)
        if poem in seen:
            continue
        seen.add(poem)
        poems.append(poem)
    if len(poems) != count:
        raise RuntimeError(f"generated {len(poems)} poems after {attempts} attempts, wanted {count}")
    return poems


def write_corpus(path: Path, poems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for i, poem in enumerate(poems):
            if i:
                handle.write("\n")
            handle.write(poem)
            handle.write("\n")


def write_vocab(path: Path, poems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    words = sorted({word for poem in poems for word in poem.split()})
    with path.open("w", encoding="ascii") as handle:
        for word in words:
            handle.write(word)
            handle.write("\n")


def write_summary(
    path: Path,
    poems: list[str],
    output_path: Path,
    vocab_output_path: Path,
    word_list_path: Path,
    dropped: dict[str, list[str]],
    pools: dict[str, dict[int, list[str]]],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    words = Counter(word for poem in poems for word in poem.split())
    line_counts = Counter()
    for poem in poems:
        for line in poem.splitlines():
            line_counts[str(line_syllables(line))] += 1
    summary = {
        "schema": "crlp.haiku_corpus.synthetic.v1",
        "output": str(output_path),
        "vocab_output": str(vocab_output_path),
        "poems": len(poems),
        "lines": len(poems) * 3,
        "seed": seed,
        "word_list": str(word_list_path),
        "word_list_sha256": sha256(word_list_path),
        "corpus_sha256": sha256(output_path),
        "unique_words": len(words),
        "line_syllable_proxy_counts": dict(sorted(line_counts.items())),
        "pool_sizes": {
            pool: {str(syllables): len(items) for syllables, items in sorted(groups.items())}
            for pool, groups in sorted(pools.items())
        },
        "dropped_seed_words": {pool: items for pool, items in dropped.items() if items},
        "note": "Synthetic haiku-like corpus; not a human-authored public-domain poetry corpus.",
    }
    with path.open("w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-list", default="data/wordlists/enable2k.txt")
    parser.add_argument("--output", default="data/haiku_synthetic_5000.txt")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20240518)
    parser.add_argument("--summary", default="build/float/haiku_corpus_synthetic/summary.json")
    parser.add_argument("--vocab-output", default="data/wordlists/haiku_synthetic_vocab.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    word_list_path = Path(args.word_list)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    vocab_output_path = Path(args.vocab_output)

    word_list = load_word_list(word_list_path)
    pools, dropped = build_pools(word_list)
    validate_templates(pools)
    poems = generate_poems(args.count, args.seed, pools)
    write_corpus(output_path, poems)
    write_vocab(vocab_output_path, poems)
    write_summary(
        summary_path,
        poems,
        output_path,
        vocab_output_path,
        word_list_path,
        dropped,
        pools,
        args.seed,
    )

    print(
        json.dumps(
            {
                "schema": "crlp.haiku_corpus.synthetic.v1",
                "output": str(output_path),
                "poems": len(poems),
                "summary": str(summary_path),
                "vocab_output": str(vocab_output_path),
                "corpus_sha256": sha256(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
