"""Find training problems whose statement overlaps an evaluation statement.

Statements are lowercased and split into alphanumeric words. For each
evaluation statement, its word n-grams are indexed; a training statement's
overlap with it is the fraction of those n-grams the training statement also
contains (containment, so a long training statement cannot dilute a short
evaluation one). A training problem is contaminated if its largest overlap
with any evaluation statement reaches the threshold.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.codeprobe.livecodebench_data import RELEASE_FILES
from scripts.codetasks.problems import read_problems

# Calibrated on 43,688 candidates (20k KodCode, all DeepCoder stdin): the one true duplicate found, a rewording
# of MBPP/119, scored 0.53 at n=8; the 0.3-0.5 band held only shared generic phrasing ("write a python function
# to find the length of the longest"). 0.3 keeps a margin below the duplicate and removed 12 candidates.
DEDUP_N = 8
DEDUP_THRESHOLD = 0.3


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class EvalIndex:
    def __init__(self, statements: dict[str, str], n: int):
        self.n = n
        self.sizes: dict[str, int] = {}
        self.postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for key, text in statements.items():
            grams = ngrams(text, n)
            if grams:
                self.sizes[key] = len(grams)
                for gram in grams:
                    self.postings[gram].append(key)

    def overlap(self, text: str) -> tuple[float, str | None]:
        """The largest containment of an evaluation statement in `text`, and that statement's key."""
        hits = Counter(key for gram in ngrams(text, self.n) for key in self.postings.get(gram, ()))
        if not hits:
            return 0.0, None
        key, count = max(hits.items(), key=lambda item: item[1] / self.sizes[item[0]])
        return count / self.sizes[key], key


def eval_statements(problem_sets: list[Path], confirmation: Path, raw: Path) -> dict[str, str]:
    """Statements keyed "<set>:<id>": our problem sets, plus the LiveCodeBench confirmation problems read from the
    raw release (statements only; their tests are not decoded)."""
    statements = {f"{directory.name}:{problem.id}": problem.statement
                  for directory in problem_sets for problem in read_problems(directory)}
    frozen = json.loads(confirmation.read_text())
    wanted = set(frozen["question_ids"])
    for name in RELEASE_FILES[frozen["release"]]:
        for line in (raw / name).open():
            record = json.loads(line)
            if record["question_id"] in wanted:
                statements[f"confirmation:{record['question_id']}"] = record["question_content"]
    missing = wanted - {key.split(":", 1)[1] for key in statements if key.startswith("confirmation:")}
    if missing:
        raise ValueError(f"confirmation problems missing from {raw}: {sorted(missing)[:5]}")
    return statements
