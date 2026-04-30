from functools import lru_cache
from itertools import combinations
from typing import List, Tuple

try:
    import eval7 as _ev7

    _RANKS = "23456789TJQKA"
    _SUITS = "shdc"
    _CARD_MAP = tuple(
        _ev7.Card(_RANKS[i % 13] + _SUITS[i // 13]) for i in range(52)
    )
    _HAS_EVAL7 = True
except ImportError:
    _HAS_EVAL7 = False


# -----------------------------
# CARD HELPERS
# -----------------------------
def rank(card: int) -> int:
    return card % 13


def suit(card: int) -> int:
    return card // 13


# -----------------------------
# MAIN EVALUATOR (5 CARDS) — pure-Python fallback
# -----------------------------
def evaluate_5(cards: List[int]) -> Tuple:

    ranks = sorted([rank(c) for c in cards], reverse=True)
    suits = [suit(c) for c in cards]

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1

    counts_sorted = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))

    is_flush = len(set(suits)) == 1
    is_straight, top_straight = _is_straight(ranks)

    # STRAIGHT FLUSH
    if is_flush and is_straight:
        return (8, top_straight)

    # FOUR OF A KIND
    if counts_sorted[0][1] == 4:
        quad = counts_sorted[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (7, quad, kicker)

    # FULL HOUSE
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        return (6, counts_sorted[0][0], counts_sorted[1][0])

    # FLUSH
    if is_flush:
        return (5, tuple(ranks))

    # STRAIGHT
    if is_straight:
        return (4, top_straight)

    # THREE OF A KIND
    if counts_sorted[0][1] == 3:
        trips = counts_sorted[0][0]
        kickers = tuple(sorted([r for r in ranks if r != trips], reverse=True))
        return (3, trips, kickers)

    # TWO PAIR
    pairs = [r for r, c in counts_sorted if c == 2]

    if len(pairs) >= 2:
        high_pair, low_pair = pairs[0], pairs[1]
        kicker = max(r for r in ranks if r not in (high_pair, low_pair))
        return (2, high_pair, low_pair, kicker)

    # ONE PAIR
    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = tuple(sorted([r for r in ranks if r != pair], reverse=True))
        return (1, pair, kickers)

    # HIGH CARD
    return (0, tuple(ranks))


# -----------------------------
# BEST OF 7 CARDS
# -----------------------------
if _HAS_EVAL7:
    def evaluate_7(cards: List[int]) -> int:
        """Fast 7-card evaluation via eval7 C extension."""
        return _ev7.evaluate([_CARD_MAP[c] for c in cards])

    @lru_cache(maxsize=500_000)
    def evaluate_7_cached(cards: tuple) -> int:
        """Cached version — pass cards as a sorted tuple."""
        return _ev7.evaluate([_CARD_MAP[c] for c in cards])
else:
    def evaluate_7(cards: List[int]) -> Tuple:
        """Pure-Python fallback: exact combinatorial evaluation (21 combos)."""
        return max(evaluate_5(list(c)) for c in combinations(cards, 5))

    @lru_cache(maxsize=500_000)
    def evaluate_7_cached(cards: tuple) -> Tuple:
        """Cached version — pass cards as a sorted tuple."""
        return max(evaluate_5(list(c)) for c in combinations(cards, 5))


# -----------------------------
# STRAIGHT DETECTION
# -----------------------------
def _is_straight(ranks: List[int]) -> Tuple[bool, int]:

    r = sorted(set(ranks), reverse=True)

    # wheel (A-5)
    wheel = {12, 0, 1, 2, 3}
    if wheel.issubset(set(r)):
        return True, 3

    for i in range(len(r) - 4):
        window = r[i:i + 5]

        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return True, window[0]

    return False, -1
