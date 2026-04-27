from functools import lru_cache
from itertools import combinations
from typing import Any, List, Tuple

try:
    import eval7  # type: ignore
except Exception:
    eval7 = None


def rank(card: int) -> int:
    return card % 13


def suit(card: int) -> int:
    return card // 13


_EVAL7_CARD_MAP: list = []
_EVAL7_CARD_TUPLE: tuple = ()
if eval7 is not None:
    ranks_str = "23456789TJQKA"
    suits_str = "shdc"
    for card in range(52):
        rank_str = ranks_str[card % 13]
        suit_str = suits_str[card // 13]
        _EVAL7_CARD_MAP.append(eval7.Card(rank_str + suit_str))
    _EVAL7_CARD_TUPLE = tuple(_EVAL7_CARD_MAP)


@lru_cache(maxsize=131072)
def _evaluate_5_cached(cards: Tuple[int, ...]) -> Tuple:
    ranks = sorted((card % 13 for card in cards), reverse=True)
    suits = [card // 13 for card in cards]

    counts = {}
    for rank_value in ranks:
        counts[rank_value] = counts.get(rank_value, 0) + 1

    counts_sorted = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))
    is_flush = len(set(suits)) == 1
    is_straight, top_straight = _is_straight(ranks)

    if is_flush and is_straight:
        return (8, top_straight)
    if counts_sorted[0][1] == 4:
        return (7, counts_sorted[0][0], counts_sorted[1][0])
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] >= 2:
        return (6, counts_sorted[0][0], counts_sorted[1][0])
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, top_straight)
    if counts_sorted[0][1] == 3:
        return (3, counts_sorted[0][0], counts_sorted[1][0], counts_sorted[2][0])
    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        return (2, counts_sorted[0][0], counts_sorted[1][0], counts_sorted[2][0])
    if counts_sorted[0][1] == 2:
        return (1, counts_sorted[0][0], *[rank_value for rank_value in ranks if rank_value != counts_sorted[0][0]])
    return (0, *ranks)


def evaluate_5(cards) -> Tuple:
    """
    Public wrapper that accepts any iterable of five cards while preserving the
    cached implementation under a hashable tuple key.
    """
    return _evaluate_5_cached(tuple(cards))


def evaluate_7(cards: List[int]) -> Any:
    if eval7 is not None:
        eval7_cards = [_EVAL7_CARD_MAP[card] for card in cards]
        return (9, int(eval7.evaluate(eval7_cards)))

    return max(evaluate_5(combo) for combo in combinations(cards, 5))


def evaluate_7_batch(card_lists: List[List[int]]) -> List[Any]:
    if eval7 is None:
        return [evaluate_7(cards) for cards in card_lists]

    cmap = _EVAL7_CARD_TUPLE
    _eval = eval7.evaluate
    return [(9, int(_eval([cmap[c] for c in cards]))) for cards in card_lists]


def _is_straight(ranks: List[int]) -> Tuple[bool, int]:
    unique_ranks = sorted(set(ranks), reverse=True)
    if len(unique_ranks) < 5:
        return False, -1

    if {12, 0, 1, 2, 3}.issubset(unique_ranks):
        return True, 3

    for idx in range(len(unique_ranks) - 4):
        if unique_ranks[idx] - unique_ranks[idx + 4] == 4:
            return True, unique_ranks[idx]

    return False, -1
