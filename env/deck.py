import random
from typing import List


class Deck:
    """
    CFR-safe deck implementation

    ✔ reproducible option
    ✔ controlled randomness
    ✔ compatible with MCCFR sampling
    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self.cards = list(range(52))
        self.reset()

    # -----------------------------
    # RESET (DETERMINISTIC OPTION)
    # -----------------------------
    def reset(self):
        self.cards = list(range(52))

        if self.seed is not None:
            random.Random(self.seed).shuffle(self.cards)
        else:
            random.shuffle(self.cards)

        self.index = 0

    # -----------------------------
    # DRAW (SAFE POINTER MODEL)
    # -----------------------------
    def draw(self) -> int:
        card = self.cards[self.index]
        self.index += 1
        return card

    # -----------------------------
    # BURN (CONSISTENT WITH INDEX MODEL)
    # -----------------------------
    def burn(self):
        self.index += 1

    # -----------------------------
    # DEAL
    # -----------------------------
    def deal(self, n: int) -> List[int]:
        return [self.draw() for _ in range(n)]

    # -----------------------------
    # HELPERS
    # -----------------------------
    @staticmethod
    def rank(card: int) -> int:
        return card % 13

    @staticmethod
    def suit(card: int) -> int:
        return card // 13

    @staticmethod
    def to_string(card: int) -> str:
        ranks = "23456789TJQKA"
        suits = "shdc"
        return ranks[card % 13] + suits[card // 13]