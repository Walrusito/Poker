import hashlib
from utils.card_abstraction import CardAbstraction


class InformationSetBuilder:

    def __init__(self):
        self.card_abs = CardAbstraction()

    def encode(self, state, player=0):

        hand = state["hands"][player]
        board = state["board"]
        street = state["street"]
        history = state["history"]

        # 🔥 ABSTRACTION LAYER
        hand_bucket = self.card_abs.bucket_hand(hand, board)

        raw = {
            "hand_bucket": hand_bucket,
            "board": board,
            "street": street,
            "history": history
        }

        return self._hash(raw)

    def _hash(self, obj):
        return hashlib.sha256(str(obj).encode()).hexdigest()