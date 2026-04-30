from utils.card_abstraction import CardAbstraction


class InformationSetBuilder:

    def __init__(self, mc_simulations=200, lut_simulations=1500,
                 lut_dir="data/lut", seed=None, cache_size=180_000):
        self.card_abs = CardAbstraction()

    def encode(self, state, player=0):

        hand = state["hands"][player]
        board = state["board"]
        street = state["street"]
        history = state["history"]

        hand_bucket = self.card_abs.bucket_hand(hand, board)

        raw = {
            "hand_bucket": hand_bucket,
            "board": board,
            "street": street,
            "history": history
        }

        return self._hash(raw)

    def encode_tuple(self, state, player=0):
        """Fast cache key using tuple hash instead of SHA256."""
        hand = state["hands"][player]
        board = state["board"]
        street = state["street"]
        history = state["history"]

        hand_bucket = self.card_abs.bucket_hand(hand, board)

        return hash((
            hand_bucket,
            tuple(board),
            street,
            tuple((p, a) for p, a in history),
        ))

    def _hash(self, obj):
        return hash((
            obj.get("hand_bucket"),
            tuple(obj.get("board", [])),
            obj.get("street"),
            tuple(tuple(x) if isinstance(x, (list, tuple)) else x
                  for x in obj.get("history", [])),
        ))
