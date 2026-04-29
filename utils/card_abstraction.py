from utils.hand_equity import HandEquity


class CardAbstraction:

    def __init__(self, num_buckets: int = 10, use_smoothing: bool = True):
        self.equity = HandEquity(simulations=200)
        self.num_buckets = num_buckets
        self.use_smoothing = use_smoothing
        self.cache: dict = {}

    # -----------------------------
    # MAIN BUCKETING
    # -----------------------------
    def bucket_hand(self, hand, board) -> int:

        key = (tuple(hand), tuple(board))

        if key in self.cache:
            return self.cache[key]

        eq = self.equity.estimate(hand, board)
        bucket = self._nonlinear_bucket(eq)

        self.cache[key] = bucket
        return bucket

    # -----------------------------
    # SOLVER-STYLE BUCKETING
    # -----------------------------
    def _nonlinear_bucket(self, eq: float) -> int:
        """
        Poker equity distribution is NOT linear.
        We use sqrt scaling to spread out low-equity hands.
        """
        eq = max(0.0, min(1.0, eq))
        transformed = eq ** 0.5  # sqrt scaling
        bucket = int(transformed * (self.num_buckets - 1))
        return min(bucket, self.num_buckets - 1)  # clamp upper bound

    # -----------------------------
    # CONTEXT-AWARE BUCKETING (street-specific exponents)
    # FIX: added clamp after exponentiation to prevent out-of-range index
    # -----------------------------
    def bucket_hand_context(self, hand, board, street: str = None) -> int:

        eq = self.equity.estimate(hand, board)

        if street == "river":
            eq = eq ** 1.2   # more separation at showdown
        elif street == "preflop":
            eq = eq ** 0.8   # compress preflop variance

        # FIX: must clamp AFTER applying the exponent, not before
        eq = max(0.0, min(1.0, eq))

        bucket = int(eq * (self.num_buckets - 1))
        return min(bucket, self.num_buckets - 1)
