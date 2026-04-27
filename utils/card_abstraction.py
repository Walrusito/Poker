import threading
from collections import OrderedDict

from utils.equity_lut import EquityLUT


class CardAbstraction:
    def __init__(
        self,
        num_buckets: int = 10,
        use_smoothing: bool = True,
        mc_simulations: int = 200,
        lut_simulations: int = 1500,
        lut_dir: str = "data/lut",
        seed=None,
        use_torch_backend: bool = False,
        torch_device: str = "cuda",
        max_cache_size: int = 100_000,
    ):
        self.equity_provider = EquityLUT(
            lut_dir=lut_dir,
            mc_simulations=mc_simulations,
            lut_simulations=lut_simulations,
            seed=seed,
            use_torch_backend=use_torch_backend,
            torch_device=torch_device,
        )
        self.num_buckets = num_buckets
        self.use_smoothing = use_smoothing
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()

    def estimate_equity(self, hand, board, num_players: int = 2) -> float:
        key = (tuple(sorted(hand)), tuple(sorted(board)), num_players)
        with self._lock:
            if key in self.cache:
                self._cache_hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            self._cache_misses += 1
            value = self.equity_provider.estimate(hand, board, num_players=num_players)
            self.cache[key] = value
            if len(self.cache) > self.max_cache_size:
                self.cache.popitem(last=False)
            return value

    def bucket_hand(self, hand, board, num_players: int = 2) -> int:
        eq = self.estimate_equity(hand, board, num_players=num_players)
        return self._nonlinear_bucket(eq)

    def reset_equity_stats(self):
        self.equity_provider.reset_stats()

    def get_equity_stats(self):
        stats = self.equity_provider.get_stats()
        total = self._cache_hits + self._cache_misses
        stats["equity_cache_hit_rate"] = self._cache_hits / max(1, total)
        stats["equity_cache_size"] = len(self.cache)
        return stats

    def _nonlinear_bucket(self, eq: float) -> int:
        eq = max(0.0, min(1.0, eq))
        transformed = eq ** 0.5 if self.use_smoothing else eq
        bucket = int(transformed * (self.num_buckets - 1))
        return min(bucket, self.num_buckets - 1)
