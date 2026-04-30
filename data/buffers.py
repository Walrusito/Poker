import random
from typing import Any, List


class ReservoirBuffer:
    """
    Reservoir Sampling Buffer — critical for Deep CFR.

    Maintains a uniform distribution over all experiences seen in time.

    Optimisations (Steps 4 & 5 from the performance plan):
      - snapshot() returns the data list directly — DataLoader(shuffle=True)
        handles ordering, avoiding redundant O(N) copy+shuffle.
      - sample() kept for backward compatibility but marked as legacy.
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self.data: List[Any] = []
        self.n_seen: int = 0

    def add(self, item: Any) -> None:
        self.n_seen += 1

        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            i = random.randint(0, self.n_seen - 1)
            if i < self.max_size:
                self.data[i] = item

    def snapshot(self) -> List[Any]:
        """Return current buffer contents without copy — DataLoader handles shuffling."""
        return self.data

    def sample(self) -> List[Any]:
        """Legacy: return a shuffled copy. Prefer snapshot() with DataLoader(shuffle=True)."""
        shuffled = self.data.copy()
        random.shuffle(shuffled)
        return shuffled

    def sample_batch(self, batch_size: int) -> List[Any]:
        """Return a random mini-batch (with replacement if buffer is smaller)."""
        k = min(batch_size, len(self.data))
        return random.choices(self.data, k=k)

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()
        self.n_seen = 0
