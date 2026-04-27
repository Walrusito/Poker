# LUT Files - v2 (Bucketed Keys)

The postflop LUT files are now VERSIONED with `_v2` suffix:
  - `flop_equity_lut_v2.json`   (auto-created at runtime)
  - `turn_equity_lut_v2.json`   (auto-created at runtime)
  - `river_equity_lut_v2.json`  (auto-created at runtime)

These use BUCKETED keys (card abstraction) instead of exact card strings.

## Why the change?
Old exact-key space: ~26 million unique flop situations → 0% LUT hit rate
New bucketed key space: ~355 unique flop keys → 93%+ hit rate within one iteration

## Key format
`{N}p:{hand_bucket}|{board_texture}[-{extra_rank_buckets}]`

Example: `2p:h2-1s|b2TUH`
  = 2 players, high+mid suited hand, two-tone tight unpaired board with high card

The preflop LUT is unchanged (hand-class keys always had ~100% hit rate).
