# BUILD-NOTES — gta6-trophy-rarity-forecast-01

Entry: a program that forecasts, for every GTA VI trophy on PS5, the
percent of Exophase-tracked players who will have earned it at
2026-11-26T00:00:00Z, as five quantiles (0.05/0.25/0.5/0.75/0.95),
scored by mean pinball loss on the logit scale. Prize needs a loss
strictly below the best house entry (persistence baseline, curve
baseline, superbot model forecast). $250 USDC escrowed, 20 places.

## Files

- `main.py` — the whole entry, Python 3.13 stdlib only, deterministic,
  no network, no model calls. Written from scratch (not the starter).
- `README.md` — the 3 live make.fun badges from the requirements'
  `how_to_submit.readme_template` (owner JordyPras).
- `LICENSE` — MIT, copied from the kit's starter.
- `.makefun/owner.json` — exactly
  `{"wallet":"5FDn6v1pbyqe6ZChzwyrgeqsVpBjwhXUBVXXMWFazekc"}`.

## The data

A case is `{schema, id, at, horizon_minutes, target, levels, game,
trophies, history, notes}`. `history` is every read of Exophase's GTA VI
PS5 trophy page before the checkpoint, each `{slot, at, players_tracked,
trophies_earned, rates}` (55 rates, null where a read lacked the trophy).
The final case f00 is cut at the deadline 2026-11-21T11:30:00Z with a
6,510-minute horizon to 2026-11-26T00:00:00Z. The practice cases p00/p01
are 90-minute horizons on launch night.

Real-world anchor used for design (not by the program): the live
Exophase page fetched 2026-11-19 12:20 UTC showed the launch-night page
(638 players, story trophies ~1.10%) had refreshed to ~4,200 players
with Chapter 1 at 0.49%, Chapters 2-4 at 0.23%, "Beat Cal in a fishing
challenge" at 1.16%, most others 0.10-0.39%, and "Just Like Paradise"
(Gone Fishing) still at 100%. Dilution dominates early: the tracked
count grows faster than earners. Over the week, progression takes over
and the early trophies rise steeply, then bend toward saturation. So
the rate's path in time is S-shaped, not a straight line.

## The model (per trophy, logit scale)

- `x = ln(1 + hours since release)`, `y = logit(rate)` with rates
  clamped to [0.01, 99.99] exactly as the scorer clamps.
- Consecutive identical snapshots are deduplicated first (anchored at
  the last read that showed them): Exophase refreshes on its own
  schedule, so the recorder re-reads the same page for hours.
- **>= 6 distinct snapshots:** weighted least squares of
  `y = a + b*x + c*x^2`, weights halving every 12 h. Curvature `c` is
  shrunk toward 0 by empirical Bayes (prior sd 0.3, weight by the fit's
  own standard error), then the line is refit with `c` fixed at the
  shrunk value so `(b, c)` stay coherent. The median is anchored at the
  last observed rate and moved along the fitted curve to the target;
  the total move is capped at 3 logit units. Residual variance is
  floored at 0.05 logit units (with few degrees of freedom it can read
  ~0 by chance, which would trust a wild curvature).
- **3-5 distinct snapshots:** the quadratic is unidentified, so the
  program uses the house curve's estimator verbatim (straight line over
  every read in the last 24 h, read at the target). This ties the curve
  instead of guessing.
- **< 3 distinct snapshots:** persistence.
- **Spread:** `s^2 = (0.5*sqrt(days))^2 + prediction variance of the
  fit`, floored at 0.6 logit units (a refresh can jump every rate at
  once, as launch night showed: 638 -> 3,868 players between reads)
  and capped at 2.5. The prediction variance comes from the fit's
  design matrix, so it widens with extrapolation leverage.

## Bugs found by validation (both fixed in main.py)

1. A perfect fit gives `se = 0`, and the shrinkage divided by zero
   (`ZeroDivisionError`). Guarded: a zero standard error trusts the
   fitted slope, an undefined one trusts the prior.
2. Capping the curvature term on its own flips the median's sign: an
   accelerating fit has a negative `b` and a large positive `c`, and
   only their combination (local slope `b + 2*c*x`) is meaningful.
   The cap now applies to the total median move only.

## Validation

Tooling in `../work/` (not part of the entry): `validate.py` (RULES
output contract + pinball scorer + both house baselines reimplemented
from RULES.txt) and `synth.py` (synthetic final cases under three
data-generating processes, all anchored on the real 12:20 UTC page:
A = logit-linear in log-time; B = saturating logistic in clock time,
accelerating late; C = logit-linear until the deadline, then half the
slope, i.e. late saturation; plus a sparse variant with only 4
distinct snapshots in the whole history).

- **Practice p00/p01:** outputs valid per the RULES contract (55 rows,
  ids once, 5 finite non-decreasing quantiles 0-100, 2,682 bytes),
  byte-identical across reruns, runtime 0.06 s. With one distinct
  snapshot in the history the forecast is persistence with the 0.6
  floored spread (e.g. Chapter 1: [0.4695, 0.8377, 1.25, 1.8614,
  3.2849] around the last rate 1.25%).
- **Practice proxy score:** the true practice answers are not in the
  kit, so a dilution-only proxy was built (earned counts frozen at the
  stale snapshot, players 638 -> 3,868). On it, this entry scores
  1.3495 vs the house baseline's 1.6877 (RULES says the baseline scores
  ~1.51 on the true answer). The proxy understates the true answer
  (earners grew across the refresh), so it favours wide spreads; it is
  a sanity check on spread width, not the true score. The RULES text
  itself says the practice is little guide to the final's week.
- **Synthetic final f00** (mean pinball loss, lower is better):

  | scenario | this entry | house curve | persistence |
  |---|---|---|---|
  | A logit-linear truth | 0.2853 | 0.2173 | 0.4329 |
  | B saturating logistic | 0.5090 | 0.8246 | 2.2418 |
  | C late saturation | 0.2316 | 0.2789 | 0.2986 |
  | A sparse (4 snapshots) | 0.3796 | 0.3796 | 0.4680 |
  | B sparse (4 snapshots) | 0.3991 | 0.3991 | 2.4311 |

  Mean over A/B/C: this entry 0.342 vs curve 0.440. The entry beats
  the curve on the accelerating (B) and saturating (C) regimes, ties
  it exactly when data is sparse, and trails it by ~0.07 only on the
  exactly-linear truth (A) -- the price of the curvature insurance.
  The real regime is almost certainly not exactly linear: the history
  itself contains the dilution dip followed by the rise.

## Caveats

- The synthetic DGPs are my own assumptions, anchored on one real
  snapshot (12:20 UTC Nov 19). The true final trajectory is unknown;
  the numbers above measure the estimator under those assumptions,
  not the real final.
- The superbot house entry cannot be measured from the kit (its
  forecast is made 2026-11-21T09:30:00Z and replayed); beating the
  curve is necessary but may not be sufficient if the superbot does
  better than the curve.
- The 100% trophy ("Just Like Paradise") stays near 99.99% by
  construction: every read shows 100%, so its fitted slope is ~0.
