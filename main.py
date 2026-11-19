"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target, from the case's
history of reads of the game's trophy page.

Why a trend, and why a curve
----------------------------
Exophase's tracked-player count grows through launch week while players
progress through the game, so a trophy's earn rate is pulled two ways:
dilution (new players syncing with no trophies yet) drags it down, and
players earning it pushes it up. On launch night the page sat unchanged
for hours (638 players tracked), then refreshed to 3,868 players and
every rate but the one at 100% lower. A persistence forecast misses the
whole move over a week. The rate's path is not a straight line either:
it dips on dilution, then rises steeply as the launch cohort plays
through, then bends as the early trophies saturate. So the program fits,
per trophy, a quadratic in log-time -- and shrinks the curvature hard,
because a quadratic through a handful of 2-decimal reads is noisy.

Model (per trophy, on the logit scale)
--------------------------------------
  * x = ln(1 + hours since the game's release) for each read. Trophy
    progress is roughly power-law in time since launch, so the log-time
    axis both linearises the growth and saturates it.
  * y = logit(rate), rate clamped to [0.01, 99.99] exactly as the scorer
    clamps it. Consecutive reads that repeat the same page (Exophase
    refreshes on its own schedule, not the recorder's) are deduplicated
    first, anchored at the last read that showed them, so a stale page
    cannot outvote the fresh reads.
  * With >= 6 distinct snapshots: weighted least squares of
    y = a + b*x + c*x^2, weights halving every 12 h. The curvature c is
    shrunk toward 0 by empirical Bayes (prior sd 0.3 logit per x^2-unit,
    weight by the fit's own standard error), then the line is refit with
    c fixed at the shrunk value so (b, c) stay coherent. The median is
    anchored at the last observed rate and moved along the fitted curve
    to the target; the total move is capped at 3 logit units as a guard
    against wild extrapolation. The residual variance is floored at
    0.05 logit units: with few degrees of freedom it can read ~0 by
    chance, which would trust a wild curvature.
  * With 3-5 distinct snapshots the quadratic is unidentified, so the
    program uses the house curve's estimator verbatim: a straight line
    over every read in the last 24 h (repeats included), read at the
    target. This ties the curve instead of guessing.
  * With < 3 distinct snapshots there is nothing to fit: persistence.
  * The spread is an honest prediction interval:
    s^2 = (0.5 * sqrt(days))^2 + prediction variance of the fit,
    floored at 0.6 logit units (a refresh can jump every rate at once,
    as launch night showed) and capped at 2.5. The prediction variance
    comes from the fit's design matrix, so it widens with the leverage
    of extrapolating past the data.

With a single distinct snapshot in the history (the practice cases: the
page sat unchanged all launch night) the forecast is persistence with
the floored spread, and the program still writes a valid forecast.

Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles non-decreasing, at most 65,536 bytes.
Deterministic, no network, no model calls.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread, logit units per square-root day (as the house baseline).
SIGMA_RW = 0.5
# Recency weights halve every HALF_LIFE_H hours.
HALF_LIFE_H = 12.0
# Prior sd of the curvature, logit units per x^2-unit (shrinkage target 0).
TAU_C = 0.3
# Distinct snapshots needed before the quadratic is trusted.
QUAD_MIN_POINTS = 6
# Floor on the residual sd, logit units (the page prints 2 decimals).
RESID_FLOOR = 0.05
# Cap on the total median move from the last observed rate, logit units.
MOVE_CAP = 3.0
# Spread floor (a refresh can jump every rate at once) and cap, logit units.
SPREAD_FLOOR = 0.6
SPREAD_CAP = 2.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def solve3(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a 3x3 system by Gaussian elimination with partial pivoting."""
    m = [row[:] + [bb] for row, bb in zip(a, b)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def inv3(a: list[list[float]]) -> list[list[float]] | None:
    """Invert a 3x3 matrix via solve3 on the unit vectors."""
    cols = []
    for j in range(3):
        e = [1.0 if k == j else 0.0 for k in range(3)]
        col = solve3(a, e)
        if col is None:
            return None
        cols.append(col)
    return [[cols[j][i] for j in range(3)] for i in range(3)]


def quantiles(med_logit: float, spread: float) -> list[float | int]:
    return [round4(100.0 / (1.0 + math.exp(-(med_logit + z * spread)))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    horizon_days = case["horizon_minutes"] / 1440.0
    s_rw = SIGMA_RW * math.sqrt(horizon_days)

    # Deduplicate consecutive identical snapshots: Exophase refreshes on
    # its own schedule, so the recorder often re-reads the same page.
    # A snapshot is anchored at the LAST read that showed it: that is
    # when the page was last known to hold those rates.
    reads = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            reads[-1] = read
            continue
        prev_key = key
        reads.append(read)
    if not reads:
        reads = history

    times_h = []
    for read in reads:
        at = parse_time(read["at"])
        h = (at - release).total_seconds() / 3600.0
        times_h.append(max(h, -0.95))  # keep 1 + h safely positive
    t_last = times_h[-1]
    x_last = math.log1p(t_last)

    target = parse_time(case["target"])
    h_target = max((target - release).total_seconds() / 3600.0, -0.95)
    x_target = math.log1p(h_target)

    rows = []
    for i, trophy in enumerate(trophies):
        pts: list[tuple[float, float, float]] = []
        y_last = None
        for j, read in enumerate(reads):
            rate = read["rates"][i]
            if rate is None:
                continue
            x = math.log1p(times_h[j])
            w = 0.5 ** ((t_last - times_h[j]) / HALF_LIFE_H)
            pts.append((x, logit(rate), w))
            y_last = logit(rate)
        if y_last is None:
            # No read ever listed this trophy: flat at 1% with a wide band.
            rows.append({"id": trophy["id"], "q": quantiles(logit(1.0), SPREAD_CAP)})
            continue
        if len(pts) < 3:
            # Too little to fit anything: persistence + floored spread.
            med = y_last
            spread = max(SPREAD_FLOOR, s_rw)
        elif len(pts) < QUAD_MIN_POINTS:
            # Sparse history: the quadratic is unidentified, so use the
            # house curve's estimator verbatim -- a straight line over
            # every read in the last 24 h (repeats included), read at the
            # target. This ties the curve instead of guessing.
            cutoff = parse_time(case["at"]).timestamp() - 24 * 3600
            recent = []
            for read in history:
                if parse_time(read["at"]).timestamp() < cutoff:
                    continue
                r = read["rates"][i]
                if r is None:
                    continue
                h = max((parse_time(read["at"]) - release).total_seconds() / 3600.0, -0.95)
                recent.append((math.log1p(h), logit(r)))
            if len(recent) < 3:
                med = y_last
                spread = max(SPREAD_FLOOR, s_rw)
            else:
                n = len(recent)
                xb = sum(p[0] for p in recent) / n
                yb = sum(p[1] for p in recent) / n
                sxx = sum((p[0] - xb) ** 2 for p in recent)
                sxy = sum((p[0] - xb) * (p[1] - yb) for p in recent)
                b = sxy / sxx if sxx > 1e-12 else 0.0
                a = yb - b * xb
                med = a + b * x_target
                rss = sum((p[1] - a - b * p[0]) ** 2 for p in recent)
                resid = math.sqrt(rss / (n - 2)) if n > 2 else 0.0
                spread = math.sqrt(s_rw * s_rw + resid * resid)
                spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
                med = max(-9.21, min(9.21, med))
        else:
            # Weighted least squares of y = a + b*x + c*x^2.
            xtwx = [[0.0] * 3 for _ in range(3)]
            xtwy = [0.0] * 3
            for x, y, w in pts:
                v = [1.0, x, x * x]
                for a in range(3):
                    xtwy[a] += w * v[a] * y
                    for b in range(3):
                        xtwx[a][b] += w * v[a] * v[b]
            coef = solve3(xtwx, xtwy)
            if coef is None:
                med = y_last
                spread = max(SPREAD_FLOOR, s_rw)
                rows.append({"id": trophy["id"], "q": quantiles(med, spread)})
                continue
            _a0, b0, c0 = coef
            n = len(pts)
            rss = sum(w * (y - (_a0 + b0 * x + c0 * x * x)) ** 2 for x, y, w in pts)
            sw = sum(w for _, _, w in pts)
            resid_var = rss / max(n - 3, 1) / (sw / n)
            # Floor the residual variance: with few degrees of freedom it
            # can read ~0 by chance, which would trust a wild curvature.
            resid_var = max(resid_var, RESID_FLOOR ** 2)
            inv = inv3(xtwx)
            if inv is None:
                se_c = float("inf")
            else:
                se_c = math.sqrt(max(resid_var * inv[2][2], 0.0))
            if se_c == 0.0:
                c_s = c0
            elif se_c == float("inf"):
                c_s = 0.0
            else:
                # Empirical-Bayes shrinkage of the curvature toward 0.
                c_s = c0 * TAU_C * TAU_C / (se_c * se_c + TAU_C * TAU_C)
            # Refit the line with curvature fixed at the shrunk value, so
            # (b, c) stay coherent instead of b carrying the unshrunk c.
            sw2 = sum(w for _, _, w in pts)
            xb = sum(w * x for x, _, w in pts) / sw2
            yb = sum(w * (y - c_s * x * x) for x, y, w in pts) / sw2
            sxx = sum(w * (x - xb) ** 2 for x, _, w in pts)
            sxy = sum(w * (x - xb) * ((y - c_s * x * x) - yb) for x, y, w in pts)
            if sxx > 1e-12:
                b0 = sxy / sxx
            # The move along the fitted curve from the last read. NOTE: b
            # and c are meaningless apart -- the local slope is b + 2*c*x,
            # so the curvature term must NOT be capped on its own (an
            # accelerating fit has a negative b and a large positive c,
            # and capping c alone flips the move negative). Only the
            # total move is bounded, as a guard against wild extrapolation.
            move = b0 * (x_target - x_last) + c_s * (x_target * x_target - x_last * x_last)
            move = max(-MOVE_CAP, min(MOVE_CAP, move))
            med = y_last + move
            med = max(-9.21, min(9.21, med))
            if inv is not None:
                v = [1.0, x_target, x_target * x_target]
                lev = sum(v[a] * sum(inv[a][b] * v[b] for b in range(3)) for a in range(3))
                pred_var = resid_var * (1.0 + max(lev, 0.0))
                s_fit = math.sqrt(max(pred_var, 0.0))
            else:
                s_fit = 0.5
            spread = math.sqrt(s_rw * s_rw + s_fit * s_fit)
            spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
        rows.append({"id": trophy["id"], "q": quantiles(med, spread)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
