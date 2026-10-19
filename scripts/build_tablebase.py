from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_endgame_rl.solver.tablebase import Tablebase, CACHE_DIR


def build_and_save(kind: str) -> None:
    norm_kind = kind.upper()
    piece_name = "Q" if "Q" in norm_kind else "R"
    filename = "kqk.npz" if piece_name == "Q" else "krk.npz"
    out_path = CACHE_DIR / filename

    print(f"[{norm_kind}] Building exact retrograde tablebase for 3-piece endgame...")
    t0 = time.time()
    tb = Tablebase.solve(piece_name)
    elapsed = time.time() - t0

    # Validate metrics
    legal_white = tb.legal[0::2].sum()
    won_white = (tb.dtm[0::2] > 0).sum()
    max_dtm = tb.dtm.max()
    win_rate = won_white / max(legal_white, 1)

    print(f"[{norm_kind}] Solved in {elapsed:.1f}s:")
    print(f"  - Legal White states: {legal_white:,}")
    print(f"  - Won White states:   {won_white:,} ({win_rate:.1%})")
    print(f"  - Max DTM:            {max_dtm} plies ({max_dtm // 2} full moves)")

    tb.save(out_path)
    print(f"[{norm_kind}] Saved cache to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and cache exact endgame tablebases.")
    parser.add_argument(
        "--kind",
        choices=["KQvK", "KRvK", "all"],
        default="all",
        help="Endgame scenario to solve and cache",
    )
    args = parser.parse_args()

    if args.kind in ("KQvK", "all"):
        build_and_save("KQvK")
    if args.kind in ("KRvK", "all"):
        build_and_save("KRvK")


if __name__ == "__main__":
    main()
