"""
PhishLens - Timing Analysis v2: Success Rate vs Verification Age
==================================================================
Computes render success rate as a function of hours elapsed between
PhishTank verifying a URL as "online" and the VM actually attempting
to render it. This is the quantified evidence for the "phishing
infrastructure has a short operational lifespan" claim (Chapter 4/5).

v2 (14 Jul 2026): reads ONLY rendered_page_features.csv - the v2
extraction script now merges source/label/verification_time/
submission_time into its output directly, so the old two-file merge
produced suffixed duplicate columns (verification_time_x/_y) and a
KeyError. No merge needed anymore.

Usage:
    python3 17_timing_analysis.py

Outputs:
    reports/render_timing_analysis.txt
    reports/render_timing_analysis.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
REPORTS   = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RENDERED_FILE = PROCESSED / "rendered_page_features.csv"
TXT_OUT = REPORTS / "render_timing_analysis.txt"
PNG_OUT = REPORTS / "render_timing_analysis.png"


def main():
    print("=" * 62)
    print("PhishLens - Timing Analysis v2: Success vs Verification Age")
    print("=" * 62)

    df = pd.read_csv(RENDERED_FILE)

    required = ["source", "render_error", "processed_at",
                "verification_time", "submission_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {RENDERED_FILE.name}: {missing}. "
                         f"Found: {df.columns.tolist()}")

    pt = df[df["source"] == "phishtank"].copy()
    print(f"\nPhishTank rows: {len(pt)}")

    for col in ["processed_at", "verification_time", "submission_time"]:
        pt[col] = pd.to_datetime(pt[col], utc=True, errors="coerce")

    n_missing_vt = pt["verification_time"].isna().sum()
    n_missing_st = pt["submission_time"].isna().sum()
    print(f"Rows missing verification_time: {n_missing_vt}")
    print(f"Rows missing submission_time  : {n_missing_st}")

    pt = pt.dropna(subset=["processed_at", "verification_time"])
    print(f"Rows usable for timing analysis: {len(pt)}")

    pt["age_hours"] = (
        (pt["processed_at"] - pt["verification_time"]).dt.total_seconds() / 3600
    )
    pt["sub_to_ver_hours"] = (
        (pt["verification_time"] - pt["submission_time"]).dt.total_seconds() / 3600
    )
    pt["total_hours"] = (
        (pt["processed_at"] - pt["submission_time"]).dt.total_seconds() / 3600
    )
    pt["alive"] = 1 - pt["render_error"]

    lines = []
    lines.append("PhishLens - PhishTank URL timing analysis")
    lines.append(f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}")
    lines.append(f"PhishTank URLs analysed: {len(pt)}")
    lines.append("")
    lines.append("Median hours, submission -> verification : "
                 f"{pt['sub_to_ver_hours'].median():.1f}")
    lines.append("Median hours, verification -> our visit  : "
                 f"{pt['age_hours'].median():.1f}")
    lines.append("Median hours, submission -> our visit    : "
                 f"{pt['total_hours'].median():.1f}")
    lines.append("")
    lines.append(f"Overall alive rate at visit: {pt['alive'].mean()*100:.1f}%")
    lines.append("")

    bins   = [0, 3, 6, 12, 24, 48, 96, float("inf")]
    labels = ["0-3h", "3-6h", "6-12h", "12-24h", "24-48h", "48-96h", "96h+"]
    pt["age_bucket"] = pd.cut(pt["age_hours"], bins=bins, labels=labels,
                              include_lowest=True)
    bucket = (pt.groupby("age_bucket", observed=True)["alive"]
                .agg(["mean", "count"]))
    bucket["alive_pct"] = (bucket["mean"] * 100).round(1)

    lines.append("Alive rate by verification age at visit:")
    lines.append(f"{'bucket':<8} {'alive %':>8} {'n':>6}")
    for b, row in bucket.iterrows():
        lines.append(f"{b:<8} {row['alive_pct']:>7.1f}% {int(row['count']):>6}")

    report = "\n".join(lines)
    print("\n" + report)
    TXT_OUT.write_text(report)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_data = bucket.dropna(subset=["alive_pct"])
    ax.bar(plot_data.index.astype(str), plot_data["alive_pct"],
           color="#2a78d6", width=0.6)
    for i, (b, row) in enumerate(plot_data.iterrows()):
        ax.text(i, row["alive_pct"] + 1.5,
                f"{row['alive_pct']:.0f}%\n(n={int(row['count'])})",
                ha="center", fontsize=8)
    ax.set_xlabel("Hours between PhishTank verification and render attempt")
    ax.set_ylabel("URLs still reachable (%)")
    ax.set_title("PhishTank URL survival vs verification age (PhishLens VM)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=200)

    print(f"\nSaved: {TXT_OUT}")
    print(f"Saved: {PNG_OUT}")
    print("(The PNG is a citable Chapter 4 figure.)")


if __name__ == "__main__":
    main()
