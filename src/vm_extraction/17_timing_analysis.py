"""
PhishLens - Timing Analysis: Success Rate vs Verification Age
==================================================================
Computes render success rate as a function of hours elapsed between
PhishTank verifying a URL as "online" and us actually attempting to
render it. Requires the fixed 14_rendered_page_extraction.py and
15_refresh_phishtank_sample.py (which now record processed_at and
verification_time respectively - earlier versions dropped these).

This is the real, quantified evidence for the "phishing infrastructure
has a short operational lifespan" claim - not an inference from a
10-URL diagnostic sample.

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
SAMPLE_FILE   = PROCESSED / "render_sample_urls.csv"


def main():
    print("=" * 62)
    print("PhishLens - Timing Analysis: Success vs Verification Age")
    print("=" * 62)

    rendered = pd.read_csv(RENDERED_FILE)
    sample = pd.read_csv(SAMPLE_FILE)

    time_cols = ["verification_time"]
    if "submission_time" in sample.columns:
        time_cols.append("submission_time")
    df = rendered.merge(sample[["url"] + time_cols], on="url", how="left")

    missing_vt = df["verification_time"].isna().sum()
    missing_pa = df["processed_at"].isna().sum() if "processed_at" in df.columns else len(df)

    if missing_vt == len(df) or missing_pa == len(df):
        print("\nERROR: no rows have both verification_time and processed_at.")
        print("This means the run used the OLD scripts (before the timing fix).")
        print("Only URLs processed with the FIXED scripts will have this data.")
        return

    usable = df.dropna(subset=["verification_time", "processed_at"]).copy()
    print(f"\nTotal rendered rows   : {len(df):,}")
    print(f"Usable for timing (has both timestamps): {len(usable):,}")
    if missing_vt > 0 or missing_pa > 0:
        print(f"  (excluded {len(df)-len(usable):,} rows processed before the "
              f"timing fix, or Tranco rows with no verification_time)")

    usable["verification_time"] = pd.to_datetime(usable["verification_time"], utc=True, errors="coerce")
    usable["processed_at"] = pd.to_datetime(usable["processed_at"], utc=True, errors="coerce")
    if "submission_time" in usable.columns:
        usable["submission_time"] = pd.to_datetime(usable["submission_time"], utc=True, errors="coerce")
    usable = usable.dropna(subset=["verification_time", "processed_at"])

    usable["gap_hours"] = (
        (usable["processed_at"] - usable["verification_time"]).dt.total_seconds() / 3600
    )
    usable["success"] = usable["render_error"] == 0

    # If submission_time is available, also report the fuller pipeline:
    # submission -> verification -> our visit
    has_submission = "submission_time" in usable.columns and usable["submission_time"].notna().any()
    if has_submission:
        sub_valid = usable.dropna(subset=["submission_time"]).copy()
        sub_valid["submit_to_verify_hrs"] = (
            (sub_valid["verification_time"] - sub_valid["submission_time"]).dt.total_seconds() / 3600
        )
        sub_valid["submit_to_processed_hrs"] = (
            (sub_valid["processed_at"] - sub_valid["submission_time"]).dt.total_seconds() / 3600
        )
        print(f"\nFull pipeline timing (n={len(sub_valid):,} rows with submission_time):")
        print(f"  Median submission -> verification: "
              f"{sub_valid['submit_to_verify_hrs'].median():.2f} hours")
        print(f"  Median submission -> our visit    : "
              f"{sub_valid['submit_to_processed_hrs'].median():.2f} hours")
        print(f"  Median verification -> our visit  : "
              f"{sub_valid['gap_hours'].median():.2f} hours")

    bins = [0, 1, 3, 6, 12, 24, 10000]
    labels = ["0-1h", "1-3h", "3-6h", "6-12h", "12-24h", "24h+"]
    usable["bucket"] = pd.cut(usable["gap_hours"], bins=bins, labels=labels)

    summary = usable.groupby("bucket", observed=True)["success"].agg(
        success_rate="mean", n="count"
    )
    summary["success_rate_pct"] = (summary["success_rate"] * 100).round(1)

    print("\nSuccess rate by time-since-verification:")
    print(summary[["n", "success_rate_pct"]])

    with open(REPORTS / "render_timing_analysis.txt", "w") as f:
        f.write("PhishLens - Render Success Rate vs Verification Age\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total usable rows: {len(usable):,}\n\n")
        f.write(summary[["n", "success_rate_pct"]].to_string())
        f.write("\n\nInterpretation: this quantifies how quickly phishing\n")
        f.write("infrastructure becomes unreachable after PhishTank verifies\n")
        f.write("it as online, supporting the case for real-time, client-side\n")
        f.write("detection over blacklist-dependent approaches.\n")
    print(f"\nSaved -> reports/render_timing_analysis.txt")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary["success_rate_pct"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Render Success Rate (%)")
    ax.set_xlabel("Time Since PhishTank Verification")
    ax.set_title("Phishing URL Availability Decay Over Time")
    ax.set_ylim(0, 100)
    for i, v in enumerate(summary["success_rate_pct"]):
        ax.text(i, v + 2, f"{v:.0f}%\n(n={summary['n'].iloc[i]})",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS / "render_timing_analysis.png", dpi=150)
    plt.close()
    print(f"Saved -> reports/render_timing_analysis.png")

    print("\n" + "=" * 62)
    print("This is your real, quantified evidence for phishing")
    print("infrastructure lifespan - cite this, not the 10-URL diagnostic.")
    print("=" * 62)


if __name__ == "__main__":
    main()
