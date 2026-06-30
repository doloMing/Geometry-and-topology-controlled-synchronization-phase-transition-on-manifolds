from __future__ import annotations

from representative_manifold_dynamics import load_or_run_tablevi_formal


def main() -> None:
    raw, summary = load_or_run_tablevi_formal(recompute=True)
    print(
        summary.groupby(["case_plain", "chi", "role"])
        .agg(
            samples=("jump_size", "size"),
            jump_median=("jump_size", "median"),
            jump_mean=("jump_size", "mean"),
            scan_median=("max_scan_difference", "median"),
            scan_mean=("max_scan_difference", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )
    print(f"Raw rows: {len(raw)}")


if __name__ == "__main__":
    main()
