from pathlib import Path

import pandas as pd


def save_summary_table(df: pd.DataFrame, out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient='records', indent=2)
