from pathlib import Path

import pandas as pd


def export_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.csv':
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient='records', indent=2)
