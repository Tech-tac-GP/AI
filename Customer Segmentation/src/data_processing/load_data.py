from pathlib import Path
import pandas as pd

def load_events(path):
    df = pd.read_csv(Path(path))
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df
