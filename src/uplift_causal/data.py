from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[2] / "data"
HILLSTROM_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
CRITEO_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
CRITEO_TYPES = {f"f{i}": pa.float32() for i in range(12)} | {
    c: pa.int8() for c in ("treatment", "conversion", "visit", "exposure")
}


def _cached(url, name, column_types=None, columns=None):
    path = DATA / f"{name}.parquet"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        raw = DATA / url.rsplit("/", 1)[-1]
        if not raw.exists():
            urlretrieve(url, raw)
        opts = pacsv.ConvertOptions(column_types=column_types or {})
        pq.write_table(pacsv.read_csv(raw, convert_options=opts), path)
        raw.unlink()
    return pd.read_parquet(path, columns=columns)


def load_hillstrom(treatment="Womens E-Mail", outcome="visit"):
    df = _cached(HILLSTROM_URL, "hillstrom")
    df = df[df.segment.isin([treatment, "No E-Mail"])].reset_index(drop=True)
    X = pd.get_dummies(
        df[["recency", "mens", "womens", "newbie", "zip_code", "channel"]].assign(
            log_history=np.log1p(df.history)
        ),
        columns=["zip_code", "channel"],
        drop_first=True,
        dtype=np.int8,
    )
    return X.assign(t=(df.segment == treatment).astype(np.int8), y=df[outcome])


def load_criteo(n=None, outcome="visit", seed=0):
    cols = [f"f{i}" for i in range(12)] + ["treatment", outcome]
    df = _cached(CRITEO_URL, "criteo", CRITEO_TYPES, cols)
    if n:
        df = df.sample(n, random_state=seed, ignore_index=True)
    return df.rename(columns={"treatment": "t", outcome: "y"})
