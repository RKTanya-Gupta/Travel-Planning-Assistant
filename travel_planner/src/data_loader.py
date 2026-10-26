"""Loading and cleaning of the four Kaggle datasets.

Columns are matched by normalised name + a few aliases so slightly different
versions of the same dataset still load.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _pick(df: pd.DataFrame, aliases, required=True):
    lookup = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in lookup:
            return lookup[_norm(a)]
    if required:
        raise KeyError(f"None of {aliases} found in columns {list(df.columns)}")
    return None


def _path(name):
    p = config.DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it from Kaggle (see docs/dataset_sources.md) "
            f"and place it in {config.DATA_DIR}."
        )
    return p


def _to_number(series):
    # '3 star', '4.2/5', '₹ 2,499' -> float
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


# ---------------------------------------------------------------- customers --
def load_customers() -> pd.DataFrame:
    df = pd.read_csv(_path(config.CUSTOMER_FILE))
    if "Gender" in df:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    df = df.dropna(subset=[config.TARGET])
    return df


# ------------------------------------------------------------------ flights --
@lru_cache(maxsize=1)
def load_flights() -> pd.DataFrame:
    raw = pd.read_csv(_path(config.FLIGHT_FILE))
    out = pd.DataFrame({
        "airline": raw[_pick(raw, ["airline"])],
        "flight": raw[_pick(raw, ["flight"])],
        "source_city": raw[_pick(raw, ["source_city", "source"])],
        "destination_city": raw[_pick(raw, ["destination_city", "destination"])],
        "departure_time": raw[_pick(raw, ["departure_time"])],
        "arrival_time": raw[_pick(raw, ["arrival_time"])],
        "stops": raw[_pick(raw, ["stops"])],
        "class": raw[_pick(raw, ["class"])],
        "duration": pd.to_numeric(raw[_pick(raw, ["duration"])], errors="coerce"),
        "days_left": pd.to_numeric(raw[_pick(raw, ["days_left"])], errors="coerce"),
        "price": pd.to_numeric(raw[_pick(raw, ["price"])], errors="coerce"),
    })
    return out.dropna(subset=["price", "duration", "days_left"])


# -------------------------------------------------------------- attractions --
@lru_cache(maxsize=1)
def load_attractions() -> pd.DataFrame:
    raw = pd.read_csv(_path(config.ATTRACTION_FILE))
    g = lambda al, req=True: _pick(raw, al, req)
    out = pd.DataFrame({
        "zone": raw[g(["Zone"])],
        "state": raw[g(["State"])],
        "city": raw[g(["City"])].astype(str).str.strip(),
        "name": raw[g(["Name"])].astype(str).str.strip(),
        "type": raw[g(["Type"])].astype(str).str.strip(),
        "hours": pd.to_numeric(raw[g(["time needed to visit in hrs", "Time Needed to Visit (hrs)"])], errors="coerce"),
        "rating": pd.to_numeric(raw[g(["Google review rating", "rating"])], errors="coerce"),
        "fee": pd.to_numeric(raw[g(["Entrance Fee in INR", "entrance_fee"])], errors="coerce"),
        "weekly_off": raw[g(["Weekly Off"], False)] if g(["Weekly Off"], False) else "None",
        "significance": raw[g(["Significance"], False)] if g(["Significance"], False) else raw[g(["Type"])],
        "reviews_lakhs": pd.to_numeric(raw[g(["Number of google review in lakhs"], False)], errors="coerce")
        if g(["Number of google review in lakhs"], False) else 0.1,
        "best_time": raw[g(["Best Time to visit"], False)] if g(["Best Time to visit"], False) else "Anytime",
        "year": raw[g(["Establishment Year"], False)] if g(["Establishment Year"], False) else "",
        "airport_50km": raw[g(["Airport with 50km Radius"], False)] if g(["Airport with 50km Radius"], False) else "",
        "dslr": raw[g(["DSLR Allowed"], False)] if g(["DSLR Allowed"], False) else "",
    })
    out["hours"] = out["hours"].fillna(out["hours"].median()).clip(0.5, 8)
    out["rating"] = out["rating"].fillna(out["rating"].median())
    out["fee"] = out["fee"].fillna(0)
    out["reviews_lakhs"] = out["reviews_lakhs"].fillna(0.0)
    for c in ["weekly_off", "significance", "best_time"]:
        out[c] = out[c].fillna("None" if c == "weekly_off" else "Anytime").astype(str).str.strip()
    for c in ["year", "airport_50km", "dslr"]:
        out[c] = out[c].fillna("").astype(str).str.strip().replace({"nan": "", "Unknown": ""})
    return out.drop_duplicates(subset=["city", "name"]).reset_index(drop=True)


# ------------------------------------------------------------------- hotels --
@lru_cache(maxsize=1)
def load_hotels() -> pd.DataFrame:
    raw = pd.read_csv(_path(config.HOTEL_FILE))
    g = lambda al: _pick(raw, al, required=False)
    city_c = _pick(raw, ["city"])
    name_c = _pick(raw, ["property_name", "hotel_name", "name"])
    star_c = g(["hotel_star_rating", "star_rating", "stars"])
    rate_c = g(["site_review_rating", "rating", "mmt_review_score", "review_rating"])
    price_c = g(["price", "room_price", "rate"])
    fac_c = g(["hotel_facilities", "facilities", "amenities"])
    desc_c = g(["hotel_description", "description", "hotel_overview"])
    loc_c = g(["locality", "area", "location", "address"])
    lat_c = g(["latitude", "lat"])
    lon_c = g(["longitude", "lon", "lng"])

    out = pd.DataFrame({
        "city": raw[city_c].astype(str).str.strip(),
        "name": raw[name_c].astype(str).str.strip(),
        "stars": _to_number(raw[star_c]).clip(0, 5) if star_c else np.nan,
        "rating": _to_number(raw[rate_c]) if rate_c else np.nan,
        "locality": raw[loc_c].fillna("").astype(str).replace("Others", "") if loc_c else "",
        "facilities": raw[fac_c].fillna("").astype(str) if fac_c else "",
        "description": raw[desc_c].fillna("").astype(str) if desc_c else "",
        "lat": pd.to_numeric(raw[lat_c], errors="coerce") if lat_c else np.nan,
        "lon": pd.to_numeric(raw[lon_c], errors="coerce") if lon_c else np.nan,
    })
    out["stars"] = out["stars"].fillna(0).round().astype(int)
    out["rating"] = out["rating"].fillna(out["rating"].median() if out["rating"].notna().any() else 3.5)
    if price_c:
        out["price"] = _to_number(raw[price_c])
        out["price_estimated"] = out["price"].isna()
        out["price"] = out["price"].fillna(out["stars"].map(config.EST_PRICE_BY_STAR))
    else:
        out["price"] = out["stars"].map(config.EST_PRICE_BY_STAR)
        out["price_estimated"] = True
    return out.drop_duplicates(subset=["city", "name"]).reset_index(drop=True)


CITY_ALIASES = {
    "bengaluru": "bangalore", "bombay": "mumbai", "new delhi": "delhi", "calcutta": "kolkata",
    "madras": "chennai", "gurugram": "gurgaon", "kochi": "cochin", "mysuru": "mysore",
    "trivandrum": "thiruvananthapuram", "vizag": "visakhapatnam", "puducherry": "pondicherry",
    "allahabad": "prayagraj", "alleppey": "alappuzha",
}


def canon(city):
    c = str(city).strip().lower()
    return CITY_ALIASES.get(c, c)


def city_match(series: pd.Series, city: str) -> pd.Series:
    return series.astype(str).map(canon) == canon(city)


@lru_cache(maxsize=1)
def load_city_coords() -> pd.DataFrame:
    # city location = median location of its hotels in the Goibibo data
    h = load_hotels()
    h = h[h["lat"].between(6, 37) & h["lon"].between(68, 98)].copy()
    h["key"] = h["city"].map(canon)
    out = (h.groupby("key")
             .agg(city=("city", lambda s: s.mode().iat[0]), lat=("lat", "median"),
                  lon=("lon", "median"), hotels=("city", "size"))
             .reset_index())
    return out.set_index("key")


def summary():
    """Row counts of every dataset, to check that all four files are found and loaded."""
    for name, loader in [("customers", load_customers), ("flights", load_flights),
                         ("attractions", load_attractions), ("hotels", load_hotels)]:
        try:
            df = loader()
            cities = f", {df['city'].nunique()} cities" if "city" in df else ""
            print(f"{name:12s} OK  {len(df):>7,} rows{cities}")
        except (FileNotFoundError, KeyError) as e:
            print(f"{name:12s} MISSING  {e}")


if __name__ == "__main__":
    summary()
