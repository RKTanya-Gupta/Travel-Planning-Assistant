"""Random test data with the same columns as the Kaggle files.

Only used by the smoke test, written to tests/mock_data/. Not for results.
Run:  python tests/make_mock_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "mock_data"
rng = np.random.default_rng(0)


def customers(n=2000):
    tiers = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    t = rng.choice(5, n, p=[.37, .15, .35, .07, .06])
    income = 16000 + t * 4500 + rng.normal(0, 2500, n)
    df = pd.DataFrame({
        "CustomerID": range(200000, 200000 + n),
        "ProdTaken": rng.integers(0, 2, n),
        "Age": (28 + t * 5 + rng.normal(0, 7, n)).round(),
        "TypeofContact": rng.choice(["Self Enquiry", "Company Invited"], n),
        "CityTier": rng.choice([1, 2, 3], n),
        "DurationOfPitch": rng.integers(5, 36, n),
        "Occupation": rng.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"], n),
        "Gender": rng.choice(["Male", "Female", "Fe Male"], n),
        "NumberOfPersonVisiting": rng.integers(1, 5, n),
        "NumberOfFollowups": rng.integers(1, 6, n),
        "ProductPitched": [tiers[i] for i in t],
        "PreferredPropertyStar": np.clip(3 + (t >= 3) + rng.integers(0, 2, n), 3, 5),
        "MaritalStatus": rng.choice(["Married", "Single", "Divorced", "Unmarried"], n),
        "NumberOfTrips": rng.integers(1, 8, n),
        "Passport": rng.integers(0, 2, n),
        "PitchSatisfactionScore": rng.integers(1, 6, n),
        "OwnCar": rng.integers(0, 2, n),
        "NumberOfChildrenVisiting": rng.integers(0, 3, n),
        "Designation": rng.choice(["Executive", "Manager", "Senior Manager", "AVP", "VP"], n),
        "MonthlyIncome": income.round(),
    })
    df.loc[rng.choice(n, 80, replace=False), "MonthlyIncome"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "Age"] = np.nan
    return df


def flights(n=20000):
    cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    src = rng.choice(cities, n); dst = rng.choice(cities, n)
    keep = src != dst; src, dst = src[keep], dst[keep]; m = len(src)
    cls = rng.choice(["Economy", "Business"], m, p=[.7, .3])
    stops = rng.choice(["zero", "one", "two_or_more"], m, p=[.2, .7, .1])
    dur = np.where(stops == "zero", rng.uniform(1.5, 3, m), rng.uniform(4, 20, m)).round(2)
    days = rng.integers(1, 50, m)
    price = (4000 + 300 * dur - 60 * days + rng.normal(0, 800, m)).clip(1800) * np.where(cls == "Business", 6, 1)
    slots = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    return pd.DataFrame({
        "Unnamed: 0": range(m), "airline": rng.choice(["Vistara", "Air_India", "Indigo", "GO_FIRST", "AirAsia", "SpiceJet"], m),
        "flight": [f"UK-{x}" for x in rng.integers(100, 999, m)], "source_city": src,
        "departure_time": rng.choice(slots, m), "stops": stops, "arrival_time": rng.choice(slots, m),
        "destination_city": dst, "class": cls, "duration": dur, "days_left": days, "price": price.round().astype(int),
    })


def attractions():
    rows = []
    spec = {"Delhi": ("Northern", "Delhi"), "Mumbai": ("Western", "Maharashtra"), "Jaipur": ("Northern", "Rajasthan"),
            "Goa": ("Western", "Goa"), "Bangalore": ("Southern", "Karnataka"), "Chennai": ("Southern", "Tamil Nadu")}
    types = [("Fort", "Historical"), ("Temple", "Religious"), ("Park", "Nature"), ("Museum", "Historical"),
             ("Beach", "Recreational"), ("Market", "Shopping"), ("Lake", "Nature"), ("Palace", "Historical")]
    for city, (zone, state) in spec.items():
        for i in range(10):
            t, s = types[rng.integers(len(types))]
            rows.append({"Unnamed: 0": len(rows), "Zone": zone, "State": state, "City": city,
                         "Name": f"{city} {t} {i+1}", "Type": t, "Establishment Year": int(rng.integers(1500, 1990)),
                         "time needed to visit in hrs": float(rng.choice([0.5, 1, 1.5, 2, 3, 4])),
                         "Google review rating": round(float(rng.uniform(3.8, 4.8)), 1),
                         "Entrance Fee in INR": int(rng.choice([0, 0, 20, 50, 100, 250, 500])),
                         "Airport with 50km Radius": "Yes",
                         "Weekly Off": rng.choice(["None", "None", "None", "Monday", "Friday"]),
                         "Significance": s, "DSLR Allowed": "Yes",
                         "Number of google review in lakhs": round(float(rng.uniform(0.05, 1.5)), 2),
                         "Best Time to visit": rng.choice(["Morning", "Afternoon", "Evening", "All"])})
    return pd.DataFrame(rows)


CITY_XY = {"Delhi": (28.63, 77.21), "New Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88),
           "Jaipur": (26.91, 75.79), "Goa": (15.50, 73.83), "Bangalore": (12.97, 77.59),
           "Chennai": (13.08, 80.27), "Pune": (18.52, 73.86), "Kolkata": (22.57, 88.36),
           "Hyderabad": (17.39, 78.49)}


def hotels():
    rows = []
    for city, (lat, lon) in CITY_XY.items():
        for i in range(15):
            star = int(rng.integers(1, 6))
            rows.append({"city": city, "property_name": f"Hotel {city} {i+1}", "hotel_star_rating": f"{star} star",
                         "site_review_rating": round(float(rng.uniform(3, 5)), 1), "locality": "Central",
                         "hotel_facilities": "Free WiFi|Parking|Restaurant" + ("|Swimming Pool" if i % 3 == 0 else ""),
                         "hotel_description": f"A {star}-star property in {city} with easy access to sights.",
                         "property_type": "Hotel", "latitude": lat + rng.normal(0, 0.02),
                         "longitude": lon + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    customers().to_csv(OUT / "Travel.csv", index=False)
    flights().to_csv(OUT / "Clean_Dataset.csv", index=False)
    attractions().to_csv(OUT / "Top Indian Places to Visit.csv", index=False)
    hotels().to_csv(OUT / "goibibo_com-travel_sample.csv", index=False)
    print(f"Synthetic mock data written to {OUT}")
