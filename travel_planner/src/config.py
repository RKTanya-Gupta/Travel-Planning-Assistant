import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "raw"))
KB_DIR = ROOT / "data" / "knowledge_base"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CUSTOMER_FILE = "Travel.csv"
FLIGHT_FILE = "Clean_Dataset.csv"
ATTRACTION_FILE = "Top Indian Places to Visit.csv"
HOTEL_FILE = "goibibo_com-travel_sample.csv"

MODEL_PATH = MODEL_DIR / "preference_model.joblib"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"

# ML
TARGET = "ProductPitched"
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus"]
# Designation is left out on purpose - it maps almost 1:1 to ProductPitched (leakage)
RANDOM_STATE = 42
# profile fields the user may keep private ("Prefer not to say")
OPTIONAL_FIELDS = ["Occupation", "Gender", "MaritalStatus", "MonthlyIncome"]

TIER_ORDER = ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]

# daily_misc = food + local travel per person per day (INR)
# rail_class = train class used for the rail estimate
# time_value = INR per person per travel hour, used to trade off price vs travel time
TIER_PROFILE = {
    "Basic":        {"hotel_stars": (1, 3), "flight_class": "Economy",  "daily_misc": 800,
                     "rail_class": "SL", "time_value": 60},
    "Standard":     {"hotel_stars": (2, 3), "flight_class": "Economy",  "daily_misc": 1200,
                     "rail_class": "3A", "time_value": 120},
    "Deluxe":       {"hotel_stars": (3, 4), "flight_class": "Economy",  "daily_misc": 1800,
                     "rail_class": "3A", "time_value": 250},
    "Super Deluxe": {"hotel_stars": (4, 5), "flight_class": "Economy",  "daily_misc": 2500,
                     "rail_class": "2A", "time_value": 400},
    "King":         {"hotel_stars": (5, 5), "flight_class": "Business", "daily_misc": 4000,
                     "rail_class": "1A", "time_value": 1000},
}

# Goibibo has no tariff column, so nightly price is estimated from stars
EST_PRICE_BY_STAR = {0: 1500, 1: 1200, 2: 1800, 3: 3000, 4: 5500, 5: 9500}

# Ground transport rates (INR, 2025-26 levels)
# Rail: Mail/Express fare per km incl. reservation/superfast/GST, roughly calibrated on
# Delhi-Mumbai (SL ~600, 3A ~1,700, 2A ~2,450, 1A ~4,100).
RAIL_FARE = {  # class: (per km, fixed per ticket)
    "SL": (0.45, 50), "3A": (1.20, 80), "2A": (1.70, 100), "1A": (2.90, 120),
}
RAIL_KMPH = 55
BUS_FARE_PER_KM = 2.0        # AC sleeper / Volvo, per person
BUS_KMPH = 45
CAB_FARE_PER_KM = 13         # sedan, one way, per vehicle
CAB_DRIVER_ALLOWANCE = 400
CAB_SEATS = 4
CAB_KMPH = 55
RAIL_DETOUR = 1.2            # straight-line km -> rail km
ROAD_DETOUR = 1.3            # straight-line km -> road km
MAX_BUS_KM = 700
MAX_CAB_KM = 500
MAX_AIRPORT_TRANSFER_KM = 300   # longest road leg to reach an airport
AIRPORT_BUFFER_H = 2.0       # check-in, security, airport transfer
# Non-metro cities with regular scheduled flights. Fares for these are estimated from the
# Kaggle metro fares (same booking lead time, scaled by distance).
AIRPORT_CITIES = [
    "Agra", "Ahmedabad", "Allahabad", "Amritsar", "Aurangabad", "Bhopal", "Bhubaneswar", "Chandigarh",
    "Cochin", "Coimbatore", "Dehradun", "Dharmsala", "Goa", "Gorakhpur", "Guwahati", "Gwalior", "Hubli",
    "Indore", "Jabalpur", "Jaipur", "Jammu", "Jodhpur", "Kannur", "Kullu", "Leh", "Lucknow", "Madurai",
    "Mangalore", "Nagpur", "Patna", "Port Blair", "Pune", "Raipur", "Rajkot", "Ranchi", "Siliguri",
    "Srinagar", "Surat", "Thiruvananthapuram", "Tirupati", "Udaipur", "Vadodara", "Varanasi",
    "Visakhapatnam",
]
MIN_FLIGHT_KM = 250
# used only when a city has no coordinates in the hotel data
GROUND_TRANSPORT_EST_PER_PERSON_ONE_WAY = 2500

DAILY_HOURS = 8.0

# MAAS (TCS GenAI Lab, OpenAI compatible LiteLLM proxy)
MAAS_BASE_URL = os.getenv("MAAS_BASE_URL", "https://genailab.tcs.in")
MAAS_API_KEY = os.getenv("MAAS_API_KEY", "")
MAAS_MODEL = os.getenv("MAAS_MODEL", "genailab-maas-gpt-5.4")
# the lab gateway uses an internal certificate, so verification is off by default
MAAS_VERIFY_SSL = os.getenv("MAAS_VERIFY_SSL", "false").lower() in ("1", "true", "yes")
MAAS_TIMEOUT = float(os.getenv("MAAS_TIMEOUT", "60"))
