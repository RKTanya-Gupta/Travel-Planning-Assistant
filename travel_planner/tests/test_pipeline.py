"""Smoke test on the mock data (no Kaggle download needed).

    python tests/make_mock_data.py
    python tests/test_pipeline.py

The model and reports are written to a temp folder, models/ is not touched.
"""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DATA_DIR"] = str(ROOT / "tests" / "mock_data")
os.environ["MAAS_API_KEY"] = ""  # offline

from src import config, llm  # noqa: E402

tmp = Path(tempfile.mkdtemp())
config.MODEL_DIR, config.REPORT_DIR = tmp, tmp
config.MODEL_PATH, config.MODEL_META_PATH = tmp / "m.joblib", tmp / "m.json"

from src.agents import TravelOrchestrator  # noqa: E402
from src.preference_model import PreferenceModel  # noqa: E402
from src.rag import RAG, trip_brief  # noqa: E402
from src.train_models import main as train  # noqa: E402

PROFILE = dict(Age=35, CityTier=1, NumberOfPersonVisiting=2, PreferredPropertyStar=4, NumberOfTrips=3,
               Passport=1, OwnCar=1, NumberOfChildrenVisiting=0, MonthlyIncome=28000,
               Occupation="Salaried", Gender="Male", MaritalStatus="Married")


def test_models():
    results = train()
    assert len(results) >= 3


def test_trips():
    # "Prefer not to say" fields arrive as None and must still give a tier
    tier, _ = PreferenceModel().predict(dict(PROFILE, Occupation=None, Gender=None, MaritalStatus=None,
                                             MonthlyIncome=None))
    assert tier in config.TIER_ORDER

    orch = TravelOrchestrator(PreferenceModel())
    start = date.today() + timedelta(days=20)

    # metro to metro: Kaggle flight fares, flight chosen for a higher tier
    r = orch.plan_trip(PROFILE, "Delhi", "Mumbai", start, start + timedelta(days=3), 150000, ["Historical"],
                       tier_override="Super Deluxe")
    t = r["transport"]
    assert t["mode"] == "Flight" and t["outbound"] and t["return"]
    assert any(o["mode"].startswith("Train") for o in t["options"])
    assert len(r["itinerary"]) == 4
    for d in r["itinerary"]:
        assert d["hours"] <= config.DAILY_HOURS
        day = d["date"].split()[0]
        for s in d["stops"]:
            off = r["attractions"].set_index("name").loc[s["name"], "weekly_off"]
            assert day.lower()[:3] not in str(off).lower(), "scheduled on weekly off day"

    # short trip, low budget: ground transport and budget adjustments
    r = orch.plan_trip(PROFILE, "Delhi", "Jaipur", start, start + timedelta(days=2), 15000, [])
    assert not r["transport"]["mode"].startswith("Flight")
    assert r["budget"]["adjustments"]

    # non-metro airports: estimated fare
    r = orch.plan_trip(PROFILE, "Jaipur", "Goa", start, start + timedelta(days=3), 80000, [],
                       tier_override="Deluxe")
    assert any("estimated fare" in o["route"] for o in r["transport"]["options"])

    # non-metro source without its own airport in the list -> road + rail options
    r = orch.plan_trip(PROFILE, "Pune", "Jaipur", start, start + timedelta(days=2), 60000, [])
    assert r["transport"]["options"]

    assert "Pune" in orch.transport.from_cities(min_hotels=1)


def test_rag():
    rag = RAG()
    checks = [("What is the hotel cancellation policy?", "Cancellation"),
              ("Is it safe to swim in Goa during the monsoon?", "Goa in the monsoon"),
              ("What ID do I need at check-in?", "Identity documents"),
              ("Suggest a hotel in Goa with a swimming pool", "Hotel Goa")]
    for q, expected in checks:
        _, hits = rag.answer(q)
        assert hits and expected in hits[0]["title"], (q, [h["title"] for h in hits])
    assert llm.chat("x", "y") is None   # offline

    # questions about the planned trip get the trip's city
    orch = TravelOrchestrator(PreferenceModel())
    start = date.today() + timedelta(days=20)
    plan = orch.plan_trip(PROFILE, "Delhi", "Jaipur", start, start + timedelta(days=2), 30000, [])
    assert "jaipur" in rag.search_query("How many days are enough for this trip?", plan).lower()
    ans, _ = rag.answer("How many days are enough for this trip?", plan=plan)
    assert "full days" in ans   # city overview used in the offline answer
    brief = trip_brief(plan)
    assert "Delhi to Jaipur" in brief and "28000" not in brief   # no profile data


if __name__ == "__main__":
    test_models()
    test_trips()
    test_rag()
    print("ALL TESTS PASSED")
