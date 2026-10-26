"""Day-by-day itinerary (greedy).

Each day has a time budget (8 h, 4 h on arrival / departure day) and the trip has an entry-fee
budget per person from the budget agent. For each day we take the highest-scoring attractions
that still fit the hours and money and are not closed that weekday, then order them by best
time of day and assign clock times.
"""
from datetime import timedelta

TIME_ORDER = {"morning": 0, "all": 1, "anytime": 1, "afternoon": 2, "evening": 3, "night": 4}
TRAVEL_BUFFER_H = 0.5   # time between two stops


def _fmt(h):
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}"


def _closed_on(weekly_off: str, day_name: str) -> bool:
    return day_name.lower() in str(weekly_off).lower()


def build_itinerary(attractions, start_date, num_days, fee_budget_per_person=float("inf"),
                    daily_hours=8.0):
    # attractions = AttractionAgent output (needs a 'score' column)
    pool = attractions.sort_values("score", ascending=False).to_dict("records")
    used, fee_left, days = set(), fee_budget_per_person, []

    for d in range(num_days):
        date = start_date + timedelta(days=d)
        day_name = date.strftime("%A")
        hours_left = daily_hours / 2 if (num_days > 1 and d in (0, num_days - 1)) else daily_hours
        chosen = []
        for a in pool:
            if a["name"] in used or _closed_on(a["weekly_off"], day_name):
                continue
            need = a["hours"] + (TRAVEL_BUFFER_H if chosen else 0)
            if need <= hours_left and a["fee"] <= fee_left:
                chosen.append(a); used.add(a["name"])
                hours_left -= need; fee_left -= a["fee"]
        chosen.sort(key=lambda a: TIME_ORDER.get(str(a["best_time"]).split()[0].lower(), 1))

        # arrival day starts in the afternoon
        clock = 14.0 if (num_days > 1 and d == 0) else 9.0
        stops = []
        for a in chosen:
            stops.append({"time": _fmt(clock), "end": _fmt(clock + a["hours"]), "name": a["name"],
                          "type": a["type"], "hours": a["hours"], "fee": a["fee"], "rating": a["rating"],
                          "best_time": a["best_time"]})
            clock += a["hours"] + TRAVEL_BUFFER_H
        label = "Arrival day" if (num_days > 1 and d == 0) else (
            "Departure day" if (num_days > 1 and d == num_days - 1) else "Full day")
        days.append({"day": d + 1, "date": date.strftime("%a %d %b %Y"), "label": label, "stops": stops,
                     "hours": round(sum(s["hours"] for s in stops), 1),
                     "fee_per_person": int(sum(s["fee"] for s in stops))})
    return days


def itinerary_stats(days):
    stops = [s for d in days for s in d["stops"]]
    return {
        "attractions": len(stops),
        "sightseeing_hours": round(sum(s["hours"] for s in stops), 1),
        "fees_per_person": int(sum(s["fee"] for s in stops)),
        "avg_rating": round(sum(s["rating"] for s in stops) / len(stops), 2) if stops else 0,
    }
