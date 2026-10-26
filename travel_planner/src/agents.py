"""Planning agents and the orchestrator.

Order: preference model -> transport -> hotel -> budget -> attractions
       -> itinerary optimizer -> budget check -> LLM summary
"""
import json
import math
from datetime import date

import numpy as np
import pandas as pd

from src import config, llm
from src.data_loader import canon, city_match, load_attractions, load_city_coords, load_flights, load_hotels
from src.optimizer import build_itinerary, itinerary_stats

NO_SURFACE_LINK = {"port blair"}   # islands: only air / sea


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


# --------------------------------------------------------------------- Transport
class TransportAgent:
    """Compares flight, train, bus and cab for the round trip.

    Flights between the 6 metros use the Kaggle fares. Other airport cities get a fare
    estimated from those fares (same lead time, scaled by distance). Cities without an
    airport reach the nearest one by road. Train / bus / cab fares are distance x rate.
    The chosen option minimises  fare + time_value(tier) x travel hours.
    """

    def __init__(self):
        self.flights = load_flights()
        self.coords = load_city_coords()
        self.hubs = sorted(self.flights["source_city"].unique())
        self.airports = self.hubs + [c for c in config.AIRPORT_CITIES if self._xy(c)]
        km = {}
        for s_, d_ in self.flights[["source_city", "destination_city"]].drop_duplicates().itertuples(index=False):
            if self._xy(s_) and self._xy(d_):
                km[(s_, d_)] = haversine_km(self._xy(s_), self._xy(d_))
        self.flights["km"] = [km.get(k) for k in zip(self.flights["source_city"], self.flights["destination_city"])]
        nonstop = self.flights[self.flights["stops"] == "zero"].dropna(subset=["km"]).groupby("km")["duration"].median()
        # flight hours = a*km + b, fitted on non-stop metro flights
        self.dur_fit = np.polyfit(nonstop.index, nonstop.values, 1) if len(nonstop) > 1 else (0.0012, 0.7)
        self._fare_fits = {}

    def from_cities(self, min_hotels=5):
        c = self.coords[self.coords["hotels"] >= min_hotels]
        return sorted(set(self.hubs) | set(c["city"]))

    def _xy(self, city):
        k = canon(city)
        if k in self.coords.index:
            r = self.coords.loc[k]
            return float(r["lat"]), float(r["lon"])
        return None

    def _is_hub(self, city):
        return any(canon(h) == canon(city) for h in self.hubs)

    def _nearest_airport(self, city):
        """(airport city, road km to it); road km is 0 if the city has its own airport."""
        for ap in self.airports:
            if canon(ap) == canon(city):
                return ap, 0.0
        p = self._xy(city)
        if p is None:
            return None, None
        dists = [(haversine_km(p, self._xy(a)), a) for a in self.airports if self._xy(a)]
        if not dists:
            return None, None
        km, ap = min(dists)
        return ap, km * config.ROAD_DETOUR

    def _estimated_fare(self, km, flight_class, days_left):
        key = (flight_class, days_left)
        if key not in self._fare_fits:
            df = self.flights[(self.flights["class"] == flight_class) & (self.flights["stops"] == "zero")
                              & ((self.flights["days_left"] - days_left).abs() <= 3)]
            m = df.dropna(subset=["km"]).groupby("km")["price"].median()   # non-stop fare per metro route
            if len(m) < 2:
                m = self.flights.dropna(subset=["km"]).groupby("km")["price"].median()
            slope, icpt = np.polyfit(m.index, m.values, 1)
            if slope < 0:
                slope, icpt = 0.0, float(m.median())
            self._fare_fits[key] = (slope, icpt)
        slope, icpt = self._fare_fits[key]
        return int(round(icpt + slope * km, -1))

    def _flight_picks(self, src, dst, days_left, flight_class, travelers):
        df = self.flights[city_match(self.flights["source_city"], src)
                          & city_match(self.flights["destination_city"], dst)]
        if df.empty:
            return []
        by_class = df[df["class"].str.lower() == flight_class.lower()]
        df = by_class if not by_class.empty else df
        near = df[(df["days_left"] - days_left).abs() <= 3]      # fares change with booking lead time
        df = near if not near.empty else df
        g = (df.groupby(["airline", "flight", "stops", "departure_time", "arrival_time", "class"], as_index=False)
               .agg(price=("price", "median"), duration=("duration", "median")))
        p = (g["price"] - g["price"].min()) / (g["price"].max() - g["price"].min() + 1e-9)
        t = (g["duration"] - g["duration"].min()) / (g["duration"].max() - g["duration"].min() + 1e-9)
        g["value_score"] = 0.7 * p + 0.3 * t
        picks = []
        for label, row in [("Cheapest", g.nsmallest(1, "price")), ("Fastest", g.nsmallest(1, "duration")),
                           ("Best value", g.nsmallest(1, "value_score"))]:
            r = row.iloc[0].to_dict()
            r.update(label=label, price=int(r["price"]), total=int(r["price"]) * travelers,
                     duration=round(float(r["duration"]), 2))
            picks.append(r)
        return picks

    def _ground(self, km, travelers, tier):
        # one-way options for a straight-line distance
        tp = config.TIER_PROFILE[tier]
        rail_km, road_km = km * config.RAIL_DETOUR, km * config.ROAD_DETOUR
        opts = []
        if km >= 60:
            cls = tp["rail_class"]
            per_km, fixed = config.RAIL_FARE[cls]
            fare = round(per_km * rail_km + fixed)
            opts.append({"mode": f"Train ({cls})", "fare_pp": fare, "cost": fare * travelers,
                         "hours": rail_km / config.RAIL_KMPH, "km": rail_km})
        if road_km <= config.MAX_BUS_KM:
            fare = round(config.BUS_FARE_PER_KM * road_km)
            opts.append({"mode": "AC bus", "fare_pp": fare, "cost": fare * travelers,
                         "hours": road_km / config.BUS_KMPH, "km": road_km})
        if road_km <= config.MAX_CAB_KM:
            cabs = math.ceil(travelers / config.CAB_SEATS)
            cost = round(cabs * (config.CAB_FARE_PER_KM * road_km + config.CAB_DRIVER_ALLOWANCE))
            opts.append({"mode": "Cab", "fare_pp": round(cost / travelers), "cost": cost,
                         "hours": road_km / config.CAB_KMPH, "km": road_km})
        return opts

    def _transfer(self, road_km, travelers, tier):
        # road leg to / from the airport: cab for Deluxe and above or short hops, else bus
        if not road_km:
            return {"mode": "", "cost": 0, "hours": 0.0}
        opts = {o["mode"]: o for o in self._ground(road_km / config.ROAD_DETOUR, travelers, tier)}
        if "Cab" in opts and (config.TIER_ORDER.index(tier) >= 2 or road_km < 60):
            return opts["Cab"]
        return opts.get("AC bus") or opts["Cab"]

    def _flight_option(self, source, destination, days_left, ret_days_left, travelers, tier):
        cls = config.TIER_PROFILE[tier]["flight_class"]
        a1, k1 = self._nearest_airport(source)
        a2, k2 = self._nearest_airport(destination)
        if (not a1 or not a2 or canon(a1) == canon(a2)
                or max(k1, k2) > config.MAX_AIRPORT_TRANSFER_KM):
            return None, [], []
        both_hubs = self._is_hub(a1) and self._is_hub(a2)
        air_km = haversine_km(self._xy(a1), self._xy(a2)) if (self._xy(a1) and self._xy(a2)) else None
        if air_km is None and not both_hubs:
            return None, [], []
        if air_km is not None and air_km < config.MIN_FLIGHT_KM:
            return None, [], []

        out, ret = [], []
        if both_hubs:
            out = self._flight_picks(a1, a2, days_left, cls, travelers)
            ret = self._flight_picks(a2, a1, ret_days_left, cls, travelers)
        if out and ret:
            bo = next(o for o in out if o["label"] == "Best value")
            br = next(o for o in ret if o["label"] == "Best value")
            air = bo["total"] + br["total"]
            air_cheapest = min(o["total"] for o in out) + min(o["total"] for o in ret)
            air_hours = (bo["duration"] + br["duration"]) / 2
            source_tag = "Kaggle fares"
        elif air_km is None:
            return None, [], []
        else:
            fare = self._estimated_fare(air_km, cls, days_left) + self._estimated_fare(air_km, cls, ret_days_left)
            air = air_cheapest = fare * travelers
            air_hours = float(np.polyval(self.dur_fit, air_km))
            source_tag = "estimated fare"

        t1, t2 = self._transfer(k1, travelers, tier), self._transfer(k2, travelers, tier)
        road = 2 * (t1["cost"] + t2["cost"])
        legs = []
        if k1:
            legs.append(f"{t1['mode'].lower()} to {a1} ({k1:.0f} km)")
        legs.append(f"flight {a1} to {a2} ({source_tag})")
        if k2:
            legs.append(f"{t2['mode'].lower()} to {destination} ({k2:.0f} km)")
        total = air + road
        opt = {"mode": "Flight" if not (k1 or k2) else "Flight + road",
               "route": ", ".join(legs), "fare_pp": round(total / travelers), "total": int(total),
               "cheapest_total": int(air_cheapest + road),
               "hours": air_hours + config.AIRPORT_BUFFER_H + t1["hours"] + t2["hours"]}
        return opt, out, ret

    def run(self, source, destination, start, end, travelers, tier="Standard"):
        if canon(source) == canon(destination):
            return {"mode": "Local", "route": destination, "options": [], "outbound": [], "return": [],
                    "total": 0, "cheapest_total": 0, "note": "Source and destination are the same city."}

        days_left = int(np.clip((start - date.today()).days, 1, 49))
        ret_days_left = int(np.clip(days_left + (end - start).days, 1, 49))
        options = []
        flight, out_picks, ret_picks = self._flight_option(source, destination, days_left, ret_days_left,
                                                           travelers, tier)
        if flight:
            options.append(flight)

        a, b = self._xy(source), self._xy(destination)
        km = haversine_km(a, b) if (a and b) else None
        island = canon(source) in NO_SURFACE_LINK or canon(destination) in NO_SURFACE_LINK
        if km is not None and not island:
            for o in self._ground(km, travelers, tier):
                options.append({"mode": o["mode"], "route": f"{source} to {destination}, about {o['km']:.0f} km",
                                "fare_pp": 2 * o["fare_pp"], "total": int(2 * o["cost"]),
                                "cheapest_total": int(2 * o["cost"]), "hours": o["hours"]})

        if not options:
            est = config.GROUND_TRANSPORT_EST_PER_PERSON_ONE_WAY * 2 * travelers
            missing = source if a is None else destination
            return {"mode": "Estimate", "route": f"{source} to {destination}", "options": [],
                    "outbound": [], "return": [], "total": est, "cheapest_total": est, "distance_km": None,
                    "note": (f"No location found for {missing} in the hotel data, so a flat "
                             f"₹{config.GROUND_TRANSPORT_EST_PER_PERSON_ONE_WAY:,} per person each way is used.")}

        time_value = config.TIER_PROFILE[tier]["time_value"]
        for o in options:
            o["hours"] = round(o["hours"], 1)
            o["score"] = o["total"] + time_value * o["hours"] * 2 * travelers
        options.sort(key=lambda o: o["score"])
        chosen = options[0]
        with_kaggle_flights = chosen["mode"].startswith("Flight") and bool(out_picks)
        note = (f"Round trip for {travelers}, booked ~{days_left} days ahead. Metro-to-metro flight fares come "
                "from the Kaggle dataset; other flight fares and train / bus / cab fares are estimates from "
                f"distance. Best option for the {tier} tier = lowest fare + value of travel time.")
        return {"mode": chosen["mode"], "route": chosen["route"], "options": options, "chosen": chosen,
                "outbound": out_picks if with_kaggle_flights else [],
                "return": ret_picks if with_kaggle_flights else [],
                "total": chosen["total"], "cheapest_total": min(o["cheapest_total"] for o in options),
                "distance_km": round(km) if km else None, "note": note}


# ------------------------------------------------------------------------- Hotel
class HotelAgent:
    """Shortlists hotels in the destination city that match the predicted tier."""

    def __init__(self):
        self.hotels = load_hotels()

    def run(self, city, tier, nights, travelers):
        nights = max(nights, 1)
        rooms = math.ceil(travelers / 2)
        lo, hi = config.TIER_PROFILE[tier]["hotel_stars"]
        in_city = self.hotels[city_match(self.hotels["city"], city)]
        match = in_city[in_city["stars"].between(lo, hi)]
        stars = f"{lo}-star" if lo == hi else f"{lo}-{hi} star"
        note = f"{len(match)} hotels in {city} match {stars} for the {tier} tier."
        if match.empty and not in_city.empty:
            match, note = in_city, f"No {stars} hotels in {city}, showing all {len(in_city)} hotels there."
        if match.empty:
            star = hi
            price = config.EST_PRICE_BY_STAR[star]
            fake = {"name": f"{star}-star hotel (estimate)", "city": city, "stars": star, "rating": np.nan,
                    "price": price, "price_estimated": True, "locality": "", "facilities": "",
                    "cost": price * nights * rooms}
            return {"options": [fake], "chosen": fake, "rooms": rooms, "nights": nights,
                    "note": f"No hotels for {city} in the dataset, using a {star}-star price estimate."}
        df = match.copy()
        # value = good rating for the money
        df["value"] = df["rating"].fillna(3.5) / np.log1p(df["price"])
        df["cost"] = (df["price"] * nights * rooms).astype(int)
        options = df.sort_values(["value", "rating"], ascending=False).head(6)
        cheaper = in_city.assign(cost=(in_city["price"] * nights * rooms).astype(int)).nsmallest(3, "price")
        opts = options.to_dict("records")
        return {"options": opts, "chosen": opts[0], "cheaper": cheaper.to_dict("records"),
                "rooms": rooms, "nights": nights,
                "note": note + (" Prices estimated from star rating (dataset has no tariff column)."
                                if bool(df["price_estimated"].any()) else "")}


# -------------------------------------------------------------------- Attraction
class AttractionAgent:
    """Ranks the destination's attractions by rating, popularity and the user's interests."""

    def __init__(self):
        self.places = load_attractions()

    def run(self, city, interests):
        df = self.places[city_match(self.places["city"], city)].copy()
        if df.empty:
            return df
        wanted = [i.lower() for i in interests]
        text = (df["significance"] + " " + df["type"]).str.lower()
        df["interest_match"] = text.apply(lambda s: any(w in s for w in wanted)) if wanted else False
        df["score"] = (df["rating"] + 0.3 * np.log1p(df["reviews_lakhs"] * 10)
                       + 1.0 * df["interest_match"].astype(float))
        return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Budget planner
class BudgetPlannerAgent:
    """Fits transport + hotel + daily costs into the budget and leaves an allowance for attractions."""

    def plan(self, budget, transport, hotel, tier, days, travelers):
        misc_rate = config.TIER_PROFILE[tier]["daily_misc"]
        steps = []
        t_cost = transport["total"]
        h_choice = hotel["chosen"]
        misc = misc_rate * days * travelers

        def fixed():
            return t_cost + h_choice["cost"] + misc

        if fixed() > budget and hotel.get("cheaper"):
            cheapest = min(hotel["cheaper"], key=lambda h: h["cost"])
            if cheapest["cost"] < h_choice["cost"]:
                steps.append(f"Hotel downgraded to {cheapest['name']} (saves ₹{h_choice['cost'] - cheapest['cost']:,}).")
                h_choice = cheapest
        switch = None
        if fixed() > budget and transport.get("cheapest_total", t_cost) < t_cost:
            saved = t_cost - transport["cheapest_total"]
            cheap = min(transport.get("options") or [{}], key=lambda o: o.get("cheapest_total", t_cost))
            if cheap and cheap is not transport.get("chosen"):
                steps.append(f"Switched to {cheap['mode']} to save money (saves ₹{saved:,}).")
                switch = {"mode": cheap["mode"], "route": cheap["route"], "chosen": cheap,
                          "total": cheap["cheapest_total"]}
                if not cheap["mode"].startswith("Flight"):
                    switch.update({"outbound": [], "return": []})
            else:
                steps.append(f"Switched to the cheapest flights (saves ₹{saved:,}).")
            t_cost = transport["cheapest_total"]
        basic = config.TIER_PROFILE["Basic"]["daily_misc"]
        if fixed() > budget and misc_rate > basic:
            steps.append(f"Daily spend reduced to the Basic level ₹{basic}/person/day.")
            misc_rate, misc = basic, basic * days * travelers

        allowance = max(budget - fixed(), 0)
        return {"transport": int(t_cost), "transport_switch": switch,
                "hotel": int(h_choice["cost"]), "hotel_choice": h_choice,
                "daily_misc": int(misc), "daily_misc_rate": misc_rate,
                "activity_allowance_per_person": allowance / travelers, "adjustments": steps}

    def summary(self, budget, plan, fees_per_person, travelers):
        activities = int(fees_per_person * travelers)
        total = plan["transport"] + plan["hotel"] + plan["daily_misc"] + activities
        breakdown = {"Transport": plan["transport"], "Hotel": plan["hotel"],
                     "Food & local travel": plan["daily_misc"], "Attraction fees": activities}
        return {"breakdown": breakdown, "total": total, "budget": budget,
                "remaining": budget - total, "within_budget": total <= budget,
                "adjustments": plan["adjustments"]}


# ------------------------------------------------------------------ Orchestrator
class TravelOrchestrator:
    def __init__(self, preference_model):
        self.model = preference_model
        self.transport = TransportAgent()
        self.hotel = HotelAgent()
        self.attraction = AttractionAgent()
        self.budget = BudgetPlannerAgent()

    def plan_trip(self, profile, source, destination, start, end, budget, interests, tier_override=None):
        days = (end - start).days + 1
        travelers = int(profile["NumberOfPersonVisiting"]) + int(profile.get("NumberOfChildrenVisiting", 0))
        tier, probs = self.model.predict(profile)
        tier = tier_override or tier

        transport = self.transport.run(source, destination, start, end, travelers, tier)
        hotel = self.hotel.run(destination, tier, nights=days - 1, travelers=travelers)
        bplan = self.budget.plan(budget, transport, hotel, tier, days, travelers)
        if bplan["transport_switch"]:
            transport = {**transport, **bplan["transport_switch"]}
        places = self.attraction.run(destination, interests)
        itinerary = build_itinerary(places, start, days, bplan["activity_allowance_per_person"],
                                    config.DAILY_HOURS) if not places.empty else []
        stats = itinerary_stats(itinerary)
        bsum = self.budget.summary(budget, bplan, stats["fees_per_person"], travelers)

        result = {"tier": tier, "probs": probs, "days": days, "travelers": travelers,
                  "source": source, "destination": destination, "start": start, "end": end,
                  "transport": transport, "hotel": hotel, "hotel_choice": bplan["hotel_choice"],
                  "attractions": places, "itinerary": itinerary, "stats": stats, "budget": bsum,
                  "interests": interests}
        result["narrative"] = self.narrate(result)
        return result

    @staticmethod
    def narrate(r):
        # LLM summary, or a plain template if MAAS is not available
        brief = {
            "trip": f"{r['source']} to {r['destination']}, {r['days']} days, {r['travelers']} travelers",
            "preference_tier": r["tier"], "interests": r["interests"],
            "hotel": r["hotel_choice"]["name"], "transport": r["transport"]["mode"],
            "transport_route": r["transport"].get("route", ""),
            "itinerary": [{"day": d["day"], "date": d["date"], "stops": [s["name"] for s in d["stops"]]}
                          for d in r["itinerary"]],
            "budget": r["budget"]["breakdown"], "total": r["budget"]["total"], "budget_limit": r["budget"]["budget"],
        }
        text = llm.chat(
            system=("You are a friendly travel planner. Write a short, warm trip summary (max 200 words) "
                    "from the JSON plan. Do not invent places, prices or facts not in the JSON."),
            user=json.dumps(brief, default=str))
        if text:
            return text
        lines = [f"**Your {r['days']}-day {r['tier']} trip to {r['destination']}** for {r['travelers']} "
                 f"traveller(s), staying at **{r['hotel_choice']['name']}**. Getting there: {r['transport']['mode']}"
                 + (f" ({r['transport']['route']})." if r["transport"].get("route") else ".")]
        for d in r["itinerary"]:
            names = ", ".join(s["name"] for s in d["stops"]) or "Free time / rest"
            lines.append(f"- **Day {d['day']} ({d['date']})**: {names}")
        b = r["budget"]
        lines.append(f"\nEstimated total ₹{b['total']:,} of your ₹{b['budget']:,} budget "
                     f"({'within budget' if b['within_budget'] else 'over budget'}).")
        return "\n".join(lines)


def flights_to_df(options):
    if not options:
        return pd.DataFrame()
    cols = ["label", "airline", "flight", "stops", "departure_time", "arrival_time", "duration", "price", "total"]
    return pd.DataFrame(options)[cols].rename(columns={"duration": "hours", "price": "fare/person"})
