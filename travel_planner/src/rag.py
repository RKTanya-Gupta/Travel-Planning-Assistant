"""RAG over tourism guides, hotel info, hotel policies and travel advisories.

Sources: guide (one doc per attraction), hotel (one doc per Goibibo hotel) and
kb (sections of data/knowledge_base/*.md). Each source has its own TF-IDF index,
see docs/architecture.md for why. The top chunks go to the MAAS LLM; without it
the chunks are shown as they are.

When a trip plan exists, a short summary of it goes to the LLM too, so questions like
"how many days are enough for this trip?" can be answered. Only trip details are sent,
never the traveller profile (age, income, ...), as per the MAAS data rules.
"""
import math
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config, llm
import pandas as pd

from src.data_loader import CITY_ALIASES, canon, load_attractions, load_hotels

HOTEL_WORDS = re.compile(r"\b(hotels?|stay(s|ing)?|rooms?|resorts?|accommodation|lodges?|homestays?|"
                         r"propert(y|ies)|guest ?houses?)\b", re.I)

SYSTEM_PROMPT = (
    "You are the assistant inside a travel planning app for trips in India.\n"
    "Answer from the context first: [Trip] is the user's current trip plan, [City] is an overview of "
    "a city's attractions from our dataset and [1], [2], ... are retrieved documents. Cite what you use.\n"
    "If the context does not fully answer the question, you may add general travel advice from your "
    "own knowledge. Say clearly that it is general advice, and do not make up prices, timings or "
    "details of specific hotels or attractions.\n"
    "If the user refers to 'this trip' and there is no [Trip] block, ask them to generate a plan "
    "first or to name the destination.\n"
    "Keep answers short and practical, use bullet points for lists."
)

REWRITE_PROMPT = (
    "Rewrite the user's last message as a short standalone search query for a travel knowledge base. "
    "Use the conversation to fill in missing details such as the city or topic. Return only the query."
)


def _stem(word):
    if len(word) > 4:
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith(("ches", "shes", "sses", "xes")):
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
    return word


def normalise(text):
    return re.sub(r"[a-z]+", lambda m: _stem(m.group()), str(text).lower())


STOP_WORDS = sorted({normalise(w) for w in ENGLISH_STOP_WORDS})


def _chunk_markdown(text, source):
    # one chunk per '## ' section
    docs = []
    for part in re.split(r"\n(?=## )", text):
        part = part.strip()
        if len(part) > 40 and part.startswith("## "):   # skip the file's intro paragraph
            title = part.splitlines()[0].lstrip("# ").strip()
            docs.append({"group": "kb", "text": part, "source": source, "title": title, "city": ""})
    return docs


def build_documents():
    docs = []
    try:
        for r in load_attractions().itertuples():
            extra = ""
            if r.year:
                extra += f" Established: {r.year}."
            if r.airport_50km:
                extra += f" Airport within 50 km: {r.airport_50km}."
            if r.dslr:
                extra += f" Photography with DSLR: {r.dslr}."
            docs.append({
                "group": "guide", "city": canon(r.city), "rating": float(r.rating),
                "source": "Tourism guide", "title": f"{r.name}, {r.city}",
                "text": (f"{r.name} is a {r.type} in {r.city}, {r.state} ({r.zone} zone). "
                         f"Significance: {r.significance}. Google rating {r.rating}/5 "
                         f"from about {r.reviews_lakhs} lakh reviews. "
                         f"Entrance fee: INR {int(r.fee)}. Time needed: about {r.hours} hours. "
                         f"Best time to visit: {r.best_time}. Weekly off: {r.weekly_off}.{extra}"),
            })
    except FileNotFoundError:
        pass
    try:
        for r in load_hotels().itertuples():
            price = f"approx. INR {int(r.price)}/night" + (" (estimated from star rating)" if r.price_estimated else "")
            stars = f"{r.stars}-star" if r.stars else "unrated"
            where = f"{r.city} ({r.locality})" if r.locality else r.city
            facilities = r.facilities.replace("|", ", ")[:300]
            docs.append({
                "group": "hotel", "city": canon(r.city),
                "source": "Hotel info", "title": f"{r.name}, {r.city}",
                "text": (f"{r.name} in {where}. {stars} hotel, guest rating {r.rating}/5, {price}. "
                         f"Facilities: {facilities}. {r.description[:400]}"),
            })
    except FileNotFoundError:
        pass
    for f in sorted(config.KB_DIR.glob("*.md")):
        docs.extend(_chunk_markdown(f.read_text(encoding="utf-8"), f.stem.replace("_", " ").title()))
    return docs


def trip_brief(plan):
    """Plain-text summary of a plan from TravelOrchestrator.plan_trip (trip details only)."""
    if not plan:
        return ""
    b, t, h = plan["budget"], plan["transport"], plan["hotel_choice"]
    lines = [
        f"{plan['source']} to {plan['destination']}, {plan['start']} to {plan['end']} ({plan['days']} days), "
        f"{plan['travelers']} traveller(s), {plan['tier']} package.",
        f"Budget ₹{b['budget']:,}, estimated total ₹{b['total']:,} "
        f"({'within' if b['within_budget'] else 'over'} budget).",
        f"Transport: {t['mode']}" + (f" ({t['route']})" if t.get("route") else "") + f", ₹{t['total']:,} return.",
        f"Hotel: {h['name']}, ₹{int(h['cost']):,} for the stay.",
        f"Interests: {', '.join(plan['interests']) or 'none given'}.",
    ]
    for d in plan["itinerary"]:
        stops = ", ".join(f"{s['name']} ({s['hours']} h)" for s in d["stops"]) or "free time"
        lines.append(f"Day {d['day']} ({d['date']}, {d['label']}): {stops}")
    places = plan["attractions"]
    if len(places):
        done = {s["name"] for d in plan["itinerary"] for s in d["stops"]}
        left = places[~places["name"].isin(done)]
        if len(left):
            lines.append("Not scheduled (no time or fee budget left): "
                         + ", ".join(f"{r.name} ({r.hours} h)" for r in left.head(10).itertuples()))
    return "\n".join(lines)


class RAG:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else build_documents()
        try:
            self.places = load_attractions()
        except FileNotFoundError:
            self.places = pd.DataFrame(columns=["city", "name", "hours", "rating"])
        self.index = {}
        for group in ("guide", "hotel", "kb"):
            ids = [i for i, d in enumerate(self.docs) if d["group"] == group]
            if not ids:
                continue
            vec = TfidfVectorizer(preprocessor=normalise, stop_words=STOP_WORDS,
                                  ngram_range=(1, 2), sublinear_tf=True)
            matrix = vec.fit_transform([self.docs[i]["title"] + ". " + self.docs[i]["text"] for i in ids])
            cities = np.array([self.docs[i]["city"] for i in ids])
            self.index[group] = (ids, vec, matrix, cities)

        known = {d["city"] for d in self.docs if d["city"]}
        names = known | {a for a, c in CITY_ALIASES.items() if c in known}
        # longest names first so "navi mumbai" wins over "mumbai"
        self._city_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True))
                                   + r")\b", re.I) if names else None

    def cities_in(self, text):
        if not self._city_re:
            return set()
        return {canon(m) for m in self._city_re.findall(text.lower())}

    def retrieve(self, query, k=4, min_score=0.05):
        cities = self.cities_in(query)
        about_hotels = bool(HOTEL_WORDS.search(query))
        hits = []
        for group, (ids, vec, matrix, doc_cities) in self.index.items():
            if group == "hotel" and not about_hotels:
                continue
            if cities and group in ("guide", "hotel"):
                # results are already limited to the city, so the city name itself
                # would only add the same score to every document
                in_city = np.isin(doc_cities, list(cities))
                sims = cosine_similarity(vec.transform([self._city_re.sub(" ", query)]), matrix)[0]
                if sims[in_city].max(initial=0) < min_score and group == "guide":
                    # general question about the city ("places to visit in Jaipur"):
                    # return its places, best rated first
                    sims = np.full(len(ids), min_score)
                sims = np.where(in_city, sims, 0)
            else:
                sims = cosine_similarity(vec.transform([query]), matrix)[0]
                if group == "kb" and cities:
                    sims = sims * 0.6   # a city is named: its attractions / hotels matter more
            order = sims
            if group == "guide":
                # equal text scores (e.g. "places to visit in Jaipur"): prefer better rated places
                order = sims + 0.002 * np.array([self.docs[i].get("rating", 0) for i in ids])
            for j in order.argsort()[::-1][:k]:
                if sims[j] >= min_score:
                    hits.append(dict(self.docs[ids[j]], score=round(float(sims[j]), 3)))
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    def city_overview(self, city):
        df = self.places[self.places["city"].map(canon) == city]
        if df.empty:
            return ""
        hours = df["hours"].sum() + 0.5 * len(df)          # + ~30 min between places
        days = math.ceil(hours / config.DAILY_HOURS)
        top = df.sort_values("rating", ascending=False).head(6)
        return (f"{df['city'].iat[0]}: {len(df)} attractions in our dataset, about {hours:.0f} hours to see "
                f"all of them including travel between places (~{days} full days at "
                f"{config.DAILY_HOURS:.0f} h/day). Highest rated: "
                + ", ".join(f"{r.name} ({r.rating}/5, {r.hours} h)" for r in top.itertuples()) + ".")

    def search_query(self, question, plan=None, history=None):
        """Standalone query for retrieval: resolves follow-ups and adds the trip's city."""
        query = question
        if history:
            if llm.is_configured():
                convo = "\n".join(f"{role}: {text[:300]}" for role, text in history[-4:])
                rewritten = llm.chat(REWRITE_PROMPT, f"Conversation:\n{convo}\n\nLast message: {question}",
                                     max_tokens=60)
                if rewritten:
                    query = rewritten.strip().strip('"')
            elif not self.cities_in(question):
                last_user = next((text for role, text in reversed(history) if role == "user"), "")
                query = f"{question} {last_user}"
        if plan and not self.cities_in(query):
            query = f"{query} {plan['destination']}"
        return query

    def answer(self, question, k=6, plan=None, history=None):
        history = [(r, t) for r, t in (history or []) if t][-6:]
        query = self.search_query(question, plan, history)
        hits = self.retrieve(query, k)
        overviews = [o for o in (self.city_overview(c) for c in sorted(self.cities_in(query))[:2]) if o]

        blocks = []
        if plan:
            blocks.append("[Trip] Current trip plan\n" + trip_brief(plan))
        blocks += [f"[City] {o}" for o in overviews]
        blocks += [f"[{i+1}] ({h['source']} - {h['title']})\n{h['text']}" for i, h in enumerate(hits)]

        reply = llm.chat(system=SYSTEM_PROMPT, max_tokens=900, history=history,
                         user="Context:\n" + ("\n\n".join(blocks) or "(nothing found)")
                              + f"\n\nQuestion: {question}")
        if reply is None:
            if not hits and not overviews:
                return "I couldn't find anything about that in the travel knowledge base.", []
            parts = [f"**City overview** - {o}" for o in overviews]
            parts += [f"**[{i+1}] {h['title']}** - {h['text'][:500]}" for i, h in enumerate(hits)]
            reply = "From the knowledge base:\n\n" + "\n\n".join(parts)
        return reply, hits
