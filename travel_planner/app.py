"""Travel Planning Assistant - Streamlit app.

Run:  streamlit run app.py
"""
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import config, llm
from src.agents import TravelOrchestrator, flights_to_df
from src.data_loader import load_attractions
from src.preference_model import PreferenceModel
from src.rag import RAG

ICON = config.ROOT / "assets" / "icon.png"
USER_ICON = config.ROOT / "assets" / "user.png"
AVATARS = {"assistant": str(ICON) if ICON.exists() else None,
           "user": str(USER_ICON) if USER_ICON.exists() else None}
st.set_page_config(page_title="Travel Planning Assistant", page_icon=str(ICON) if ICON.exists() else None,
                   layout="wide")


@st.cache_resource(show_spinner="Loading model and datasets...")
def get_orchestrator():
    return TravelOrchestrator(PreferenceModel())


@st.cache_resource(show_spinner="Building the search index...")
def get_rag():
    return RAG()


@st.cache_data(ttl=300, show_spinner=False)
def maas_status():
    if not llm.is_configured():
        return "Q&A assistant: offline (retrieved text only)"
    try:
        models = llm.list_models()
    except Exception as e:
        return f"Q&A assistant: connection failed - {llm.describe_error(e)[:200]}"
    if config.MAAS_MODEL not in models:
        return "Q&A assistant: model not available, check MAAS_MODEL in .env"
    return "Q&A assistant: connected"


try:
    orch = get_orchestrator()
    rag = get_rag()
except FileNotFoundError as e:
    st.error(f"Setup incomplete: {e}")
    st.info("1. Put the 4 Kaggle CSVs in `data/raw/` (see docs/dataset_sources.md)\n"
            "2. Train the models: `python -m src.train_models`\n3. Restart the app.")
    st.stop()

meta = orch.model.meta
from_cities = orch.transport.from_cities()
dest_cities = sorted(set(load_attractions()["city"]))
interest_options = sorted(set(load_attractions()["significance"]))


PRIVATE = "Prefer not to say"
LABELS = {"Occupation": "occupation", "Gender": "gender", "MaritalStatus": "marital status",
          "MonthlyIncome": "monthly income"}
hidden_acc = meta.get("hidden_field_accuracy", {})


def pick(options, preferred):
    return options.index(preferred) if preferred in options else 0


def privacy_note():
    full = meta.get("test_accuracy")
    if not full or not hidden_acc:
        return ("Fields set to 'Prefer not to say' are filled with typical values, "
                "so the package tier prediction becomes less reliable.")
    small = ", ".join(f"{LABELS[f]} {hidden_acc[f]:.0%}" for f in ["Occupation", "Gender", "MaritalStatus"]
                      if f in hidden_acc)
    return (f"'Prefer not to say' fields are filled with typical values. Tier prediction accuracy is "
            f"{full:.0%} with all fields. Hiding one field gives: {small}. Hiding income has the "
            f"biggest effect ({hidden_acc.get('MonthlyIncome', 0):.0%}), because income is the main "
            "signal in the data; you can then pick the package tier yourself below.")


# sidebar
with st.sidebar:
    st.header("Plan a trip")
    with st.form("trip"):
        st.subheader("Trip")
        source = st.selectbox("From", from_cities, index=pick(from_cities, "Delhi"))
        destination = st.selectbox("To", dest_cities, index=pick(dest_cities, "Jaipur"))
        start = st.date_input("Start date", date.today() + timedelta(days=21))
        end = st.date_input("End date", date.today() + timedelta(days=24))
        budget = st.number_input("Total budget (₹)", 5000, 2_000_000, 60000, step=5000)
        interests = st.multiselect("Interests", interest_options, default=interest_options[:2])

        st.subheader("Traveller profile")
        st.caption("Used only for the package tier prediction. It stays on this machine and is "
                   "never sent to the Q&A assistant.")
        age = st.number_input("Age", 18, 80, 32)
        income = st.number_input("Monthly income (₹)", 5000, 500000, 25000, step=1000)
        hide_income = st.checkbox(f"{PRIVATE} (income)", False)
        c1, c2 = st.columns(2)
        adults = c1.number_input("Adults", 1, 6, 2)
        kids = c2.number_input("Children", 0, 4, 0)
        occupation = st.selectbox("Occupation", meta["categories"]["Occupation"] + [PRIVATE])
        gender = st.selectbox("Gender", meta["categories"]["Gender"] + [PRIVATE])
        marital = st.selectbox("Marital status", meta["categories"]["MaritalStatus"] + [PRIVATE])
        c1, c2 = st.columns(2)
        city_tier = c1.selectbox("Home city tier", [1, 2, 3])
        star = c2.selectbox("Preferred hotel stars", [3, 4, 5])
        trips = st.number_input("Trips per year", 0, 20, 2)
        passport = st.checkbox("Has passport", True)
        own_car = st.checkbox("Owns a car", False)
        st.caption(privacy_note())
        override = st.selectbox("Package tier", ["Use ML prediction"] + config.TIER_ORDER)
        submitted = st.form_submit_button("Generate plan", type="primary")
    st.caption(maas_status())

if submitted:
    if end < start:
        st.sidebar.error("End date must be after the start date.")
    else:
        profile = {"Age": age, "CityTier": city_tier, "NumberOfPersonVisiting": adults,
                   "PreferredPropertyStar": star, "NumberOfTrips": trips, "Passport": int(passport),
                   "OwnCar": int(own_car), "NumberOfChildrenVisiting": kids,
                   "MonthlyIncome": None if hide_income else income,
                   "Occupation": None if occupation == PRIVATE else occupation,
                   "Gender": None if gender == PRIVATE else gender,
                   "MaritalStatus": None if marital == PRIVATE else marital}
        st.session_state.hidden = [f for f in config.OPTIONAL_FIELDS if profile[f] is None]
        st.session_state.tier_from_ml = override == "Use ML prediction"
        with st.spinner("Planning your trip..."):
            st.session_state.plan = orch.plan_trip(
                profile, source, destination, start, end, budget, interests,
                tier_override=None if override == "Use ML prediction" else override)
        st.session_state.page = "Plan"

st.title("Travel Planning Assistant")
st.caption("Preference prediction, transport / hotel / attraction / budget agents, Q&A over travel guides "
           "and an optimised day-by-day itinerary.")
# st.tabs always jumps back to the first tab on a rerun (e.g. after asking a question),
# so the selected page is kept in session_state instead
PAGES = ["Plan", "Itinerary", "Ask", "Model report"]
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "Plan"
page = st.radio("Page", PAGES, key="page", horizontal=True, label_visibility="collapsed")
st.divider()

plan = st.session_state.get("plan")

if page == "Plan":
    if not plan:
        st.info("Fill in the form on the left and click **Generate plan**.")
    else:
        b = plan["budget"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Package tier", plan["tier"])
        m2.metric("Trip length", f"{plan['days']} days, {plan['travelers']} pax")
        m3.metric("Estimated total", f"₹{b['total']:,}")
        m4.metric("Remaining budget", f"₹{b['remaining']:,}", delta_color="normal" if b["within_budget"] else "inverse",
                  delta="within budget" if b["within_budget"] else "over budget")

        hidden = st.session_state.get("hidden", [])
        if hidden and st.session_state.get("tier_from_ml", True):
            accs = [hidden_acc[f] for f in hidden if f in hidden_acc]
            if len(hidden) == len(config.OPTIONAL_FIELDS) and "all" in hidden_acc:
                accs = [hidden_acc["all"]]
            est = f" Expected accuracy is about {min(accs):.0%}" + (" or lower" if len(hidden) > 1 and len(accs) > 1 else "") \
                + f" instead of {meta['test_accuracy']:.0%}." if accs and meta.get("test_accuracy") else ""
            st.info(f"Tier predicted without your {', '.join(LABELS[f] for f in hidden)}.{est} "
                    "If it does not look right, choose the package tier in the sidebar.")

        with st.expander("Preference model probabilities"):
            probs = pd.Series(plan["probs"]).reindex(config.TIER_ORDER).dropna()
            st.bar_chart(probs)

        st.subheader("Transport")
        t = plan["transport"]
        if t.get("options"):
            st.write(f"**{t['mode']}** - {t['route']} - ₹{t['total']:,} round trip")
            odf = pd.DataFrame(t["options"])[["mode", "route", "hours", "fare_pp", "total"]]
            st.dataframe(odf.rename(columns={"hours": "hours (one way)", "fare_pp": "₹ / person (return)",
                                             "total": "₹ total"}), hide_index=True)
        else:
            st.write(f"**{t['mode']}** - ₹{t['total']:,}")
        st.caption(t["note"])
        if t.get("outbound"):
            c1, c2 = st.columns(2)
            c1.markdown("**Outbound flights**")
            c1.dataframe(flights_to_df(t["outbound"]), hide_index=True)
            c2.markdown("**Return flights**")
            c2.dataframe(flights_to_df(t["return"]), hide_index=True)

        st.subheader("Hotel")
        h = plan["hotel"]
        st.caption(h["note"] + f" {h['rooms']} room(s) x {h['nights']} night(s).")
        hdf = pd.DataFrame(h["options"])
        cols = [c for c in ["name", "locality", "stars", "rating", "price", "cost"] if c in hdf]
        st.dataframe(hdf[cols].rename(columns={"price": "₹ / night", "cost": "stay total (₹)"}),
                     hide_index=True)
        st.success(f"Selected: **{plan['hotel_choice']['name']}** - ₹{int(plan['hotel_choice']['cost']):,} for the stay")

        st.subheader("Attractions")
        a = plan["attractions"]
        if a.empty:
            st.warning("No attractions found for this city in the dataset.")
        else:
            st.dataframe(a[["name", "type", "significance", "rating", "fee", "hours", "best_time", "weekly_off",
                            "interest_match"]].head(15), hide_index=True)

        st.subheader("Budget")
        c1, c2 = st.columns([2, 1])
        c1.bar_chart(pd.Series(b["breakdown"], name="₹"))
        c2.dataframe(pd.DataFrame({"₹": b["breakdown"]}))
        for adj in b["adjustments"]:
            c2.info(adj)
        if not b["within_budget"]:
            st.error(f"Even after adjustments the trip costs ₹{-b['remaining']:,} more than the budget. "
                     "Try fewer days, a nearer destination or a higher budget.")

if page == "Itinerary":
    if not plan:
        st.info("Generate a plan first.")
    else:
        st.markdown(plan["narrative"])
        s = plan["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attractions", s["attractions"]); c2.metric("Sightseeing hours", s["sightseeing_hours"])
        c3.metric("Fees / person", f"₹{s['fees_per_person']:,}"); c4.metric("Avg rating", s["avg_rating"])
        for d in plan["itinerary"]:
            with st.expander(f"Day {d['day']} - {d['date']} - {d['label']} ({d['hours']} h)", expanded=True):
                if d["stops"]:
                    st.dataframe(pd.DataFrame(d["stops"])[["time", "end", "name", "type", "hours", "fee", "rating"]],
                                 hide_index=True)
                else:
                    st.write("Free time / rest / travel.")
        rows = [dict(day=d["day"], date=d["date"], **s) for d in plan["itinerary"] for s in d["stops"]]
        if rows:
            st.download_button("Download itinerary (CSV)", pd.DataFrame(rows).to_csv(index=False),
                               file_name="itinerary.csv", mime="text/csv")

if page == "Ask":
    if plan:
        st.write(f"Ask about your {plan['source']} - {plan['destination']} trip, attractions, hotels, "
                 "hotel policies or travel advisories.")
    else:
        st.write("Ask about attractions, hotels, hotel policies or travel advisories. "
                 "Generate a plan first to ask about your own trip.")
    st.caption("For example: How many days are enough for this trip? / What is the cancellation policy? / "
               "Is it safe to visit Goa in monsoon? / Which forts are in Jaipur?")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    _, right = st.columns([6, 1])
    if right.button("Clear chat"):
        st.session_state.chat = []

    for role, msg, src in st.session_state.chat:
        with st.chat_message(role, avatar=AVATARS[role]):
            st.markdown(msg)
            if src:
                st.caption("Sources: " + "; ".join(src))
    q = st.chat_input("Your question")
    if q:
        st.chat_message("user", avatar=AVATARS["user"]).markdown(q)
        history = [(r, m) for r, m, _ in st.session_state.chat]
        with st.spinner("Searching..."):
            ans, hits = rag.answer(q, plan=plan, history=history)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", ans)}
        src = [f"[{i}] {h['source']} - {h['title']}" for i, h in enumerate(hits, 1) if not cited or i in cited]
        if "[Trip]" in ans:
            src.insert(0, "[Trip] your trip plan")
        if "[City]" in ans:
            src.insert(0, "[City] attractions dataset")
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(ans)
            if src:
                st.caption("Sources: " + "; ".join(src))
            if llm.is_configured() and llm.last_error:
                st.caption(f"LLM call failed, showing the retrieved text instead. {llm.last_error[:300]}")
        st.session_state.chat += [("user", q, []), ("assistant", ans, src)]

if page == "Model report":
    st.write(f"Target: **{config.TARGET}** (customer's package tier). Best model: "
             f"**{meta['best_model']}**, trained on {meta['train_rows']:,} customers.")
    comp = config.REPORT_DIR / "model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp), hide_index=True)
    c1, c2 = st.columns(2)
    for col, img, cap in [(c1, "model_comparison.png", "Model comparison"),
                          (c2, "confusion_matrix.png", "Confusion matrix (best model)"),
                          (c1, "feature_importance.png", "Feature importance")]:
        p = config.REPORT_DIR / img
        if p.exists():
            col.image(str(p), caption=cap)
