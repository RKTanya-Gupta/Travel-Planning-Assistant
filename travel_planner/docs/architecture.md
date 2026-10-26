# Architecture

```
Streamlit app (app.py)
  sidebar: trip details + traveller profile
        |
        v
TravelOrchestrator (src/agents.py)
  1. Preference model    predicts the package tier (Basic ... King)
                         tier decides hotel stars, flight class, train class, daily spend
  2. Transport agent     flight / train / bus / cab options for the round trip
  3. Hotel agent         hotels in the city within the tier's star range, best value
  4. Budget agent        transport + hotel + daily spend vs budget
                         -> cheaper hotel -> cheapest transport -> lower daily spend
                         -> money left = entry fee allowance
  5. Attraction agent    score = rating + popularity + interest match
  6. Optimiser           greedy day plan (hours per day, fees, weekly off, best time)
  7. Summary             budget breakdown, MAAS LLM writes a short trip summary

RAG (src/rag.py) - Ask tab
  documents : attractions, hotels, data/knowledge_base/*.md
  retrieval : one TF-IDF index per source, top 4 chunks
  answer    : MAAS LLM, only from the retrieved chunks, with [n] citations
```

## Transport agent

1. City coordinates come from the Goibibo hotel data (median hotel location per city).
2. **Flight.** Each city is mapped to its nearest airport (itself if it has one, otherwise the
   nearest airport within 300 km by road).
   - Both airports are metros: real fares from the Kaggle flight data for the same booking lead
     time. The agent shows the cheapest, fastest and best value flight each way.
   - Otherwise: fare = non-stop metro fare for that lead time, scaled by distance. Flight time
     comes from a line fitted on non-stop metro flights (hours = 0.68 + 0.0012 x km).
   - Road legs to/from the airport are added (cab for Deluxe and above, bus otherwise).
3. **Train / bus / cab.** Straight-line distance x 1.2 (rail) or 1.3 (road), times the per-km
   rate. Train class follows the tier (SL, 3A, 2A, 1A). Bus up to 700 km, cab up to 500 km.
4. The option with the lowest `fare + time value x travel hours` wins. Time value grows with
   the tier (Rs 60 per hour for Basic up to Rs 1,000 for King), so a Basic traveller is sent by
   sleeper train and a King traveller flies.

## RAG

A single TF-IDF index over everything did not work. The ~4,000 hotel rows outnumber the policy
notes about 200 to 1, and words like "check-in" or "ID" occur in most hotel descriptions, so they
got almost no weight. "Which forts are in Jaipur?" returned four Jaipur hotels. Now:

- each source (attractions, hotels, knowledge base) has its own index
- if the question names a city, attraction and hotel results are limited to that city, and the
  city name is removed from the query for those two indexes
- hotel rows are only searched when the question is about hotels or stays
- plurals are stemmed ("forts" -> "fort")
- a general question about a city ("places to visit in Jaipur") returns that city's attractions,
  best rated first
- every column of the attractions dataset is in the attraction text: type, significance, rating,
  number of reviews, fee, time needed, best time, weekly off, establishment year, airport within
  50 km and DSLR rule

What the LLM gets for each question:

- **[Trip]** a summary of the current plan (route, dates, transport, hotel, day-by-day stops,
  places left out). Only trip details, never the traveller profile (age, income, gender ...).
- **[City]** an overview of the city from the attractions data: number of places, total hours
  needed, how many 8-hour days that is, top-rated places.
- **[1]..[4]** the retrieved documents.
- the last few chat turns, so follow-up questions work. Follow-ups are first rewritten by the LLM
  into a standalone search query ("how many days for this?" -> "how many days to visit Jaipur").

The prompt asks the model to answer from this context and cite it. If the context is not enough
it may add general travel advice, marked as such, but no invented prices or details.

## MAAS client

`src/llm.py` uses the OpenAI SDK against the GenAI Lab LiteLLM proxy (`https://genailab.tcs.in`).
Any model from `/v1/models` can be used (MaaS guidelines), set with `MAAS_MODEL` in `.env`.

- SSL verification is off (`httpx.Client(verify=False)`), the same as the lab's
  `maas-connectivity-check.py`, because the gateway uses an internal certificate.
- GPT-5 models do not accept a custom temperature, so they get `max_completion_tokens` and
  `reasoning_effort` instead. If a model rejects a parameter, the call is retried with defaults.
- Errors are kept in `llm.last_error` and shown in the app instead of failing silently.

## Design choices

| Choice | Reason |
|---|---|
| Agents are plain classes with a `run()` / `plan()` method | easy to test and explain |
| Fixed order in the orchestrator | each agent needs the previous one's output |
| TF-IDF instead of embeddings | no model download in the lab VM, fast, fine for keyword-heavy text |
| Greedy optimiser | simple, fast, respects time / money / closing days |
| LLM is optional | all features work offline, the LLM improves the wording |
| ML target = ProductPitched | ready-made label with 5 ordered tiers |

## ML pipeline

`ColumnTransformer` (numeric: median impute + scaling, categorical: mode impute + one-hot) feeding
4 models. 5-fold CV on the training split, then accuracy / precision / recall / macro-F1 on a
stratified 20% test split. The best model by macro-F1 is refitted on all data and saved.
