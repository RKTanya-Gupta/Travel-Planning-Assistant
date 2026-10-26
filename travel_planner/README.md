# Travel Planning Assistant

Case study 8 - create personalised travel itineraries.

The user enters a trip (from, to, dates, budget, interests) and a short profile in a Streamlit app.
The app predicts the user's package tier with an ML model, runs four agents (transport, hotel,
attractions, budget), builds a day-by-day itinerary and answers travel questions from a small
knowledge base using the MAAS LLM (RAG).

| Part | What it does |
|---|---|
| Preference model | Predicts the package tier (Basic, Standard, Deluxe, Super Deluxe, King) from age, income, family size etc. Logistic Regression, Decision Tree, Random Forest and Gradient Boosting are compared. |
| Transport agent | Compares flight, train, bus and cab for the round trip. Metro-to-metro fares come from the Kaggle flight dataset, everything else is estimated from distance. |
| Hotel agent | Hotels in the destination within the tier's star range, ranked by rating for the price. |
| Attraction agent | Attractions ranked by rating, popularity and the user's interests. |
| Budget agent | Fits transport + hotel + daily spend into the budget, downgrades if needed and sets the money left for entry fees. |
| RAG (Ask tab) | TF-IDF search over attractions, hotels, hotel policies and travel advisories. The MAAS LLM answers with citations, using the current trip plan and earlier chat turns, so "how many days are enough for this trip?" works. |
| Itinerary optimiser | Greedy day-by-day plan that respects hours per day, fee budget, weekly-off days and best time of day. |

## Setup

```bash
pip install -r requirements.txt
```

1. Put the four Kaggle CSVs in `data/raw/` (names and links in `docs/dataset_sources.md`) and
   check that all of them load:
   ```bash
   python -m src.data_loader
   ```
2. Train the models (writes `models/` and `reports/`):
   ```bash
   python -m src.train_models
   ```
3. Connect the LLM: copy `.env.example` to `.env` and paste the key from the `api_key` file in
   the MaaS folder (Desktop > Project > Instructions > MaaS). Then check the connection:
   ```bash
   python -m src.llm
   ```
   It lists the models your key can use and sends a test message. The app also works without
   the LLM; answers then show the retrieved text directly.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

### Changing the LLM model

Set `MAAS_MODEL` in `.env` to any id printed by `python -m src.llm` (the same list as `/v1/models`
in the GenAI Lab swagger), e.g.

```
MAAS_MODEL=genailab-maas-gpt-5.4
MAAS_MODEL=genailab-maas-gpt-5.4-mini
MAAS_MODEL=azure_ai/genailab-maas-DeepSeek-V3-0324
```

Restart Streamlit after editing `.env`. The sidebar shows whether the model is reachable.

## Tests

The smoke test runs on generated mock data, so it does not need the Kaggle files:

```bash
python tests/make_mock_data.py
python tests/test_pipeline.py
```

## Project layout

```
app.py                  Streamlit app (pages: Plan, Itinerary, Ask, Model report)
src/config.py           paths, file names, tier rules, fare rates, MAAS settings
src/data_loader.py      loading and cleaning of the 4 datasets, city coordinates
src/train_models.py     trains and compares the 4 models, saves the best one
src/preference_model.py loads the saved model and predicts the tier
src/agents.py           transport, hotel, attraction and budget agents + orchestrator
src/optimizer.py        itinerary optimiser
src/rag.py              retrieval + answer generation
src/llm.py              MAAS client and connection check
data/raw/               Kaggle CSVs
data/knowledge_base/    hotel policies and travel advisories (RAG documents)
notebooks/              EDA and model training
docs/                   dataset sources, architecture, sprint log, presentation outline
tests/                  mock data generator and smoke test
```

## Privacy

Occupation, gender, marital status and monthly income can be set to "Prefer not to say". The
model then fills them with typical values, so the tier prediction gets less reliable; the form shows
by how much (accuracy measured at training time and stored in `models/model_meta.json`). Hiding
income has the biggest effect. The profile is only used by the local model and is never sent to
the LLM.

## Known limitations

- The hotel dataset has no price column, so nightly prices are estimated from the star rating.
  About half of the hotels are unrated.
- The flight dataset only covers 6 metros (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai)
  and was collected in 2022. Fares for other airports are estimated from these fares.
- Train, bus and cab fares are distance x per-km rate (see `src/config.py`), not live prices.
- Hotel policies and travel advisories are short reference notes written by the team.
