# Sprint log

| Sprint | Goal | Output | Status | Notes |
|--------|------|--------|--------|-------|
| 1. Problem and data | Requirements, datasets, EDA | `docs/dataset_sources.md`, `notebooks/01_EDA.ipynb`, `src/data_loader.py` | Done | 4,888 customers, 5 tiers (Basic 37.7% ... King 4.7%). Age and MonthlyIncome ~4.6% missing. "Fe Male" typo (155 rows) and "Unmarried" merged into Single. Designation maps 1:1 to the tier, dropped. 48% of hotels unrated, no price column. |
| 2. ML models | Preprocessing, 4 models, save best | `notebooks/02_Model_Training.ipynb`, `src/train_models.py`, `reports/` | Done | Random Forest best: accuracy 0.882, macro-F1 0.878 (GB 0.865, DT 0.839, LR 0.700). Top features: MonthlyIncome (0.45), Age (0.15), NumberOfTrips. |
| 3. Agents and RAG | Transport, hotel, attraction, budget agents; RAG + MAAS | `src/agents.py`, `src/rag.py`, `src/llm.py`, `data/knowledge_base/` | Done | Transport compares flight / train / bus / cab. RAG uses one index per source (single index returned hotels for every question). MAAS needed `verify=False` and GPT-5 parameter handling. |
| 4. Optimiser and UI | Itinerary optimiser, Streamlit app | `src/optimizer.py`, `app.py` | Done | Hours per day, fee allowance and weekly-off days respected. |
| 5. Test and present | Tests, docs, presentation | `tests/`, `README.md`, `docs/presentation_outline.md` | In progress | Smoke test passes on mock data. |

## Test cases

| # | Input | Expected | Actual |
|---|-------|----------|--------|
| 1 | Delhi to Mumbai, 4 days, 2 adults, Standard | Kaggle flights, flight chosen | Flight, Rs 9,498 return for 2, train 3A shown as Rs 6,924 / 25 h |
| 2 | Delhi to Mumbai, Basic | Cheaper mode for low tier | Train (SL) Rs 2,676 for 2 |
| 3 | Pune to Bangalore, Deluxe (Pune not in flight data) | Estimated fare from Pune | Flight (estimated fare) Rs 9,780 for 2 |
| 4 | Delhi to Jaipur, Standard | Short trip by surface | Train (3A) Rs 1,680, bus and cab also listed |
| 5 | Delhi to Mumbai, King tier, budget Rs 60,000 | Budget agent downgrades | Hotel downgraded, business flight replaced by 1A train, total Rs 54,424 |
| 6 | Attraction with weekly off | Not scheduled on closing day | Passed (smoke test) |
| 7 | Ask: "What is the cancellation policy?" | Hotel policy chunk cited | Cancellation policy [1] |
| 8 | Ask: "Is it safe to visit Goa in monsoon?" | Advisory chunk cited | Goa in the monsoon [1] |
| 9 | Ask: "What ID do I need at check-in?" | Identity documents chunk | Identity documents at check-in [1] |
| 10 | MAAS key missing / wrong | App still works | Offline answers, error shown in sidebar |
| 11 | Ask after planning Delhi to Jaipur: "How many days are sufficient for this travel?" | Uses the plan and attraction hours | Trip plan + Jaipur overview (hours needed, days) sent to the LLM |
| 12 | Follow-up question without naming the city | Earlier turns used | Query rewritten with the city, last turns passed to the LLM |
