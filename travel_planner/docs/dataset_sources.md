# Dataset sources

All four datasets are public Kaggle datasets. Download them, unzip and put the CSV files in
`data/raw/` with these names (or change the names in `src/config.py`).

| # | Used for | Kaggle dataset | File |
|---|----------|----------------|------|
| 1 | Preference model (package tier) | Holiday Package Purchase Prediction - kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction | `Travel.csv` |
| 2 | Transport agent (flights) | Flight Price Prediction (EaseMyTrip) - kaggle.com/datasets/shubhambathwal/flight-price-prediction | `Clean_Dataset.csv` |
| 3 | Attraction agent, RAG tourism guides | Travel Dataset: Guide to India's Must-See Places - kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places | `Top Indian Places to Visit.csv` |
| 4 | Hotel agent, RAG hotel info, city locations | Hotels on Goibibo (PromptCloud) - kaggle.com/datasets/PromptCloudHQ/hotels-on-goibibo | `goibibo_com-travel_sample.csv` |

The loaders in `src/data_loader.py` match column names loosely, so small header differences
between versions of the same dataset are fine.

## What is in each dataset

1. **Travel.csv** - 4,888 customers with age, income, occupation, family size, preferred hotel
   star and the package pitched (Basic 37.7%, Deluxe 35.4%, Standard 15.2%, Super Deluxe 7.0%,
   King 4.7%). The package tier is used as the label for travel preference.
2. **Clean_Dataset.csv** - 300,153 flight fares on 30 routes between 6 metros: airline, stops,
   duration, class and days before departure. Economy fares booked 1-3 days ahead average about
   Rs 13,800 against about Rs 4,900 at 30+ days, so the agent prices flights by booking lead time.
3. **Top Indian Places to Visit.csv** - about 325 attractions with city, type, Google rating,
   entrance fee, time needed, weekly off day and best time to visit. These are the fields the
   itinerary optimiser needs.
4. **goibibo_com-travel_sample.csv** - 4,000 hotels in 558 cities with star rating, guest
   rating, facilities, description and latitude/longitude. Besides hotel search, the median hotel
   location per city gives us city coordinates, which the transport agent uses for distances.
   153 cities with at least 5 hotels are offered in the "From" list.

## Other inputs

- **Fare rates** for train, bus and cab (`src/config.py`). Train: Indian Railways Mail/Express
  fares after the July and December 2025 revisions, roughly Delhi-Mumbai SL Rs 600, 3A Rs 1,700,
  2A Rs 2,450, 1A Rs 4,100. Bus: AC sleeper about Rs 2/km. Cab: sedan about Rs 13/km plus
  Rs 400 driver allowance (2025-26 operator rates).
- **Airport cities** (`AIRPORT_CITIES` in `src/config.py`) - non-metro cities with regular
  scheduled flights.
- **Knowledge base** (`data/knowledge_base/`) - hotel policies and travel advisories written by
  the team, because Kaggle has no suitable text dataset for these.

## Limitations

- Goibibo has no tariff column, so nightly prices are estimated from the star rating
  (`EST_PRICE_BY_STAR`). 48% of hotels have no star rating.
- Flight fares are from 2022 and only for the 6 metros. Other routes use an estimated fare.
- `Designation` in Travel.csv maps 1:1 to the package tier (Executive = Basic, Manager = Deluxe,
  Senior Manager = Standard, AVP = Super Deluxe, VP = King), so it is not used as a feature.

## Data handling

The datasets are public. They are kept only in the AI Lab repository. Only trip details and
public dataset text are sent to the MAAS LLM.
