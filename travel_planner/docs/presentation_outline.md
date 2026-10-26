# Presentation outline

1. **Problem and why it matters.** Planning a trip means comparing transport, hotels and sights
   against a budget. The assistant does this for the user, personalised with ML, agents and RAG.
2. **Datasets.** The 4 Kaggle datasets, what each one feeds and their limits
   (`docs/dataset_sources.md`).
3. **EDA.** Tier distribution, income vs tier, the Designation leakage, flight price vs days
   left, hotel coverage.
4. **Sprints.** Table from `docs/sprint_log.md`.
5. **Architecture.** Diagram from `docs/architecture.md`: agent flow, RAG flow, optimiser.
6. **ML models.** Comparison table and chart (`reports/`), confusion matrix, feature importance,
   why Random Forest was picked.
7. **Agents.** One slide each for transport, hotel, attraction and budget: input, logic, output.
   For transport show the mode comparison (Delhi-Mumbai for Basic vs Standard tier).
8. **RAG.** Sources, why one index per source, MAAS answer with citations, 2 example questions.
9. **Optimised itinerary.** Greedy algorithm, constraints, sample output.
10. **Demo.** (a) Delhi to Mumbai, 4 days, Rs 80k; (b) Pune to Goa to show a non-metro city;
    (c) Delhi to Jaipur on a low budget to show the budget agent; (d) Ask tab: "cancellation
    policy?" and "Goa in monsoon?"; (e) Model report tab.
11. **Results.**
12. **Challenges.** No hotel prices, flights only between metros, no advisory text on Kaggle,
    leakage, getting the MAAS connection working, 6-hour lab sessions.
13. **Team contributions.** `docs/team_contributions.md`.
14. **Future work.** Live fare APIs, embeddings + FAISS, multi-city trips, user feedback.
