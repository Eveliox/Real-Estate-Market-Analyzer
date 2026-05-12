"""Real Estate Market Analyzer - production modules.

Each submodule corresponds to a phase of the data-science pipeline:

    data_loader     Phase 1 - fetch raw data into data/raw/
    preprocessing   Phase 2 - clean and prepare data
    eda             Phase 2 - exploratory analysis helpers
    models          Phase 4 - regression model training
    forecasting     Phase 5 - time-series forecasting
    roi_calculator  Phase 6 - real estate finance math
    export          Phase 7 - emit CSV / SQLite for Power BI

Notebooks in ../notebooks/ orchestrate these modules interactively.
"""

__version__ = "0.1.0"
