# Institutional response lag in Chinese vocational education

Data and core Python code accompanying the manuscript submitted to Nature Human Behaviour.

- `data/source_data/`: Origin input workbooks for Figures 2–4.
- `data/analysis/`: frozen events and matched chains, sensitivity tables, and supplementary case data.
- `code/scripts/`: analysis, replay, calibration, simulation scripts and helper modules.

The main sample contains 1,537 events and 800 matched chains. Supplementary case data are separate from this sample.

Install core dependencies with `pip install -r requirements.txt`; optional replay dependencies are listed in `requirements-replay.txt`. To run analyses from the frozen sample, copy `data/analysis/event_chains/` into `code/outputs/event_analysis/` first.

