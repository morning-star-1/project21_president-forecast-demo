# President Forecast Demo

A Streamlit demo that combines historical election results with Monte Carlo simulation to forecast electoral outcomes.

## Screenshot
![Screenshot](docs/screenshot.png)

Replace `docs/screenshot.png` with a real screenshot or GIF.

## Quickstart
### Prerequisites
- Python 3.11+
- Network access to download the MIT elections dataset

### Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data
- Default dataset URL is defined in the sidebar.
- `polls.csv` can be uploaded as an optional override.

## Tests
```bash
python -m unittest discover -s tests
```
