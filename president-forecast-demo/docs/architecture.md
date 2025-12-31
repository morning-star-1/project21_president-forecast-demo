# Architecture

## Overview
- Streamlit UI for election forecast inputs
- Monte Carlo simulation in `app.py`
- Downloads historical results from a public dataset

## Data flow
Dataset -> baseline margins -> simulation -> charts

## Key decisions
- Use a public dataset URL for convenience
- Keep simulation parameters user-adjustable
