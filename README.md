US Tech Sector Heatmap (Blocks)

What this is
- A Streamlit app that builds a sector-level heatmap for US tech-related blocks.
- Block size = total daily volume.
- Block color = daily % change (volume-weighted when available).
- Heatmap size can switch between total volume and total turnover (volume x price).
- Includes a momentum backtest module (sector rotation).

Quick start
1. Install deps
   pip install -r requirements.txt
2. No API key required (uses yfinance / Yahoo Finance)
3. Run
   streamlit run app.py
4. Trading planner (separate app)
   streamlit run trading_app.py

Alpaca paper trading (optional)
- Set env vars: APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
- Default paper URL: https://paper-api.alpaca.markets

Notes
- Uses daily candles via yfinance (Yahoo Finance). Data may be delayed or missing.
- Edit sectors.json to change blocks and symbols.
