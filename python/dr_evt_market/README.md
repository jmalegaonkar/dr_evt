# dr_evt_market

A federation market over DR_EVT. Four platforms that mirror real machines each run one
DR_EVT simulation over the share of the machine they expose. A job stream with bids
enters a queue; at each window the auction takes a prefix of the queue and sends every
winner to the platform it won. See `docs/api/MARKET_PYTHON_API.md`.
