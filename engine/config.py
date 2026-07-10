from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SECTOR_MAP_PATH = ROOT / "engine" / "sector_map.yaml"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065  # India 10y-ish, tweak in request
MIN_HISTORY_DAYS = 252  # 1 year minimum
DEFAULT_LOOKBACK_YEARS = 5
