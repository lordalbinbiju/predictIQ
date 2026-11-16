"""Central configuration constants and paths for PredictIQ."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATASET_PATH = DATA_DIR / "telco_churn.csv"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
DB_PATH = BASE_DIR / "database" / "predictiq.db"
ASSETS_DIR = BASE_DIR / "assets"
STYLES_PATH = ASSETS_DIR / "styles.css"

APP_TITLE = "PredictIQ"
APP_TAGLINE = "Turning Customer Data into Predictive Decisions"
APP_SUBTITLE = "Customer Churn Prediction & Explainable Analytics Platform"

# Color palette (SAS Viya-inspired enterprise light theme with navy/blue accents)
PALETTE = {
    "navy": "#0A2540",
    "primary_blue": "#0F4C81",
    "accent_blue": "#2563EB",
    "sky_blue": "#0EA5E9",
    "teal": "#0D9488",
    "amber": "#D97706",
    "crimson": "#DC2626",
    "emerald": "#059669",
    "slate_900": "#0F172A",
    "slate_700": "#334155",
    "slate_500": "#64748B",
    "slate_200": "#E2E8F0",
    "bg_light": "#F5F7FA",
    "card_bg": "#FFFFFF",
}

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

RISK_THRESHOLDS = {
    "low_max": 0.35,
    "high_min": 0.65,
}

NAVIGATION_ITEMS = [
    "Overview",
    "Data Management",
    "Exploratory Data Analysis",
    "Model Training",
    "Model Evaluation",
    "Customer Predictions",
    "Explainable AI",
    "Business Insights",
    "Model History",
    "About",
]

POSITIVE_TARGET_VALUES = {"yes", "1", "true", "churn", "churned", "y"}
NEGATIVE_TARGET_VALUES = {"no", "0", "false", "retained", "stay", "n"}

COMMON_ID_COLUMNS = {"customerid", "customer_id", "id", "cust_id", "account_id"}
COMMON_TARGET_COLUMNS = ["Churn", "churn", "Exited", "exited", "Target", "target", "label"]
