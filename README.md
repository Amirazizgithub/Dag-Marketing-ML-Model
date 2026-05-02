# Dag-Marketing-ML-Model

A production-ready what-if simulator that uses a **Directed Acyclic Graph (DAG)** of lightweight ML models to predict how changes in industry (e.g. ecommerce, healthcare, automobile, etc) KPIs cascade through the funnel to impact revenue.

## Why Causal DAG (not correlation)?

| Approach | Problem |
|----------|---------|
| Correlation matrix | Linear only — can't capture diminishing returns |
| Per-KPI regression | Circular dependencies (A predicts B, B predicts A) |
| VAR model | Linear, hard to tune, not interpretable |
| **Causal DAG** | **Non-linear, no cycles, interpretable, fast** |

## Project Structure

```
spinotale-marketing-dagmodel/
├── ⚙️ model_configs/
│   └── __init__.py                # Configuration schemas and hyperparameters
├── 🧠 models/
│   ├── __init__.py
│   ├── edge_model.py              # Per-edge LightGBM wrapper (Base Model logic)
│   └── dag_engine.py              # Orchestrator for train, simulate, forecast, and explain
├── 🏗️ components/
│   ├── __init__.py
│   ├── dag_definition.py          # DAG structure & node relationships (conversions, AOV, etc.)
│   ├── data_loading.py            # BigQuery/GCP connectors and data fetching
│   ├── data_preprocessing.py      # Feature engineering, scaling, and lag creation
│   └── model_trainer.py           # Training loop and factor analysis implementation
├── 🦊 gitlab-pipelines/
│   ├── .gitlab-ci-dev.yml         # Dev pipeline (Linting, Formatting, Unit Tests)
│   ├── .gitlab-ci-prod.yml        # Prod pipeline (GKE Autopilot/Cloud Run deployment)
│   └── .gitlab-ci-stag.yml        # Staging pipeline (Integration testing)
├── 🚀 api/
│   ├── __init__.py
│   └── router.py                  # FastAPI endpoints for React frontend integration
├── 🛠️ utils/
│   ├── __init__.py
│   └── logger.py                  # Standardized logging for GCP Cloud Logging
├── 🐳 Dockerfile                  # Containerization instructions
├── 📦 requirements.txt            # Production dependencies (LightGBM, FastAPI, etc.)
├── 🦊 .gitlab-ci.yml              # Main GitLab CI Entry point (GitLab Shared Runners)
├── 🙈 .gitignore                  # Files to exclude from Git (venv, .pyc, secrets)
├── 🐳 .dockerignore               # Files to exclude from Docker build
├── 📜 LICENSE                     # Project licensing terms
└── 📖 README.md                   # Project documentation and setup guide
```

## Architecture

```
Layer 0 (Root):       Ad Spend ──→ CPC
                         ↓           ↓
Layer 1 (Traffic):    Impressions → Clicks → CTR
                         ↓           ↓       ↓
Layer 2 (Engagement): Sessions → New Users → Bounce Rate → Engaged Sessions
                         ↓           ↓           ↓              ↓
Layer 3 (Conversion): Add to Cart ──────→ Conversions    →    AOV
                                              ↓               ↓
Layer 4 (Target):                          Revenue
                                              ↓
Layer 5 (Derived):                          ROAS
```

Each arrow is a trained **LightGBM model** (not a static number). It captures non-linear relationships like diminishing returns on ad spend.

## ⚙️ Installation

## 🛠️ Prerequisites

- Python 3.12+
- Google Cloud Project with Vertex AI enabled
- Service account key for Google Cloud authentication

### ⚙️ Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Amirazizgithub/Dag-Marketing-ML-Model.git
cd dag-marketing-ml-model
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Google Cloud authentication:**
   Place your service account key file as central-sa-key.json in the root directory.

## 🏃‍♂️ Usage

### Running the Application

Start the FastAPI server with auto-reload:
```bash
uvicorn app:app --reload
```

The API will be available at http://127.0.0.1:8000

### API Documentation

Once running, visit http://127.0.0.1:8000/docs for interactive Swagger UI documentation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Single-point what-if scenario |
| `/forecast` | POST | Multi-day revenue forecast with CI |
| `/explain` | POST | Per-tweak revenue impact breakdown |
| `/baselines` | GET | Current baseline values |
| `/dag` | GET | Full DAG topology |
| `/model-report` | GET | R², MAPE per edge model |
| `/ui/slider-config` | GET | Config for React slider panel |

### Example: `/simulate`

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "manual_overrides": {"ad_spend": 25000, "bounce_rate": 35},
    "locked_kpis": ["cpc"]
  }'
```

Response:
```json
{
  "values": {"ad_spend": 25000, "impressions": 178432, "clicks": 5120, ...},
  "sources": {"ad_spend": "manual", "impressions": "predicted", "cpc": "locked", ...},
  "deltas": {"impressions": {"baseline": 145000, "predicted": 178432, "change_pct": 23.1}, ...}
}
```

### Example: `/forecast`

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "manual_overrides": {"ad_spend": 20000},
    "days": 10,
    "ci_level": 0.95
  }'
```

## How the What-If Propagation Works

When a user tweaks **any** KPI:

1. **Manual override** → value is set directly (green in UI)
2. **Upstream KPIs** → stay at baseline (they caused this KPI, not the other way)
3. **Downstream KPIs** → re-predicted by edge models using the new value
4. **Locked KPIs** → immune to cascade, held at baseline
5. **Revenue** → always recomputed at the bottom of the DAG

Example: User sets `Sessions = 5000` manually:
- `Ad Spend`, `Impressions`, `Clicks` → unchanged (upstream)
- `New Users` → re-predicted from `sessions=5000 + ad_spend` (downstream)
- `Bounce Rate` → re-predicted from `sessions=5000 + ctr` (downstream)
- `Conversions` → re-predicted from the new cascade values
- `Revenue` → updated

Minimum: 6 months daily data. Recommended: 12+ months for seasonality.

## Customising the DAG

Edit `dag_definition.py` to:
- Add new KPIs (e.g., `email_signups`, `cart_abandonment`)
- Change parent-child relationships
- Adjust min/max ranges

The engine auto-discovers the topology — no code changes needed elsewhere.

## 🐳 Docker Deployment

### Build the Docker image:
```bash
docker build -t dag-marketing-ml-model-{ENVIRONMENT}:latest .
```

### Run the container:
```bash
docker run -p 8000:8000 dag-marketing-ml-model-{ENVIRONMENT}:latest
```

## 🛠️ CI/CD Pipelines

The project includes GitLab CI/CD pipelines for:

- **Development**: Code formatting and linting
- **Staging**: Automated testing and deployment
- **Production**: Full deployment pipeline

Pipelines are triggered based on branch:
- development branch: Development pipeline
- staging branch: Staging deployment
- production branch: Production deployment

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚙️ Support

For support or questions:
- Create an issue in the repository
- Contact the development team

## Roadmap

- [ ] Enhanced data validation
- [ ] Additional AI model integrations
- [ ] Real-time insight streaming
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
