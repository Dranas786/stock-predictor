import os


def env(name: str, default: str) -> str:
    """
    Read an environment variable with a default.

    Keeping this tiny helper avoids repeating os.getenv everywhere and
    makes it easy to override settings in Docker/CI without code changes.
    """
    return os.getenv(name, default)


# --- Database (Postgres) ---
POSTGRES_HOST = env("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(env("POSTGRES_PORT", "5432"))
POSTGRES_DB = env("POSTGRES_DB", "stock_predictor")
POSTGRES_USER = env("POSTGRES_USER", "stock")
POSTGRES_PASSWORD = env("POSTGRES_PASSWORD", "stock")

POSTGRES_DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- MLflow ---
MLFLOW_TRACKING_URI = env("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --- S3 / MinIO (artifact store) ---
S3_ENDPOINT_URL = env("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY_ID = env("S3_ACCESS_KEY_ID", "minio")
S3_SECRET_ACCESS_KEY = env("S3_SECRET_ACCESS_KEY", "minio12345")
S3_BUCKET = env("S3_BUCKET", "mlflow-artifacts")
