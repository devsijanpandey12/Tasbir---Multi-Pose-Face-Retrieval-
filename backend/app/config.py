"""Centralized configuration and upload guardrails for Tasbir."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TASBIR_", extra="ignore")

    app_name: str = "Tasbir — Multi-Angle Photo Finder"
    app_version: str = "1.0.0"

    # Storage paths
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models" / "buffalo_l"

    # Model download source (official InsightFace v0.7 buffalo_l ONNX pack)
    buffalo_l_url: str = (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    )

    # Upload guardrails
    max_pool_batch: int = 200
    max_photo_size_mb: int = 15
    max_video_size_mb: int = 25
    max_video_seconds: float = 12.0
    max_reference_photos: int = 5

    # Image processing settings
    max_detection_dim: int = 1600
    thumb_max_dim: int = 480

    # Detection & matching thresholds
    min_det_score: float = 0.50
    min_face_size_px: int = 24
    min_blur_variance: float = 20.0
    default_threshold: float = 0.42
    pass2_anchor_threshold: float = 0.50
    pass2_min_face_score: float = 0.25
    pass2_min_outfit_score: float = 0.55
    pass2_combined_threshold: float = 0.44

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tasbir.db"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def outfit_embeddings_path(self) -> Path:
        return self.data_dir / "outfit_embeddings.npy"

    @property
    def det_model_path(self) -> Path:
        return self.models_dir / "det_10g.onnx"

    @property
    def rec_model_path(self) -> Path:
        return self.models_dir / "w600k_r50.onnx"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
