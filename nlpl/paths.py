import os


current_file: str = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR: str = os.path.dirname(current_file)
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
CACHE_DIR: str = os.path.join(ROOT_DIR, ".cache")
MODELS_DIR: str = os.path.join(ROOT_DIR, "models")
VECTORS_DIR: str = os.path.join(ROOT_DIR, "vectors")
TENSORBOARD_LOGS_DIR: str = os.path.join(ROOT_DIR, ".tensorboard_logs")
