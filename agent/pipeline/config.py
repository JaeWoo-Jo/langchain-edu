import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Elasticsearch
ES_URL = os.getenv("ES_URL", "https://elasticsearch-edu.didim365.app")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_INDEX = os.getenv("ES_INDEX", "edu-recipe-rag")

# 데이터 디렉토리
DATA_DIR = Path(__file__).parent / "data"
RECIPES_DIR = DATA_DIR / "recipes"
NUTRITION_DIR = DATA_DIR / "nutrition"
INGREDIENTS_DIR = DATA_DIR / "ingredients"

# 식품안전처 API
FOOD_SAFETY_API_KEY = os.getenv("FOOD_SAFETY_API_KEY", "")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
EMBEDDING_BATCH_SIZE = 100
