from pydantic_settings import BaseSettings
from enum import Enum
from pathlib import Path
import os

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


class ServiceType(str, Enum):
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    # Deepseek settings
    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # Vision Model settings (独立配置)
    VISION_API_KEY: str
    VISION_BASE_URL: str
    VISION_MODEL: str = "qwen3-vl-plus"

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:7b"
    OLLAMA_REASON_MODEL: str = "deepseek-r1:7b"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"
    OLLAMA_AGENT_MODEL: str = "qwen2.5:7b"

    # Service selection
    CHAT_SERVICE: ServiceType = ServiceType.DEEPSEEK
    REASON_SERVICE: ServiceType = ServiceType.DEEPSEEK
    AGENT_SERVICE: ServiceType = ServiceType.DEEPSEEK

    # Search settings
    SERPAPI_KEY: str = ""  # 可选，留空则禁用联网搜索
    SEARCH_RESULT_COUNT: int = 3

    # Database settings
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Neo4j settings
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # ⚠️ 生产环境必须配置密码
    NEO4J_DATABASE: str = "neo4j"

    # JWT settings
    # ⚠️ 生产环境必须更换为随机生成的强密钥（至少 32 字节）
    # 生成方法：python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = "change-this-to-random-secure-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 小时

    # Redis settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_CACHE_EXPIRE: int = 3600
    REDIS_CACHE_THRESHOLD: float = 0.75  # 降低阈值以提高召回率

    # Embedding settings
    EMBEDDING_TYPE: str = "ollama"  # ollama 或 sentence_transformer
    EMBEDDING_MODEL: str = "bge-m3"  # bge-m3 对中文优化更好
    EMBEDDING_THRESHOLD: float = 0.75  # 语义相似度阈值，降低以提高召回率

    # GraphRAG settings
    GRAPHRAG_PROJECT_DIR: str = str(ROOT_DIR / "app" / "graphrag")
    GRAPHRAG_DATA_DIR: str = "data"
    GRAPHRAG_QUERY_TYPE: str = "local"
    GRAPHRAG_RESPONSE_TYPE: str = "text"
    GRAPHRAG_COMMUNITY_LEVEL: int = 3
    GRAPHRAG_DYNAMIC_COMMUNITY: bool = False

    # GraphRAG API 配置（默认复用 DeepSeek 配置，可通过环境变量覆盖）
    GRAPHRAG_API_BASE: str = ""
    GRAPHRAG_API_KEY: str = ""
    GRAPHRAG_MODEL_NAME: str = "deepseek-chat"

    # GraphRAG Embedding 配置
    EMBEDDING_API_BASE: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL_NAME: str = "text-embedding-v3"

    # PDF 处理配置
    PDF_OUTPUT_DIR: str = str(ROOT_DIR / "data" / "pdf_outputs")
    MINERU_API_URL: str = "http://localhost:8000/"
    MINERU_OUTPUT_DIR: str = "/tmp/mineru_output"
    TABLE_DESCRIPTION_API_KEY: str = ""
    TABLE_DESCRIPTION_MODEL: str = "deepseek-chat"
    TABLE_DESCRIPTION_BASE_URL: str = ""
    IMAGE_DESCRIPTION_API_KEY: str = ""
    IMAGE_DESCRIPTION_MODEL: str = "qwen3-vl-plus"
    IMAGE_DESCRIPTION_BASE_URL: str = ""

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def REDIS_URL(self) -> str:
        """构建 Redis URL"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def NEO4J_CONN_URL(self) -> str:
        """构建 Neo4j 连接 URL"""
        return f"{self.NEO4J_URL}"

    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # 忽略多余的环境变量


settings = Settings()
