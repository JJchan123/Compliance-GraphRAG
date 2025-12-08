import os
import pandas as pd

# --- GraphRAG imports ---
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

def build_local_search_engine() -> LocalSearch:
    """
    從 GraphRAG index 建立一個可重用的 LocalSearch search_engine。
    """
    # --- 1. 路徑與基本設定 ---
    G33A_INPUT_DIR = "kyc_workflow/output"
    LANCEDB_URI = f"{G33A_INPUT_DIR}/lancedb"
    COMMUNITY_LEVEL = 2
    GRAPHRAG_API_KEY = os.environ.get("GRAPHRAG_API_KEY")

    if not GRAPHRAG_API_KEY:
        raise RuntimeError("請先在環境變數中設定 GRAPHRAG_API_KEY")

    # --- 2. 讀取 index parquet ---
    entity_df = pd.read_parquet(f"{G33A_INPUT_DIR}/entities.parquet")
    community_df = pd.read_parquet(f"{G33A_INPUT_DIR}/communities.parquet")
    relationship_df = pd.read_parquet(f"{G33A_INPUT_DIR}/relationships.parquet")
    report_df = pd.read_parquet(f"{G33A_INPUT_DIR}/community_reports.parquet")
    text_unit_df = pd.read_parquet(f"{G33A_INPUT_DIR}/text_units.parquet")
    covariate_df = pd.read_parquet(f"{G33A_INPUT_DIR}/covariates.parquet")

    # --- 3. 轉成 GraphRAG 內部資料結構 ---
    entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
    text_units = read_indexer_text_units(text_unit_df)
    claims = read_indexer_covariates(covariate_df)
    
    covariates = {"claims": claims}

    # --- 4. 建立 LanceDB vector store ---
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    # --- 5. 建立 GraphRAG 內部的 LLM / embedding / tokenizer ---
    chat_config = LanguageModelConfig(
        api_key=GRAPHRAG_API_KEY,
        type=ModelType.Chat,
        model_provider="openai",
        model="gpt-4.1-mini",
        max_retries=10,
    )
    embed_config = LanguageModelConfig(
        api_key=GRAPHRAG_API_KEY,
        type=ModelType.Embedding,
        model_provider="openai",
        model="text-embedding-3-small",
        max_retries=5,
    )

    manager = ModelManager()
    chat_model = manager.get_or_create_chat_model(
        name="g33a_local_search", model_type=ModelType.Chat, config=chat_config
    )
    text_embedder = manager.get_or_create_embedding_model(
        name="g33a_local_search_embedding", model_type=ModelType.Embedding, config=embed_config
    )
    tokenizer = get_tokenizer(chat_config)

    # --- 6. 建立 context_builder ---
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        tokenizer=tokenizer,
    )

    # --- 7. LocalSearch 參數 ---
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 20,
        "top_k_relationships": 20,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }
    model_params = {"max_tokens": 2_000, "temperature": 0.0}

    # --- 8. 最終組裝 ---
    search_engine = LocalSearch(
        model=chat_model,
        context_builder=context_builder,
        tokenizer=tokenizer,
        model_params=model_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    return search_engine