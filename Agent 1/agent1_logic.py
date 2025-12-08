import os
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates,
)

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch


# ====== 建立 SBVR LocalSearch Engine ======

def build_sbvr_local_search_engine(
    index_root: str = "kyc_workflow/output",
    sbvr_system_prompt_path: str = "sbvr_local_search_system_prompt.txt",
    chat_model_name: str = "gpt-4.1-mini",
    embedding_model_name: str = "text-embedding-3-small",
    community_level: int = 2,
) -> LocalSearch:
    """
    建立一個專門用來做「SBVR 規則抽取」的一段式 LocalSearch。
    - system_prompt：使用 GraphRAG 風格的 SBVR 模板（含 {response_type}, {context_data}）
    - response：直接是 SBVR JSON 字串
    """

    input_dir = index_root
    lancedb_uri = os.path.join(index_root, "lancedb")

    api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("請先設定 GRAPHRAG_API_KEY 或 OPENAI_API_KEY")

    # 1) 讀取 index parquet
    entities_df = pd.read_parquet(os.path.join(input_dir, "entities.parquet"))
    communities_df = pd.read_parquet(os.path.join(input_dir, "communities.parquet"))
    relationships_df = pd.read_parquet(os.path.join(input_dir, "relationships.parquet"))
    reports_df = pd.read_parquet(os.path.join(input_dir, "community_reports.parquet"))
    text_units_df = pd.read_parquet(os.path.join(input_dir, "text_units.parquet"))
    covariates_df = pd.read_parquet(os.path.join(input_dir, "covariates.parquet"))

    # 2) 轉成 GraphRAG 內部物件
    entities = read_indexer_entities(entities_df, communities_df, community_level)
    relationships = read_indexer_relationships(relationships_df)
    reports = read_indexer_reports(reports_df, communities_df, community_level)
    text_units = read_indexer_text_units(text_units_df)
    claims = read_indexer_covariates(covariates_df)
    covariates = {"claims": claims}

    # 3) LanceDB vector store（entity description embeddings）
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=lancedb_uri)

    # 4) LLM / embedding / tokenizer 設定
    chat_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Chat,
        model_provider="openai",
        model=chat_model_name,
        max_retries=10,
    )
    embed_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Embedding,
        model_provider="openai",
        model=embedding_model_name,
        max_retries=5,
    )

    manager = ModelManager()
    chat_model = manager.get_or_create_chat_model(
        name="sbvr_local_search_chat",
        model_type=ModelType.Chat,
        config=chat_config,
    )
    text_embedder = manager.get_or_create_embedding_model(
        name="sbvr_local_search_embedding",
        model_type=ModelType.Embedding,
        config=embed_config,
    )
    tokenizer = get_tokenizer(chat_config)

    # 5) context builder
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

    # 6) Local search 參數（可按需要調）
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }
    model_params = {"max_tokens": 2_000, "temperature": 0.0}

    # 7) 載入 SBVR system_prompt 模板
    with open(sbvr_system_prompt_path, "r", encoding="utf-8") as f:
        sbvr_system_prompt = f.read()

    # 8) 建立一段式 SBVR LocalSearch
    search_engine = LocalSearch(
        model=chat_model,
        context_builder=context_builder,
        tokenizer=tokenizer,
        system_prompt=sbvr_system_prompt,
        response_type=(
            "Return SBVR-style business rules as a single JSON object with a top-level 'rules' array."
        ),
        model_params=model_params,
        context_builder_params=local_context_params,
    )

    return search_engine


