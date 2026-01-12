import os
import pandas as pd
from typing import Any, Optional

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

def build_search_engine(
    index_root: str,
    system_prompt_path: Optional[str] = None,
    response_type: str = "multiple paragraphs",
    chat_model_name: str = "gpt-4.1-mini",
    embedding_model_name: str = "text-embedding-3-small",
    community_level: int = 2,
) -> LocalSearch:
    """
    A general-purpose LocalSearch engine builder.
    It can create different engines by passing different system_prompt_path and response_type.
    """
    input_dir = index_root
    lancedb_uri = os.path.join(index_root, "lancedb")

    api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GRAPHRAG_API_KEY or OPENAI_API_KEY first")

    # 1) Read index parquet
    entities_df = pd.read_parquet(os.path.join(input_dir, "entities.parquet"))
    communities_df = pd.read_parquet(os.path.join(input_dir, "communities.parquet"))
    relationships_df = pd.read_parquet(os.path.join(input_dir, "relationships.parquet"))
    reports_df = pd.read_parquet(os.path.join(input_dir, "community_reports.parquet"))
    text_units_df = pd.read_parquet(os.path.join(input_dir, "text_units.parquet"))
    covariates_df = pd.read_parquet(os.path.join(input_dir, "covariates.parquet"))

    # 2) Convert to GraphRAG internal objects
    entities = read_indexer_entities(entities_df, communities_df, community_level)
    relationships = read_indexer_relationships(relationships_df)
    reports = read_indexer_reports(reports_df, communities_df, community_level)
    text_units = read_indexer_text_units(text_units_df)
    claims = read_indexer_covariates(covariates_df)
    covariates = {"claims": claims}

    # 3) LanceDB vector store
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=lancedb_uri)

    # 4) LLM / embedding / tokenizer setup
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
        name="generic_chat",
        model_type=ModelType.Chat,
        config=chat_config,
    )
    text_embedder = manager.get_or_create_embedding_model(
        name="generic_embedding",
        model_type=ModelType.Embedding,
        config=embed_config,
    )
    tokenizer = get_tokenizer(chat_config)

    # 5) Context builder
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
    model_params = {"max_tokens": 4000, "temperature": 0.0}

    # 7) Load system_prompt template
    system_prompt = None
    if system_prompt_path:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    # 8) Create LocalSearch
    search_engine = LocalSearch(
        model=chat_model,
        context_builder=context_builder,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        response_type=response_type,
        model_params=model_params,
        context_builder_params=local_context_params,
    )

    return search_engine
