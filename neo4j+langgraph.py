import os
import re
from typing import Dict, List, Optional, TypedDict

from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings 
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase


class GraphRAGPipeline:
    """
    Encapsulates the Graph RAG logic for local retrieval and generation.
    """

    # ===== 1. Initialization and Setup =====
    def __init__(self, system_prompt_path="system_prompt.txt"):
        print("🚀 Initializing GraphRAG Pipeline...")
        # --- Model Configuration ---
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4.1-mini")

        # --- Neo4j Connection ---
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

        # --- Load System Prompt ---
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            print(f"⚠️ Warning: System prompt file not found at {system_prompt_path}. Using a default prompt.")
            self.system_prompt = "You are a helpful assistant."


        # --- Compile the LangGraph ---
        self.query_engine = self._build_graph()
        print("✅ Pipeline Initialized Successfully.")

    # ===== 2. Core RAG Logic (as private methods) =====
    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Performs a vector similarity search against the Clause index."""
        vector_store = Neo4jVector.from_existing_index(
            embedding=self.embedding_model,
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="clauses",
            text_node_property="full_text",
        )
        results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        
        print("\n--- Vector Search Results ---")
        for doc, score in results_with_scores:
            print(f"Score: {score:.4f}\t Clause UID: {doc.metadata['uid']}")
        print("-----------------------------\n")

        return [dict(doc) for doc, score in results_with_scores]

    def _format_context_block(self, record: Dict) -> str:
        """Formats a graph record into a markdown block for the LLM."""
        if not record: return ""
        
        context_parts = [
            f"---",
            f"**Source:** Document {record.get('doc_code', 'N/A')}, Clause {record.get('clause_id', 'N/A')}, Page {record.get('page', 'N/A')}",
            f"**Content:** {record.get('text', '')}"
        ]
        if record.get('parent_heading'):
            context_parts.append(f"**Parent:** {record['parent_heading']}")
        
        siblings = record.get('sibling_clauses', [])
        if siblings:
            context_parts.append(f"**Sibling Clauses:** {', '.join(siblings[:3])}")

        attachments = record.get('attachments', [])
        if attachments:
            attachment_strs = [f"- **{att['type']}:** {att['caption']}" for att in attachments if att and att.get('caption')]
            if attachment_strs:
                context_parts.append("**Related Attachments:**\n" + "\n".join(attachment_strs))
                
        context_parts.append("---")
        return "\n".join(context_parts)

    # ===== 3. LangGraph Nodes =====
    def _local_retriever(self, state: "QueryState") -> dict:
        """Handles local, semantic queries using vector search and graph expansion."""
        query = state["query"]
        print(f"🔍 Performing local retrieval for: '{query}'")
        candidate_clauses = self._vector_search(query, top_k=5)
        
        rich_context_parts = []
        cypher_query = """
            MATCH (c:Clause {uid: $clause_uid})
            OPTIONAL MATCH (doc:Document)-[:HAS_SECTION|HAS_SUBSECTION*0..]->(c)
            OPTIONAL MATCH (parent)-[:HAS_CLAUSE|HAS_SUBSECTION]->(c)
            OPTIONAL MATCH (parent)-[:HAS_CLAUSE|HAS_SUBSECTION]->(sibling:Clause) WHERE c <> sibling
            OPTIONAL MATCH (c)-[:HAS_IMAGE|HAS_TABLE]->(media)-[:HAS_CAPTION]->(caption:MultimodalCaption)
            RETURN c.uid AS clause_uid, c.clause_id AS clause_id, c.full_text AS text, c.page_start AS page,
                   doc.doc_code AS doc_code, parent.heading AS parent_heading,
                   collect(DISTINCT sibling.clause_id) AS sibling_clauses,
                   collect(DISTINCT {type: labels(media)[0], caption: caption.caption}) AS attachments
        """
        with self.driver.session() as session:
            for clause in candidate_clauses:
                uid = clause['metadata']['uid']
                result = session.run(cypher_query, clause_uid=uid)
                expanded_context = result.single()
                if expanded_context:
                    context_block = self._format_context_block(expanded_context.data())
                    rich_context_parts.append(context_block)
        return {"retrieved_context": "\n\n".join(rich_context_parts)}

    def _generate_answer(self, state: "QueryState") -> dict:
        """Generates the final answer using the LLM."""
        prompt = f"""{self.system_prompt}
Answer the question based ONLY on the provided context. Cite sources using [Doc ID, Clause ID, Page].
Question: {state['query']}
Context: {state['retrieved_context']}
Answer:"""
        response = self.llm.invoke(prompt)
        return {"final_answer": response.content}

    # ===== 4. Graph Construction and Public Interface =====
    def _build_graph(self) -> "StateGraph":
        """Compiles and returns the LangGraph engine."""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("local_retriever", self._local_retriever)
        workflow.add_node("generate", self._generate_answer)

        # Set a fixed entry point, as we no longer need routing
        workflow.set_entry_point("local_retriever")

        # Define the edges
        workflow.add_edge("local_retriever", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def run(self, query: str) -> str:
        """
        The main public method to run a query through the pipeline.
        """
        print(f"\n--- Running new query ---\nQuery: {query}")
        result = self.query_engine.invoke({"query": query})
        print(f"Final Answer: {result['final_answer']}")
        return result['final_answer']

# Define the state for the graph, needs to be accessible by the class
class QueryState(TypedDict):
    query: str
    retrieved_context: str
    final_answer: Optional[str]
