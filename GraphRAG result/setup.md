# GraphRAG Setup Workflow

This document describes the complete workflow for setting up a GraphRAG project, including initialization, prompt tuning, indexing, and querying.

---

# 1. Create GraphRAG Project

Initialize a new GraphRAG workspace.

```bash
graphrag init --root "{project_path}"
````

Example:

```bash
graphrag init --root "C:\Users\User\Desktop\Microsoft_GraphRAG\{project_name}"
```

After initialization, the directory structure will be:

```
{project_name}
│
├── input
├── prompts
├── settings.yaml
├── .env
```

---

# 2. Prepare Input Data

Place your document(s) into the `input` folder.

Example:

```
{project_name}
│
├── input
│   └── {filename}.txt
```

Or if using CSV:

```
{project_name}
│
├── input
│   └── {filename}.csv
```

---

# 3. Configure API Key

Edit the `.env` file and add your OpenAI API key.

```
GRAPHRAG_API_KEY=your_openai_api_key
```

---

# 4. Configure settings.yaml

Modify the input configuration depending on the file format.

## Text Input

```yaml
input:
  storage:
    type: file
    base_dir: "input"
  file_type: text
```

---

## CSV Input with Metadata

```yaml
input:
  storage:
    type: file
    base_dir: "input"
  file_type: csv
  metadata: [title, version, author, section_num, section_title]
```

---

# 5. Define Entity Types (Optional but Recommended)

For regulatory documents (e.g., financial compliance, data privacy), it is recommended to manually define entity types.

Example:

```yaml
extract_graph:
  entity_types:
    - PolicyRequirement
    - Procedure
    - Role
    - RiskType
    - Condition
    - Control
    - Outcome
```

---

# 6. Run Auto Prompt Tuning

Run prompt tuning to generate optimized prompts.

```bash
python -m graphrag prompt-tune \
--root "{project_path}" \
--config "{project_path}\settings.yaml" \
--domain "financial compliance" \
--selection-method all \
--no-discover-entity-types \
--output "{project_path}\auto_tuned_prompts"
```

Example:

```bash
python -m graphrag prompt-tune --root "C:\Users\User\Desktop\Microsoft_GraphRAG\{project_name}" --config "C:\Users\User\Desktop\Microsoft_GraphRAG\{project_name}\settings.yaml" --domain "financial compliance" --selection-method all --no-discover-entity-types --output "C:\Users\User\Desktop\Microsoft_GraphRAG\{project_name}\auto_tuned_prompts"
```

Generated prompts will appear in:

```
{project_name}
│
├── auto_tuned_prompts
│   ├── extract_graph.txt
│   ├── summarize_descriptions.txt
│   ├── community_report_graph.txt
│   └── community_report_text.txt
```

---

# 7. Update settings.yaml to Use Tuned Prompts

Modify the prompt paths.

## Graph Extraction

```yaml
extract_graph:
  model_id: default_chat_model
  prompt: "auto_tuned_prompts/extract_graph.txt"
```

---

## Description Summarization

```yaml
summarize_descriptions:
  model_id: default_chat_model
  prompt: "auto_tuned_prompts/summarize_descriptions.txt"
```

---

## Community Reports

```yaml
community_reports:
  graph_prompt: "auto_tuned_prompts/community_report_graph.txt"
  text_prompt: "auto_tuned_prompts/community_report_text.txt"
```

---

# 8. Build GraphRAG Index

Run the indexing pipeline.

```bash
graphrag index --root "{project_path}"
```

Example:

```bash
graphrag index --root "C:\Users\User\Desktop\Microsoft_GraphRAG\{project_name}"
```

This process will generate:

```
output/
cache/
logs/
```

The vector database will be stored in:

```
output/lancedb
```

---

# 9. Query GraphRAG

After indexing, queries can be executed.

---

## Local Search

Best for specific questions.

```bash
graphrag query --root "{project_path}" --method local --query "What are the regulatory requirements for {topic}?"
```

---

## Global Search

Best for high-level summaries.

```bash
graphrag query --root "{project_path}" --method global --query "Summarize the supervisory framework."
```

---

## Drift Search

Uses community information for exploratory queries.

```bash
graphrag query --root "{project_path}" --method drift --query "Explain how the regulation addresses risk management."
```

---

# 10. Overall Pipeline

The GraphRAG workflow can be summarized as:

```
Document
   ↓
Project Initialization
   ↓
Input Preparation
   ↓
Prompt Tuning
   ↓
Prompt Integration
   ↓
GraphRAG Indexing
   ↓
Knowledge Graph + Vector Database
   ↓
Query and Retrieval
```

---
