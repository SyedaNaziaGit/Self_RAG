# Self-RAG with LangGraph 

**Self-RAG** is an advanced retrieval-augmented generation technique that significantly enhances **traditional retrieval systems** by integrating continuous **self-reflection** throughout the text generation process. Unlike standard pipelines that blindly trust retrieved external documents, this approach uses a **large language model** to independently assess every stage of execution. Specifically, the system evaluates whether retrieval is genuinely necessary for a given user query, filters out **irrelevant evidence**, and verifies that generated responses remain strictly grounded in the provided facts to prevent **hallucinations**. Furthermore, if an initial response is deemed unhelpful or unsupported, the framework dynamically **refines queries** and enters iterative correction loops to guarantee accuracy. Ultimately, implementing this adaptive architecture within **LangGraph** allows developers to build robust, **fact-checking** conversational agents that exercise rigorous quality control over their own outputs.

 **Self-RAG (Self-Reflective Retrieval-Augmented Generation)** architecture—an advanced RAG framework where the Large Language Model (LLM) actively evaluates and fact-checks its own actions, evidence, and outputs at every step.

---

### 1. Limitations of Traditional RAG
The source identifies three primary flaws with standard RAG pipelines that Self-RAG aims to fix:
* **Indiscriminate Retrieval**: Standard RAG fetches external documents for every query—even simple questions the LLM already knows from training—wasting computation and leading to unconfident answers.
* **Blind Trust**: Standard RAG blindly relies on retrieved documents based purely on semantic similarity, even when those documents fail to address the actual question.
* **Lack of Self-Verification**: Traditional RAG outputs answers without checking whether the generated response contains hallucinations or accurately resolves the query.

---

### 2. Core Self-Reflection Questions
Self-RAG introduces **self-reflection** mechanisms that continuously evaluate four key questions throughout processing:
1. **Is Retrieval Necessary?** Evaluates whether external document retrieval is needed or if the query can be answered directly using general LLM knowledge.
2. **Are Documents Relevant?** Filters out noise by verifying whether each retrieved document contains useful context to answer the question.
3. **Is the Response Grounded?** Evaluates whether every factual claim in the generated answer is backed by retrieved evidence, categorizing answers into *Fully Supported*, *Partially Supported*, or *No Support*.
4. **Is the Response Useful?** Verifies whether the final grounded response actually answers the user's intent.

---

### 3. Architecture & Implementation
The tutorial builds the Self-RAG graph step-by-step using **LangGraph** and **Pydantic** structured outputs:
* **Adaptive Routing**: Directs basic queries to immediate LLM generation while sending factual/company-specific queries to vector search.
* **Relevance Filtering**: Evaluates each retrieved document individually and drops non-relevant chunks.
* **Hallucination Control (`is_supported`) & Revision**: Evaluates grounding. If claims are partially supported or ungrounded, a strict revision node re-writes the answer to strip out fabricated details until it becomes fully supported.
* **Query Rewriting (`is_use`)**: If a fully supported answer is still not useful, the system rewrites the search query with high-signal keywords, re-runs retrieval, and retries up to a set loop limit.

---
I have generated a **Study Guide report** on the Self-RAG architecture in your Studio panel, detailing its components, state schema, and decision flows.

Here is a deep-dive explanation of how the two key self-reflection nodes—the **Hallucination Checker** and the **Query Rewriter**—operate within the LangGraph pipeline:

---

### 1. The Hallucination Checker (`is_supported`) & Revision Node

The **`is_supported`** node acts as a strict verification layer that prevents the LLM from outputting fabricated details or facts not present in retrieved context.

* **Evaluating Grounding**: The node receives the original user question, the generated candidate answer, and the extracted context from relevant documents. It evaluates every factual claim in the answer against the retrieved evidence.
* **Three Output Classifications**:
  1. **Fully Supported**: Every single claim and figure in the answer is directly backed by the retrieved context.
  2. **Partially Supported**: Some claims match the context, but the LLM injected extra unevidenced facts from its parametric memory (e.g. adding unmentioned side effects or policy details).
  3. **No Support**: The LLM hallucinated the entire answer because the retrieved context contained no matching data.
* **The Revision Loop (`revise_answer`)**:
  * If the verdict is *Partially Supported* or *No Support*, the pipeline routes execution to a **`revise_answer`** node.
  * This node uses a strict system prompt instructing the LLM to strip out any unevidenced statements and rewrite the response strictly bounded by the provided context.
  * The revised output is fed back into `is_supported` to re-verify grounding. A maximum retry counter (e.g. 5 retries) prevents infinite looping.

---

### 2. The Query Rewriter (`rewrite_question`) & Usefulness Loop (`is_use`)

Even if an answer contains zero hallucinations (*Fully Supported*), it might still fail to address the user's intent if the LLM picked a narrow fact or ignored the core prompt.

* **Usefulness Evaluation (`is_use`)**: After passing the hallucination check, the answer is evaluated for user utility. The node checks whether the generated response directly and satisfactorily answers the initial question.
* **Query Rewriting**:
  * If the answer is classified as *Not Useful*, control flows to the **`rewrite_question`** node.
  * This node reformulates the user's original question into a search query optimized for vector store retrieval by stripping conversational filler, preserving key entities, and appending 2 to 5 high-signal keywords.
* **Adaptive Search Loop**:
  * The rewritten query is sent back to the vector retriever to fetch a fresh batch of document chunks.
  * The pipeline restarts the relevance filtering, generation, grounding check, and usefulness loop with the new context.
  * If the rewrite count exceeds the maximum limit without yielding a useful answer, the system gracefully terminates with a *"No Answer Found"* response.

---

**A step-by-step implementation of Self-RAG (Self-Reflective Retrieval-Augmented Generation) using LangGraph, where the LLM evaluates and fact-checks its own retrieval, evidence, and answers.**

Standard RAG fetches documents and trusts them blindly. Self-RAG adds continuous **self-reflection**: the model decides whether retrieval is needed, filters out irrelevant chunks, checks that its answer is grounded in the evidence, and rewrites the query when the answer isn't useful. The result is a conversational agent that applies quality control to its own outputs.

This repo builds the pipeline in seven notebooks, one capability at a time, plus a streamlined web-search notebook and Streamlit frontend.

---

## Why Self-RAG?

Traditional RAG pipelines have three main weaknesses:

| Problem | What goes wrong |
|---|---|
| **Indiscriminate retrieval** | Documents are fetched for every query, even simple ones the LLM already knows, wasting compute and adding noise. |
| **Blind trust** | Retrieved chunks are used based on semantic similarity alone, even when they don't actually answer the question. |
| **No self-verification** | The answer is returned without checking for hallucinations or whether it resolves the user's intent. |

Self-RAG fixes each of these by making the LLM judge its own work at every stage.

## The Four Self-Reflection Questions

Throughout the pipeline, the system asks four questions:

1. **Is retrieval necessary?** Answer directly from the LLM's knowledge, or search the vector store?
2. **Are the retrieved documents relevant?** Each chunk is judged individually, and irrelevant ones are dropped.
3. **Is the response grounded?** Every claim is checked against the evidence and labelled *Fully Supported*, *Partially Supported*, or *No Support*.
4. **Is the response useful?** Does the grounded answer actually address what the user asked?

## Architecture

```
          [ User Query ]
                │
                ▼
    ┌───────────────────────┐
    │   Adaptive Routing    │
    │ (Retrieval needed?)   │
    └───────────┬───────────┘
       No │           │ Yes
          ▼           ▼
┌───────────┐    ┌────────────────────────┐
│ Direct    │    │     Vector Search      │
│ LLM Reply │    └───────────┬────────────┘
└─────┬─────┘                │
      │                      ▼
      │          ┌────────────────────────┐
      │          │  Relevance Filtering   │
      │          │ (Are chunks relevant?) │
      │          └───────────┬────────────┘
      │                      ▼
      │          ┌────────────────────────┐
      │          │   Answer Generation    │
      │          └───────────┬────────────┘
      │                      ▼
      │          ┌────────────────────────┐  Partially supported /
      │          │  Hallucination Check   ├─────── No support ───┐
      │          │     (is_supported)     │                      ▼
      │          └───────────┬────────────┘          ┌───────────────────┐
      │       Fully          │                       │   Revise Answer   │
      │     supported        │ ◄─────────────────────┤ (strip ungrounded)│
      │                      ▼        retry loop     └───────────────────┘
      │          ┌────────────────────────┐
      │          │    Usefulness Check    │
      │          │        (is_use)        │
      │          └───────────┬────────────┘
      │            Useful │     │ Not useful
      │                   │     ▼
      │                   │  ┌───────────────────┐
      │                   │  │ Rewrite Question  │──► re-search vector store
      │                   │  └───────────────────┘
      │                   ▼
      └───────────────► [ Final Grounded Output ]
```

## Key Nodes Explained

### Adaptive Routing
Sends simple, general-knowledge questions straight to the LLM, and routes factual or document-specific questions to vector search.

### Relevance Filtering
Grades each retrieved chunk on its own and keeps only the ones that contain useful context.

### Hallucination Checker (`is_supported`) and Revision (`revise_answer`)
Compares the candidate answer against the retrieved context claim by claim.

- **Fully Supported:** every claim is backed by the context.
- **Partially Supported:** some claims match, but the LLM added facts from its own memory.
- **No Support:** the answer isn't grounded in the context at all.

If the verdict is partial or none, `revise_answer` rewrites the answer using only the provided context, and the result is re-checked. A retry limit prevents infinite loops.

### Usefulness Check (`is_use`) and Query Rewriter (`rewrite_question`)
An answer can be fully grounded and still miss the point. If `is_use` marks it *not useful*, `rewrite_question` reformulates the query for better retrieval (dropping filler, keeping key entities, adding a few high-signal keywords). Retrieval then runs again with the new query. After a maximum number of rewrites, the system ends gracefully with a "No Answer Found" response.

## Learning Path

| Notebook | What it covers |
|---|---|
| `1_step1_self_rag.ipynb` | Foundation: documents, retrieval, and basic generation |
| `2_step2_self_rag.ipynb` | Next layer of the Self-RAG graph |
| `3_step3_self_rag.ipynb` | Next layer of the Self-RAG graph |
| `4_step4_self_rag.ipynb` | Next layer of the Self-RAG graph |
| `5_step5_self_rag.ipynb` | Next layer of the Self-RAG graph |
| `6_step6_self_rag.ipynb` | Next layer of the Self-RAG graph |
| `7_step7_self_rag.ipynb` | Final, complete Self-RAG graph |
| `SelfRAG_web.ipynb` | Web version of the pipeline |

<!-- VERIFY: replace the repeated rows with one line per step, e.g.
     step 2 = adaptive routing, step 3 = relevance filtering, step 4 = answer generation,
     step 5 = is_supported + revise_answer, step 6 = is_use + rewrite_question,
     step 7 = full graph with retry limits. Also clarify what "web" means in SelfRAG_web. -->

Run them in order; each notebook builds on the previous one.

## Project Structure

```
Self_RAG/
├── documents/              # Source documents for the knowledge base
├── 1_step1_self_rag.ipynb  # Step-by-step build of the Self-RAG graph
├── 2_step2_self_rag.ipynb
├── 3_step3_self_rag.ipynb
├── 4_step4_self_rag.ipynb
├── 5_step5_self_rag.ipynb
├── 6_step6_self_rag.ipynb
├── 7_step7_self_rag.ipynb
├── SelfRAG_web.ipynb       # Web version
├── env_template            # Template for your API keys
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Orchestration | LangGraph, LangChain |
| Structured outputs | Pydantic (graders return typed verdicts) |
| LLM | <!-- VERIFY: e.g. OpenAI / Groq --> |
| Embeddings and vector store | <!-- VERIFY: e.g. FAISS / Chroma --> |
| Frontend | Streamlit |
| Environment | Jupyter Notebook |

## Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab (or VS Code with the Jupyter extension)
- An API key for your LLM provider

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SyedaNaziaGit/Self_RAG.git
cd Self_RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langgraph langchain langchain-community pydantic python-dotenv streamlit jupyter
```

<!-- VERIFY: add your LLM, embedding, and vector store packages
     (e.g. langchain-openai, faiss-cpu, pypdf), or add a requirements.txt -->

### 4. Configure environment variables

Copy the template and add your keys:

```bash
cp env_template .env
```

On Windows, use `copy env_template .env`. Then open `.env` and fill in your values.

>  Never commit your `.env` file. Make sure it is listed in `.gitignore`.

### 5. Add your documents

Place the files you want to query in the `documents/` folder.

## Running the Project

**Notebooks**

```bash
jupyter notebook
```

Open the notebooks from `1_step1_self_rag.ipynb` through `7_step7_self_rag.ipynb` and run each cell top to bottom.

**Streamlit frontend**

```bash
streamlit run <your_app_file>.py
```

<!-- VERIFY: replace <your_app_file>.py with the real filename, or remove this section
     if the Streamlit app isn't in the repo yet -->

## References

- Asai et al., [*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*](https://arxiv.org/abs/2310.11511)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

---
