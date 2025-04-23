# RAG Multi-Agent System

A CrewAI-based multi-agent system for Retrieval-Augmented Generation (RAG) with Query Builder, Retriever, Reviewer, and Decider agents.

## Setup

1. **Clone the Repository**

   ```bash
   git clone <your-repo-url>
   cd rag-multi-agent-system
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Sample Data**

   - Place `.csv`, `.md`, or `.json` files in the `data/` directory.
   - Example files: `sample.csv`, `sample.md`, `sample.json`.

5. **Run the System**

   ```bash
   python src/main.py
   ```

## Project Structure

- `src/`: Source code (`main.py`, `rag_toolkit.py`).
- `config/`: Configuration files (`crew.yaml`).
- `data/`: Input data files.
- `tests/`: Unit tests (to be added).

## Usage

- The system processes a user query through four agents:
  1. **Query Builder**: Generates query variants.
  2. **Retriever**: Fetches relevant document chunks from ChromaDB.
  3. **Reviewer**: Filters chunks for relevance.
  4. **Decider**: Selects the best answer.
- Output is printed to the console (to be enhanced in future phases).

## Next Steps

- Implement agent logic (Phase 3).
- Test RAG pipeline with sample data (Phase 4).
- Add summarizer agent and feedback loop (future enhancements).