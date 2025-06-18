# 🔍 AI Research Explorer

A web app for exploring AI research areas, viewing topic hierarchies, and finding researchers working in various fields.

### 🚀 Features

- View hierarchical AI research topics, courtesy of gpt-4.1-mini.
- Find papers and researchers relevant to each topic.
- Expand a researcher's profile to see all the topics they are working in.
- Clickable links to get you on your way!

### 🛠️ Tech Stack

Python, LangGraph, React/TypeScript, Node.js/Express, PostgreSQL, Docker.

### 🤖 Usage

To fetch data,

```bash

cd data_pipeline
# Create new environment via uv, pip, conda, etc.
pip install -r requirements-build.txt
python -m src.main \
    --force_paper_ingest            # Re-ingest paper data even if already processed.         (default: False)
    --force_author_ingest           # Re-ingest author data even if already processed.        (default: False)
    --top_per_month 50              # Number of top papers to ingest per month.              (default: 50)
    --num_months 12                 # How many past months of data to consider.              (default: 12)
    --max_researcher_updates 20     # Maximum number of researcher links to update.          (default: 20)
```


To start the app,

```bash

docker-compose up --build
```
