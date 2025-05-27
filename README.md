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
python -m src.main
```


To view data,

```bash

docker-compose up --build
```
