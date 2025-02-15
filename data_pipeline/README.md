# Data pipeline

### Overview

This Python-based data pipeline populates a database consisting of papers, researchers, and topics.

### Paper Ingestion

This pipeline ingests ArXiV data and uses an LLM to hypothesize topics and trends.

```bash

python -m core.pipeline \
    --author-tracking-period-months 24 \  # Months to track author activity  
    --author-num-papers-enter-threshold 7 \  # Papers required to start tracking an author  
    --author-num-papers-keep-threshold 0 \  # Papers required to keep tracking an author  
    --paper-tracking-period-months 2 \  # Months to track paper activity  
    --temperature 0.0 \  # LLM randomness control (higher = more creative)  
    --max-completion-tokens 40 \  # Max tokens in LLM output  
    --limit 1000 \  # Max papers classified per call  
    --sample-freq 50 \  # Frequency of paper classification sampling  
    --new-topic-threshold 3  # Threshold to define a new research topic  
```

### Author Search

This pipeline automates the extraction of researcher profile links and affiliations from Google search results.

```bash

python -m core.extract_researcher_links \
    --model gpt-4o-mini \  # The LLM model to use  
    --temperature 0.0 \  # Sampling temperature for the LLM (higher = more randomness)  
    --max_completion_tokens 75 \  # Max tokens for the LLM response  
    --max_results 5 \  # Maximum number of search results to process  
    --limit 1  # Limit on the number of researcher links to update  
```
