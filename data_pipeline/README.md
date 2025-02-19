# Data pipeline

### Overview

This Python-based data pipeline populates a database consisting of papers, researchers, and topics.

### Paper Ingestion

This pipeline ingests ArXiV data and uses an LLM to hypothesize topics and trends.

```bash

python -m core.pipeline \
    --author_tracking_period_months 24 \  # Months to track author activity  
    --author_num_papers_enter_threshold 7 \  # Papers required to start tracking an author  
    --author_num_papers_keep_threshold 0 \  # Papers required to keep tracking an author  
    --paper_tracking_period_months 2 \  # Months to track paper activity
    --redownload \  # whether to redownload the data
    --cleanup_downloaded \  # whether to clean up the downloaded data (keep the data in case of using it again)
    --temperature 0.0 \  # LLM randomness control (higher = more creative)  
    --max_completion_tokens 40 \  # Max tokens in LLM output  
    --limit 1000 \  # Max papers classified per call  
    --sample_freq 50 \  # Frequency of paper classification sampling  
    --new_topic_threshold 3  # Threshold to define a new research topic  
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
