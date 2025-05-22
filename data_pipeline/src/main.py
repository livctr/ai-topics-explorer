from data_pipeline.src.extract_scholar_info import ingest_scholar_info
from data_pipeline.src.extract_topic_info import run_agentic_classification


if __name__ == "__main__":
    scholar_info = ingest_scholar_info(
        top_per_month=50,
        num_months=1,
    )
    run_agentic_classification(scholar_info)
