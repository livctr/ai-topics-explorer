from argparse import ArgumentParser

from datetime import date

from data_pipeline.src.extract_scholar_info import ingest_scholar_info
from data_pipeline.src.extract_topic_info import run_agentic_classification
from data_pipeline.src.extract_researcher_links import run_researcher_info_extraction
from data_pipeline.src.data_models import (
    load_scholar_info_from_file, write_scholar_info,
    load_researcher_links_from_file, write_researcher_links
)


if __name__ == "__main__":
    parser = ArgumentParser(description="Ingest and process scholar information.")
    parser.add_argument(
        "--force_paper_ingest",
        action='store_true',
    )
    parser.add_argument(
        "--top_per_month",
        type=int,
        default=50,
        help="Number of top papers to ingest per month."
    )
    parser.add_argument(
        "--num_months",
        type=int,
        default=1,
        help="Number of months to consider for ingestion."
    )
    parser.add_argument(
        "--max_researcher_updates",
        type=int,
        default=1,
        help="Maximum number of researcher links to update."
    )
    args = parser.parse_args()

    scholar_info = load_scholar_info_from_file()
    last_ingestion_date = date.fromisoformat(scholar_info.date)
    today = date.today()
    is_outdated = (today - last_ingestion_date).days > 30
    if args.force_paper_ingest or is_outdated:
        scholar_info = ingest_scholar_info(
            top_per_month=args.top_per_month,
            num_months=args.num_months,
        )
        write_scholar_info(scholar_info)
        run_agentic_classification(scholar_info)

    # Always see if researcher links can be updated
    rll = load_researcher_links_from_file()
    rll = run_researcher_info_extraction(rll, scholar_info, max_update=1)
    write_researcher_links(rll)
