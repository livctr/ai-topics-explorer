from argparse import ArgumentParser

from datetime import date

from src.extract_scholar_info import get_high_relevance_papers, fill_author_info
from src.extract_topic_info import run_topics_classification
from src.extract_researcher_links import run_researcher_info_extraction
from src.data_models import (
    load_scholar_info_from_file, write_scholar_info,
    load_researcher_links_from_file, write_researcher_links,
    ScholarInfo
)


if __name__ == "__main__":
    parser = ArgumentParser(description="Ingest and process scholar information.")
    parser.add_argument(
        "--force_paper_ingest",
        action='store_true',
    )
    parser.add_argument(
        "--force_author_ingest",
        action='store_true',
        help="Force re-ingestion of author information."
    )
    parser.add_argument(
        "--force_topics_ingest",
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
        default=12,
        help="Number of months to consider for ingestion."
    )
    parser.add_argument(
        "--max_researcher_updates",
        type=int,
        default=20,
        help="Maximum number of researcher links to update."
    )
    args = parser.parse_args()

    scholar_info = load_scholar_info_from_file()
    last_ingestion_date = date.fromisoformat(scholar_info.date)
    today = date.today()
    is_outdated = (today - last_ingestion_date).days > 30

    # Ingest papers and researcher IDs
    if args.force_paper_ingest or is_outdated:
        papers_dict, authors_dict = get_high_relevance_papers(
            top_per_month=args.top_per_month,
            num_months=args.num_months
        )

        # Convert dictionaries to lists
        papers_list = list(papers_dict.values())
        researchers_list = list(authors_dict.values())

        scholar_info = ScholarInfo(
            date=today.isoformat(),
            papers=papers_list,
            researchers=researchers_list
        )
        write_scholar_info(scholar_info)

    # Ingest researcher fields and details
    if args.force_author_ingest or is_outdated:
        scholar_info = load_scholar_info_from_file()
        fill_author_info(scholar_info)  # modifies scholar_info.researchers in place
        write_scholar_info(scholar_info)

    # Run agentic classification to extract topics
    if args.force_topics_ingest or is_outdated:
        scholar_info = load_scholar_info_from_file()
        run_topics_classification(scholar_info)  # in place modification
        write_scholar_info(scholar_info)

    # Always see if researcher links can be updated
    rll = load_researcher_links_from_file()
    rll = run_researcher_info_extraction(rll, scholar_info, max_update=arg.max_researcher_updates)
    write_researcher_links(rll)
