<<<<<<< HEAD
import json

from src.write_db import build_data, update_db


def generate_data_json():
    from src.search_semantic_scholar import (
        get_high_relevance_papers,
        get_author_info
    )
    from src.get_topics import (
        get_papers_embeddings,
        cluster_embeddings_two_level_balanced,
        get_cluster_topics
    )

    papers_dict, authors_dict = get_high_relevance_papers(
        top_per_month=200,
        num_months=12,
        fields_of_study="Computer Science"
    )
    authors_dict = get_author_info(authors_dict, min_paper_cnt=2)

    embd_model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    cluster_naming_model = "Qwen/Qwen2.5-7B-Instruct"

    embds = get_papers_embeddings(papers_dict, embd_model_name, batch_size=32, max_length=8192)
    supercluster_to_clusters, cluster_to_ids = cluster_embeddings_two_level_balanced(
        embds,
        n_clusters=144,
        n_superclusters=12,
        random_state=42
    )
    super_topics, cluster_topics = get_cluster_topics(
        supercluster_to_clusters,
        cluster_to_ids,
        papers_dict,
        cluster_naming_model=cluster_naming_model,
        sample_size=8,
        num_rounds=5
    )

    topics, researchers, papers, works_in = build_data(
        papers_dict,
        authors_dict,
        supercluster_to_clusters,
        cluster_to_ids,
        super_topics,
        cluster_topics
    )

    data = {
        "topics": topics,
        "researchers": researchers,
        "papers": papers,
        "works_in": works_in
    }
    with open("./data.json", "w") as f:
        json.dump(data, f)
    return data
=======
from argparse import ArgumentParser

from datetime import date
import datetime

from src.extract_scholar_info import get_high_relevance_papers, fill_author_info
from src.extract_topic_info import run_topics_classification
from src.extract_researcher_links import run_researcher_info_extraction
from src.data_models import (
    load_scholar_info_from_file, write_scholar_info,
    load_researcher_links_from_file, write_researcher_links,
    ScholarInfo
)
>>>>>>> agentic_classification


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

<<<<<<< HEAD
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--get_data",
        action="store_true",
        help="Get data from Semantic Scholar and generate data.json"
    )
    parser.add_argument(
        "--update_db",
        action="store_true",
        help="Update the database with data.json"
    )
    args = parser.parse_args()

    if args.get_data:
        generate_data_json()

    if args.update_db:
        with open("/app/data/data.json", "r") as f:
            data = json.load(f)
        update_db(data)
=======
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
        import pdb ; pdb.set_trace()
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
    rll = run_researcher_info_extraction(rll, scholar_info, max_update=20)
    write_researcher_links(rll)
>>>>>>> agentic_classification
