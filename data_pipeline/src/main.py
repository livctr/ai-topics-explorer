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


if __name__ == "__main__":

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
