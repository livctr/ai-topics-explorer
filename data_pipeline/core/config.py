import os

class DataPaths:

    BASE_DIR = "./data"
    LOG_DIR = "./logs"

    PROGRESS_LOG_PATH = os.path.join(LOG_DIR, 'progress_log.tmp')

    CONFERENCE_DIR = os.path.join(BASE_DIR, 'conference')
    AUTHORS_PATH = os.path.join(CONFERENCE_DIR, 'authors.txt')

    PROF_DIR = os.path.join(BASE_DIR, 'professor')
    SEARCH_RESULTS_PATH = os.path.join(PROF_DIR, 'search_results.json')
    US_PROF_PATH = os.path.join(PROF_DIR, 'us_professor.json')
    NOT_US_PROF_PATH = os.path.join(PROF_DIR, 'not_us_professor.json')
    PROMPT_DATA_PREFIX = str(os.path.join(PROF_DIR, 'prompt_data'))

    ARXIV_FNAME = 'arxiv-metadata-oai-snapshot.json'
    ARXIV_PATH = os.path.join(BASE_DIR, ARXIV_FNAME)
    ML_ARXIV_PATH = os.path.join(BASE_DIR, 'arxiv-metadata-oai-snapshot-ml.json')

    PAPER_DIR = os.path.join(BASE_DIR, "paper_embeddings")
    EMBD_MODEL = "all-mpnet-base-v2-embds"
    EMBD_PATH = os.path.join(PAPER_DIR, EMBD_MODEL)
    PAPER_DATA_PATH = os.path.join(PAPER_DIR, "paper_data")

    FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend_data')
    FRONTEND_PROF_PATH = os.path.join(FRONTEND_DIR, 'us_professor.json')
    FRONTEND_EMBD_PATH = os.path.join(FRONTEND_DIR, EMBD_MODEL)  # contains id, title, author, weights
    FRONTEND_ITA_PATH = os.path.join(FRONTEND_EMBD_PATH, 'ita.csv')
    FRONTEND_WEIGHTS_PATH = os.path.join(FRONTEND_EMBD_PATH, 'weights.pt')

    # create BASE_DIR LOG_DIR FRONTEND_DIR PROF_DIR CONFERENCE_DIR PAPER_DIR

    @staticmethod
    def ensure_directories():
        # create BASE_DIR LOG_DIR FRONTEND_DIR PROF_DIR CONFERENCE_DIR PAPER_DIR
        for directory in [DataPaths.BASE_DIR,
                          DataPaths.LOG_DIR,
                          DataPaths.FRONTEND_DIR,
                          DataPaths.PROF_DIR,
                          DataPaths.CONFERENCE_DIR,
                          DataPaths.PAPER_DIR]:
            os.makedirs(directory, exist_ok=True)
