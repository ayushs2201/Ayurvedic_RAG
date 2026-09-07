from pyserini.search import FaissSearcher
from pyserini import trectools
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.lucene import LuceneSearcher
import os

import pandas as pd

import argparse
import os
import pandas as pd
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher

# Disable SSL verification if required
os.environ.pop('SSL_CERT_FILE', None)

DATA_DIR = "../../data"
TOPICS = DATA_DIR + "/topics.csv"


def main():
    parser = argparse.ArgumentParser(description="Run Pyserini retrieval algorithms.")
    parser.add_argument(
        "--run",
        choices=["bm25", "dense-faiss"],
        required=True,
        help="Specify retrieval mode: 'bm25' or 'dense-faiss'",
    )
    args = parser.parse_args()

    # Shared Configuration
    num_hits = 100
    tag = f"walert.rag.{args.run}"
    topics = pd.read_csv(TOPICS)

    # Mode Selection
    if args.run == "bm25":
        index_path = "../../target/indexes/bm25"
        output_filename = "../../target/runs/rag-bm25.txt"
        print("Initializing Lucene BM25 Searcher...")
        searcher = LuceneSearcher(index_path)

    elif args.run == "dense-faiss":
        index_path = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
        query_encoder = "castorini/tct_colbert-v2-hnp-msmarco"
        output_filename = "../../target/runs/rag-dense-faiss.txt"
        print("Initializing FAISS Dense Searcher...")
        searcher = FaissSearcher(index_path, query_encoder)

    # Execute Search and Write Runs
    output_writer = get_output_writer(
        output_filename,
        OutputFormat("trec"),
        "w",
        max_hits=num_hits,
        tag=tag,
        topics=topics,
    )

    with output_writer:
        for question_id, question in topics[["question_id", "question"]].values:
            hits = searcher.search(str(question), num_hits)
            output_writer.write(str(question_id), hits)

    print(f"Done! Results written to {output_filename}")


if __name__ == "__main__":
    main()