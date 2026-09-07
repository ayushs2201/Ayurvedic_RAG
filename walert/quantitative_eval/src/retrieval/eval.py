from ranx import compare,Qrels,Run
import argparse
import sys
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_set', choices=['known', 'inferred'])
    parser.add_argument('qrel')
    parser.add_argument('runs', nargs='+', default=[]),
    args = parser.parse_args()
    
    alpha = 0.01
    topic_set = args.topic_set
    qrels = pd.read_csv(args.qrel, sep=r'\s+', names=['q_id', 'iter', 'doc_id', 'score'], header=None,dtype={'q_id': str, 'doc_id': str})

    # Filter based on topic set
    if topic_set == "known":
        # Extracts the topic number T<num> and filters T001 to T100
        selected_df = qrels[qrels['q_id'].apply(
            lambda x: x.startswith('T') and x[1:4].isdigit() and 1 <= int(x[1:4]) <= 100
        )]
    elif topic_set == "inferred":
        # Filters T101 and above
        selected_df = qrels[qrels['q_id'].apply(
            lambda x: x.startswith('T') and x[1:4].isdigit() and int(x[1:4]) > 100
        )]
    else:
        selected_df = qrels

    if selected_df.empty:
        print(f"Error: No qrels found for split '{topic_set}'.")
        print("Sample q_ids in file:", qrels['q_id'].head(5).tolist())
        sys.exit(1)
        

    runs = [Run.from_file(run, kind="trec") for run in args.runs]
    
    qrels = Qrels.from_df(selected_df,
                              q_id_col="q_id",
                              doc_id_col="doc_id",
                              score_col="score")


    
    report = compare( 
        qrels=qrels,
        runs=runs, 
        metrics=["ndcg@1","ndcg@3","ndcg@5"],
        max_p=alpha,  # P-value threshold
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4,  
    )

    print("{} Topics".format(topic_set))
    print(report)
    print(report.to_latex())



if __name__ == "__main__":
    sys.exit(main())