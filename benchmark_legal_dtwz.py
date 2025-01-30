import numpy as np
from dataworkz.dtwz_ai import AIDtwz

import asyncio
import datetime as dt
import os
import time

import pandas as pd

import logging

# Create and configure logger
log_file_name = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(
    filename=log_file_name + ".log", format="%(asctime)s %(message)s", filemode="w"
)

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

benchmark_name_to_qna: dict[str, str] = {
    "privacy_qa": [
        {
            "expt_name": "privacy_qna_rerank",
            "uuid": "04809ddf-e395-4384-97f2-a80cfb1e2813",
        },
        {
            "expt_name": "privacy_qna_rerank",
            "uuid": "be0c53c8-33fe-4da9-ad6e-897d2f9c5fc1",
        },
    ],
    "contractnli": [
        {
            "expt_name": "contractnli_qna_v1",
            "uuid": "bafd8ef9-f535-489a-9d37-1940f5a12b10",
        },
        {
            "expt_name": "contractnli_qna_rerank",
            "uuid": "87c1eb00-fe80-4012-b8b8-82afd3181d12",
        },
    ],
    "maud": [
        {"expt_name": "maud_qna_v1", "uuid": "04d1b00f-1739-486b-a3eb-a3fcb4c07e2a"},
        {
            "expt_name": "maud_qna_rerank",
            "uuid": "fc0f4993-b461-4b26-b1fb-ce26ce55ef57",
        },
    ],
    "cuad": [
        {
            "expt_name": "cuad_qna_rerank",
            "uuid": "8a7bed94-bb5c-42bf-8244-eaec6cdeaed4",
        },
        {"expt_name": "cuad_qna_v2", "uuid": "c8710d96-ec61-43a1-aa40-666f4d579afc"},
    ],
}

# enable answer based deterministic metrics
ANSWER_METRICS = False


async def main() -> None:
    qa_data = pd.read_csv("./data/legalbench_qa_data.csv")
    qa_data = qa_data.head(1)
    dtwz_ai_client = AIDtwz(True)
    dtwz_ai_client.set_llm_provider_id("599fc5b5-551b-452e-825b-970d2cfe68fe")
    results = []
    for row in qa_data.itertuples():
        question = row.question
        gt_answer = row.gt_answer
        source = row.source
        for system in benchmark_name_to_qna[row.source]:
            expt_name = system["expt_name"]
            uuid = system["uuid"]
            dtwz_ai_client.set_system_id(uuid)
            debug_str = f"INFO: running {source} experiment: {expt_name} on question:{question}..."
            logger.debug(debug_str)
            # get the merge input chunks from DTWZ system response, which is used to score retrieval later on.
            logger.debug(f"INFO: fetching answer...")
            merge_input_chunks = dtwz_ai_client.get_chunks(question)
            if merge_input_chunks is None:
                logger.error(f"ERROR: could not fetch answer data")
                exit()
            # compute metrics on the retrieved answer against ground truth answer.
            logger.debug(f"INFO: computing metrics...")
            scores = await dtwz_ai_client.score_retrieval(
                question, gt_answer, merge_input_chunks
            )
            if ANSWER_METRICS:
                answer_scores = dtwz_ai_client.score_system(question, gt_answer)
                scores = scores | answer_scores
            # append the results to a list of dictionaries.
            results.append(
                {
                    "question": question,
                    "gt_answer": gt_answer,
                    "system_answer": dtwz_ai_client.get_answer(),
                    "source": source,
                    "experiment_name": expt_name,
                }
                | scores
            )
        # adding in sleep to throttle the requests to avoid rate limiting by DTWZ AI API
        time.sleep(5)
    qa_results = pd.DataFrame(results)
    # Create a save location for this run
    run_name = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_path = f"./benchmark_results/{run_name}_DTWZ"
    os.makedirs(benchmark_path, exist_ok=True)
    output_file = f"{benchmark_path}/results_dtwz.csv"
    # drop unnecessary columns for stats generation and save the results to a csv file
    if ANSWER_METRICS:
        columns = [
            "rouge_l_recall",
            "rouge_l_precision",
            "rouge_l_f1",
            "token_overlap_recall",
            "token_overlap_precision",
            "token_overlap_f1",
            "token_overlap_faithfulness",
            "rouge_p_by_sentence",
            "token_overlap_p_by_sentence",
            "bleu_score_by_sentence",
        ]
        qa_results.drop(columns, axis=1, inplace=True)
    qa_results.to_csv(output_file, index=False)

    logger.debug(f"INFO: generating stats")
    # Drop the question and gt_answer columns
    df = qa_results.drop(["question", "gt_answer", "system_answer"], axis=1)
    df_columns = df.columns  # Get a list of all column names in DataFrame
    remove_columns = np.array(["source", "experiment_name"])
    # Remove unwanted columns from list of column names
    df_columns = np.setdiff1d(df_columns, remove_columns)
    # Create a Pivot Table with the Average of Numeric Value Columns
    pivot_table = df.groupby(["source", "experiment_name"]).mean()[df_columns]

    stats_df = pivot_table.reset_index()
    stats_file = f"{benchmark_path}/stats_dtwz.csv"
    stats_df.to_csv(stats_file, index=False)


if __name__ == "__main__":
    asyncio.run(main())
