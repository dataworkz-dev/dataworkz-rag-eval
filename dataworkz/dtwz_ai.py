import time
from .dataworkz_api import DataworkzAPI
from continuous_eval.metrics.retrieval import (
    PrecisionRecallF1,
    RankedRetrievalMetrics,
    ContextPrecision,
)
from continuous_eval.metrics.generation.text import (
    DeterministicAnswerCorrectness,
    DeterministicFaithfulness,
    DebertaAnswerScores,
    BertAnswerSimilarity,
    BertAnswerRelevance
)

import logging
logger = logging.getLogger()


class AIDtwz:
    dtwz_client: DataworkzAPI
    system_id: str
    llm_provider_id: str
    answer: str
    context: str
    response: dict
    llm_eval: bool
    retries: int = 15
    retry: bool = True

    def __init__(self, llm_eval=False):
        self.dtwz_client = DataworkzAPI()
        self.llm_eval = llm_eval
        return

    def find_key_by_value(self, json_obj, target_value, current_path=""):
        """
        Recursively searches for the full path of the key in a nested JSON
        object where the value matches the target_value.

        :param json_obj: The JSON object (can be a dictionary or list).
        :param target_value: The value to search for.
        :param current_path: The current nested path being explored.
        :return: The full path of the key as a string, or None if not found.
        """
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if value == target_value:
                    return json_obj["data"]["Input"]
                elif isinstance(value, (dict, list)):
                    found = self.find_key_by_value(value, target_value, new_path)
                    if found:
                        return found
        elif isinstance(json_obj, list):
            for index, item in enumerate(json_obj):
                new_path = f"{current_path}[{index}]"
                found = self.find_key_by_value(item, target_value, new_path)
                if found:
                    return found
        return None

    def set_system_id(self, system_id: str):
        self.system_id = system_id
        return

    def set_llm_provider_id(self, llm_provider_id: str):
        self.llm_provider_id = llm_provider_id
        return

    def get_qna_systems(self):
        return self.dtwz_client.get_qna_systems()

    def get_qna_system_details(self):
        return self.dtwz_client.get_system_details(self.system_id)

    def get_llm_provider_details(self):
        return self.dtwz_client.get_llm_providers(self.system_id)

    def get_chunks(self, query: str):
        while self.retry:
            self.response = self.dtwz_client.get_answer(
                self.system_id,
                query,
                self.llm_provider_id,
                properties="include_probe=true",
            )
            if self.response is not None:
                self.answer = self.response["answer"]
                self.context = self.response['context']
                return self.find_key_by_value(
                    self.response["probe"], "MERGE_NEIGHBOURING_CONTEXT"
                )
            if self.retries == 0:
                logger.error(f'Failed to get a response after 10 attempts.')
                return None
            logger.debug("Retrying...")
            time.sleep(10)
            self.retries -= 1
    
    def get_answer(self):
        return self.answer
    
    def get_search(self, query: str):
        self.response = self.dtwz_client.get_search(self.system_id, query)
        return self.response

    async def score_retrieval(self, query: str, gt_answer: str, chunks: list[dict]):
        retrieved_chunks = []
        for chunk in chunks:
            retrieved_chunks.append(chunk["Content"])
        datum = {
            "question": query,
            "retrieved_context": retrieved_chunks,
            "ground_truth_context": gt_answer,
        }
        metrics = PrecisionRecallF1()
        ranked_metrics = RankedRetrievalMetrics()
        overall_metrics = (metrics(**datum)) | (ranked_metrics(**datum))
        # Adding context precision to the metrics computed via LLM based method.
        # Needs the OpenAI API key set as an environ variable
        if self.llm_eval:
            llm_context_precision = ContextPrecision()
            llm_context_precision_metrics = llm_context_precision(**datum)
            # Renaming LLM metrics to reflect they are from the LLM based method.
            for key in llm_context_precision_metrics.keys():
                new_key = f"LLM_{key}"
                overall_metrics[new_key] = llm_context_precision_metrics[key]
        return overall_metrics

    # placeholder to compute deterministic metrics for the final answer
    def score_system(self, query: str, gt_answer: str):
        retrieved_context = []
        for ctx in self.context:
            retrieved_context.append(ctx["data"])
        datum = {
            "question": query,
            "answer": self.answer,
            "ground_truth_answers": gt_answer,
            "ground_truth_context": gt_answer,
            "retrieved_context": retrieved_context,
        }
        correctness_metric = DeterministicAnswerCorrectness()
        score = correctness_metric(**datum)
        faithfulness_metric = DeterministicFaithfulness()
        score |= faithfulness_metric(**datum)
        deberta_metric = DebertaAnswerScores()
        score |= deberta_metric(**datum)
        bert_similarity_metrics = BertAnswerSimilarity()
        score |= bert_similarity_metrics(**datum)
        datum = {
            "question": query,
            "answer": self.answer
        }
        bert_relevance_metrics = BertAnswerRelevance()
        score |= bert_relevance_metrics(**datum)
        return score
