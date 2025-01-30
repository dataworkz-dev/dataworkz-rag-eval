"""
Package that incorporates Class definition needed to access the DataworkZ API
"""

import os
import urllib.parse

import requests
import logging

logger = logging.getLogger()


class DataworkzAPI:
    """
    Class definition for the Dataworkz API package
    """

    def __init__(
        self, token_var="DATAWORKZ_API_TOKEN", service_url="DATAWORKZ_SERVICE_URL"
    ):
        self.token: str = str(os.getenv(token_var))
        if self.token == "":
            raise RuntimeError("ERROR: DATAWORKZ_API_TOKEN is not set")
        self.service_url: str = str(os.getenv(service_url))
        if self.service_url == "":
            raise RuntimeError("ERROR: DATAWORKZ_SERVICE_URL is not set")
        self.authorization_header: dict = {
            "Content-Type": "application/json",
            "Authorization": "SSWS " + self.token,
        }

    def get_response(self, uri: str, auth_header: dict, timeout: int):
        """
        Function to process URI requests and handle exceptions. It has a timeout for each request
        The response is None if an error occurs
        """
        try:
            response = requests.get(uri, headers=auth_header, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"ERROR: HTTP Error occured: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"ERROR: Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"ERROR: Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"ERROR: Request error occurred: {req_err}")
        return None

    def get_qna_systems(self):
        """
        function call for getting all the QNZ systems or RAG apps configured in the Dataworkz system
        returns the JSON string response with key being uuid of RAG app and value being name of the
        app raises an exception if api is not responsive
        """
        api_url = self.service_url + "/api/qna/v1/systems"
        response = self.get_response(api_url, self.authorization_header, 300)
        if response is None:
            raise RuntimeError("ERROR: No QNA system details found")
        return response

    def get_system_details(self, system_id: str):
        """
        function call for getting all the QNA system  details on the system
        for the given system_id
        returns JSON of the LLM providers
        raises an exception if api is not responsive
        """
        api_url = self.service_url + "/api/qna/v1/systems/" + system_id
        response = self.get_response(api_url, self.authorization_header, 300)
        if response is None:
            raise RuntimeError("ERROR: No QNA system details found")
        return response

    def get_llm_providers(self, system_id: str):
        """
        function call for getting all the LLM providers configured on the system
        for the given system_id
        returns JSON of the LLM providers
        raises an exception if api is not responsive
        """
        api_url = (
            self.service_url + "/api/qna/v1/systems/" + system_id + "/llm-providers"
        )
        response = self.get_response(api_url, self.authorization_header, 300)
        if response is None:
            raise RuntimeError("ERROR: No LLM provider details found")
        return response

    def get_filters(self, system_id: str):
        """
        function call for getting all the filters configured on the system
        for the given system_id
        returns JSON of the filters for a particulat QNA
        raises an exception if api is not responsive
        """
        api_url = self.service_url + "/api/qna/v1/systems/" + system_id + "/filters"
        response = self.get_response(api_url, self.authorization_header, 300)
        if response is None:
            raise RuntimeError("ERROR: No filter details found")
        return response

    def get_qna_history(self, system_id: str):
        """
        function call for getting the question history for 6 months
        on the system for the given system_id
        returns JSON of the question history for a particulat QNA
        raises an exception if api is not responsive
        """
        api_url = (
            self.service_url + "/api/qna/v1/systems/" + system_id + "/questionshistory"
        )
        response = self.get_response(api_url, self.authorization_header, 300)
        if response is None:
            raise RuntimeError("ERROR: No question history details found")
        return response

    def get_answer(
        self,
        system_id: str,
        query: str,
        llm_provider_id: str,
        results_filter: str | None = None,
        properties: str | None = None,
    ):
        """
        function to get the answer from the RAG app identified by system_id
        using the LLM specified by llm_provider_id. Additionally, filter can be
        used to filter the results pre-defined in the RAG app. properties is used
        to set flags like include_probe, context
        returns the json response containing the LLM answer and the probe data by default
        raises an exception in case of HTTP errors
        """
        api_url = (
            self.service_url
            + "/api/qna/v1/systems/"
            + system_id
            + "/answer?questionText="
        )
        # write a line of code to encode a string into a UTF-8 for use in a URL
        query_encoded = urllib.parse.quote(query)
        api_url = api_url + query_encoded + "&llmProviderId=" + llm_provider_id
        if results_filter is not None:
            api_url = api_url + "&filter=" + results_filter
        if properties is not None:
            api_url = api_url + "&properties=" + properties
        response = self.get_response(api_url, self.authorization_header, 2400)
        if response is None:
            logger.error(f"ERROR: Answer system unavailable")
            return None
        return response

    def get_search(self, system_id: str, query: str):
        """
        function to get the retrieved chunks from the RAG app identified by system_id
        returns the json response containing the LLM answer and the probe data by default
        raises an exception in case of HTTP errors
        """
        api_url = (
            self.service_url + "/api/qna/v1/systems/" + system_id + "/search?query="
        )
        # write a line of code to encode a string into a UTF-8 for use in a URL
        query_encoded = urllib.parse.quote(query)
        api_url = api_url + query_encoded
        response = self.get_response(api_url, self.authorization_header, 2400)
        if response is None:
            logger.error(f"ERROR: Retrieval system unavailable")
            return None
        return response

    def get_question_details(self, system_id: str, question_id: str):
        """
        function to get the previous question details from the RAG app identified by system_id
        returns the json response containing the LLM answer and the probe data by default
        raises an exception in case of HTTP errors
        """
        api_url = (
            self.service_url
            + "/api/qna/v1/systems/"
            + system_id
            + "/questions/"
            + question_id
        )
        response = self.get_response(api_url, self.authorization_header, 2400)
        if response is None:
            raise RuntimeError("ERROR: Previous questions unavailable")
        return response
