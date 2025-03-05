# README

This is a simple code repository that includes two Python scripts for running benchmarks on various RAG pipelines on the Dataworkz platform. We can use the tool to benchmark RAG pipelines by running it against a Golden Truth dataset and comparing the Retrieval strategies. 
## Files

1. `benchmark_*.py`: These scripts contain the main function which runs the benchmarking process. It imports necessary functions from other modules, applies them on data, evaluates performance of RAG pipelines, and outputs results in a table format into `benchmark_results` folder.
2. `dtwz_ai.py`: Contains pipeline evaluation functions  for evaluating any given pipeline, and `dataworkz_api.py` provides the wrappers around the Dataworz API.
3. `requirements.txt`: Contains a list of Python dependencies required for running these scripts.
4. `data/` directory: Contains the dataset used in this project. The data should be placed here and accessed by providing the relative path to it from benchmarking.py. 
5. `README.md`: This file you are reading right now. It provides an overview of the repository and its content.

## How to get API keys and experimentation details 
Use the following link to understand how to get the API keys - [Generate Dataworkz API Key](https://docs.dataworkz.com/product-docs/api/generate-api-key-in-dataworkz)
   
### Set Environment Variables 
1. set DATAWORKZ_API_KEY to the value above
2. set DATAWORKZ_SERVICE_URL to the url of your dataworkz service. e.g. https://ragapps.dataworkz.com 

### Getting QNA system details

Once the API Key and Service URL are available, use either of the methods below to get the QNA system details - 

#### Method 1: Using Python script
Run the `get_dtwz_detials.py` and follow the intructions on the screen, making note of the credentails that are needed for further steps. 
### OR
#### Method 2: Using Dataworkz API directly
Follow the steps below - 
1. Get the QNA systems configured - [Get QNA systems](https://docs.dataworkz.com/product-docs/api#qna-v1-systems)
2. Select the QNA system IDs for which you need to do the RAG Evaluation, and get the LLM Provider ID that you wish to use. [Get LLM Provider ID](https://docs.dataworkz.com/product-docs/api#qna-v1-systems-systemid-llm-providers)

Use the above details to modify the benchmarking to script, to either compare two or more pipelines, or even evaluate a single QNA RAG pipeline. Please refer to the example below

### example
```
benchmark_name_to_qna: dict[str, str] = {
    "name_of_the_source_column_in_qa_data_file": [
        {
            "expt_name": "experiment_name_1",
            "uuid": "uuid_of_experiment_from_previous_step",
        },
        {
            "expt_name": "experiment_name_2",
            "uuid": "uuid_of_experiment_from_previous_step",
        },
    ]
}

LLM_PROVIDER_ID = "llm_provider_id_from_previous_step"
```

## How to Run

1. Install Python dependencies using pip: 
   
   ```bash
   pip install -r requirements.txt
   ```
2. Place your data in the "data/" directory as instructed above. Make sure that the paths in benchamark_*.py correspond to your actual data location. and the data is in the format as specified by the headers, `["question", "gt_answer", "gt-context", "source"]`. The source is used in the benchmarking script to iterate over the alternative pipelines.
3. Run `benchmark_*.py` script:
   
    ```bash
    python benchmark_*.py
    ```
4. The results will be stored in the `benchmark_results` folder along with the timestamp of the run. 


Please note that running these scripts require a working Python environment along with necessary libraries installed (as specified in requirements.txt). Also, the data paths and other configurations are assumed based on the current file structure and might need adjustments depending on your specific setup.

## Troubleshooting 
* If running on a Mac with MPS enabled for pytorch, please set this environment variable to avoid any memory issues: ```PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0```
