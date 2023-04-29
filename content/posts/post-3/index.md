---
title: "Supercharge SageMaker with Embeddings"
date: 2023-05-13T15:50:53-05:00
description: "Integrate SageMaker with a vector database"
draft: false
tags:
  - "sagemaker"
  - "langchain"
  - "deep lake"
  - "aws"
  - "activeloop"
---

## Introduction

Deep Lake serves as a vector database that can integrate with Amazon SageMaker, allowing the storage of embeddings which
are vector representations of data used in deep learning models. By leveraging Deep Lake's vector database capabilities,
developers can accelerate the training and deployment of their deep learning models. In this blog post, we will explore
the details of this integration and how the use of vector databases can enhance the performance and accuracy of deep
learning models.

## Getting Started

### Step 1: Installing required libraries and authenticating with Deep Lake and SageMaker

First, we will install everything we'll need.

```bash
!python3 -m pip install --upgrade langchain deeplake sagemaker tiktoken boto3
```

You'd need to authenticate into Deep Lake and AWS. You can get an API key from the Deep Lake
platform [here](https://app.activeloop.ai/)

````bash
!aws configure set aws_access_key_id [REDACTED]
!aws configure set aws_secret_access_key [REDACTED]
!aws configure set default.region us-east-1
!activeloop login -t [REDACTED]
!export ACTIVELOOP_USERNAME=[REDACTED]
````

Now you can proceed to deploy the SageMaker endpoint as usual:

```python
import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
model_name = "all-MiniLM-L6-v2"  # "mpt-7b-instruct" "dolly-v2-12b" "flan-t5-xxl" "all-MiniLM-L6-v2"
```

```python
from sagemaker.huggingface.model import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data=f"s3://{sess.default_bucket()}/{model_name}/model.tar.gz",
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    model_server_workers=1
)
```

```python
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.8xlarge",  # "ml.g5.4xlarge"
    endpoint_name=model_name,
    model_data_download_timeout=3600,
    container_startup_health_check_timeout=3600,
    update_endpoint=True
)
```

After the endpoint is deployed, you can use the following code to create a SageMaker endpoint embeddings object:

```python
import json
from typing import Dict, List

from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]


embeddings = SagemakerEndpointEmbeddings(
    endpoint_name="all-MiniLM-L6-v2",
    region_name="us-east-1",
    content_handler=ContentHandler()
)
```

### Step 2: Indexing Apache [Spark](https://github.com/apache/spark) Code Base

To index the code base, first clone the repository, parse the code, break it into chunks, and apply indexing:

````bash
!git clone --depth 1 https://github.com/apache/spark
````

Next, load all files inside the repository:

```python
import os
from langchain.document_loaders import TextLoader

root_dir = "spark"
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass
```

Subsequently, divide the loaded files into chunks:

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
```

Performing the indexing process roughly takes 4 minutes to calculate embeddings and upload them to Activeloop.

```python
import os
from langchain.vectorstores import DeepLake

username = os.getenv("[ACTIVELOOP_USERNAME]")
dataset_path = f"hub://{username}/spark"

db = DeepLake(dataset_path=f"hub://{username}/spark", embedding_function=embeddings)
db.add_documents(texts)
```

If the dataset has been already created, you can load it later without recomputing embeddings as seen below.

### Step 3: Conversational Retriever Chain

First, load the dataset, establish the retriever, and create the Conversational Chain:

```python
Dataset(path=f"hub://{username}/spark", read_only=True, tensors=["embedding", "ids", "metadata", "text"])
```

The Deep Lake dataset serving as a VectorStore has 4 tensors including the embedding, its ids, metadata including the
filename of the text, and the text itself. A preview of the dataset would look something like this:

| tensor    | htype   | shape         | dtype   | compression |
|-----------|---------|---------------|---------|-------------|
| embedding | generic | (23156, 1536) | float32 | None        |
| ids       | text    | (23156, 1)    | str     | None        |
| metadata  | json    | (23156, 1)    | str     | None        |
| text      | text    | (23156, 1)    | str     | None        |

```python
from langchain.vectorstores import DeepLake

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
```

```python
retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 10
```

Connect to the `LLM` for question answering.

```python
import json

from langchain import SagemakerEndpoint
from langchain.chains import RetrievalQA
from langchain.llms.sagemaker_endpoint import LLMContentHandler


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs) -> bytes:
        input_str = json.dumps({prompt: prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


model = SagemakerEndpoint(
    endpoint_name="all-MiniLM-L6-v2",
    region_name="us-east-1",
    credentials_profile_name="default",
    content_handler=ContentHandler(),
)

qa = RetrievalQA.from_llm(model, retriever=retriever)
```
