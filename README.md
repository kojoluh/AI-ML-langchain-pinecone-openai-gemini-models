# AI-ML-langchain-pinecone-openai-gemini-models

## Overview
- Deep Dive into LangChain
- This is a direct follow on the Udemy course (https://www.udemy.com/course/master-langchain-pinecone-openai-build-llm-applications).
- I am glad to have taken the course and took time to recreate/follow the lesson examples.

## How to get Started
- clone the project
```
$ git clone https://github.com/kojoluh/AI-ML-langchain-pinecone-openai-gemini-models
```

- make a copy of the .env.sample file renamed as .env
- get your OpenAI API key using the OpenAI platform (https://platform.openai.com/)
- add your OpenAI API key to the .env file
- try run the initial setup using
```
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

os.environ.get('OPENAI_API_KEY')
```

- Alternatively, you can follow as below to create the project folder as below and copy the files into it.


- create a new directory/folder
```
$ mkdir langChainDemo
```

- switch into the directory
```
$ cd langChainDemo
```

- Install the requirements : the [pip3] is used on mac instead of pip
```
$ pip [pip3] install -r ./requirements.txt -q
```

- show if langchain is installed and its version details
```
$ pip3 show langchain
```

- show if openai is installed and its version details
```
$ pip3 show openai
```

- now start jupyter notebook
```
$ jupyter notebook
```

- if above does not work, use what is below *NB: only for test lab
```
$ jupyter lab --app-dir /opt/homebrew/share/jupyter/lab
```
- refer to https://github.com/jupyter/notebook/issues/6974 if you have issues with jupyter and python3 on Mac



