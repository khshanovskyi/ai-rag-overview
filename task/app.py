import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY

#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """

"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """

"""


#TODO:
# Implement MicrowaveRAG class that will:
# - In constructor it should apply Embeddings client, Chat completion client and set up vectorstore
# - vectorstore should be set up from saved locally index (if present) or from microwave_manual.txt (split to chunks
#   and load to FAISS vector DB)
# - Provide method `retrieve_context` that will make similarity search by user input `k` documents with some score
# - Provide method `augment_prompt` that will make user prompt augmentation (retrieved context + user input)
# - Provide method `generate_answer` that will apply augmented prompt and generate answer with LLM

class MicrowaveRAG:
    ...



#TODO:
# Create method that will create MicrowaveRAG and run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)
