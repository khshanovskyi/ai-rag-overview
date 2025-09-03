import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        # TODO:
        #  Check if `microwave_faiss_index` folder exists
        #  - Exists:
        #       It means that we have already converted data into vectors (embeddings), saved them in FAISS vector
        #       store and saved it locally to reuse it later.
        #       - Load FAISS vectorstore from local index (FAISS.load_local(...))
        #           - Configure folder_path `microwave_faiss_index`
        #           - Configure embeddings `self.embeddings`
        #           - Allow dangerous deserialization (for our case it is ok, but don't do it on PROD)
        #  - Otherwise:
        #       - Create new index
        #  Return create vectorstore
        return None

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        # TODO:
        #  1. Create Text loader:
        #       - file_path is `microwave_manual.txt`
        #       - encoding is `utf-8`
        #  2. Load documents with loader
        #  3. Create RecursiveCharacterTextSplitter with
        #       - chunk_size=300
        #       - chunk_overlap=50
        #       - separators=["\n\n", "\n", "."]
        #  4. Split documents into `chunks`
        #  5. Create vectorstore from documents
        #  6. Save indexed data locally with index name "microwave_faiss_index"
        #  7. Return created vectorstore
        return None

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # TODO:
        #  Make similarity search with relevance scores`:
        #       - query=query
        #       - k=k
        #       - score_threshold=score

        context_parts = []
        # TODO:
        #  Iterate through results and:
        #       - add page content to the context_parts array
        #       - print result score
        #       - print page content

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = None #TODO: Format USER_PROMPT with context and query

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        # TODO:
        #  1. Create messages array with such messages:
        #       - System message from SYSTEM_PROMPT
        #       - Human message from augmented_prompt
        #  2. Invoke llm client with messages
        #  3. print response content
        #  4. Return response content
        return None


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        #TODO:
        # Step 1: make Retrieval of context
        # Step 2: Augmentation
        # Step 3: Generation



main(
    MicrowaveRAG(
        # TODO:
        #  1. pass embeddings:
        #       - AzureOpenAIEmbeddings
        #       - deployment is the text-embedding-3-small-1 model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        #  2. pass llm_client:
        #       - AzureChatOpenAI
        #       - temperature is 0.0
        #       - azure_deployment is the gpt-4o model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        #       - api_version=""
    )
)