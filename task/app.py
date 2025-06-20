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
        #  Check if `microwave_faiss_index` folder exists (os.path.exists("{folder_name}"))
        #  - Exists:
        #       It means that we have already converted data into vectors (embeddings), saved them in FAISS vector
        #       store and saved it locally to reuse it later.
        #       - Load FAISS vectorstore from local index (FAISS.load_local(...))
        #       - Configure folder_path `microwave_faiss_index`
        #       - Configure embeddings `self.embeddings`
        #       - Configure allow_dangerous_deserialization `True` (for our case it is ok, but don't do it on PROD)
        #       - Make variable assignment to `vectorstore`
        #  - Otherwise:
        #       - Make variable assignment of `self._create_new_index()` to `vectorstore`
        #  Return `vectorstore`
        return None

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        # TODO:
        #  1. Create langchain_community.document_loaders.TextLoader:
        #       - file_path is `microwave_manual.txt`
        #       - encoding is `utf-8`
        #       - assign it to `loader` variable
        #  2. Load documents via `loader.load()` and assign to `documents` variable
        #  3. Load `documents` via `loader.load()`
        #  4. Create RecursiveCharacterTextSplitter with
        #       - chunk_size=300
        #       - chunk_overlap=50
        #       - separators=["\n\n", "\n", "."]
        #  5. Split `documents` via created `text_splitter` into `chunks`
        #  6. Create `vectorstore` via FAISS.from_documents:
        #       - documents=chunks
        #       - embeddings=self.embeddings
        #  7. Save indexed data locally `vectorstore.save_local("microwave_faiss_index")`
        #  8. Return created `vectorstore`
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
        #  Make `similarity_search_with_relevance_scores` in `vectorstore`:
        #       - query is `microwave_manual.txt`
        #       - k=k
        #       - score_threshold=score
        #       - assign results to `relevant_docs` variable

        context_parts = []
        # TODO:
        #  Iterate through results (`for (doc, score) in relevant_docs`) and:
        #       - add `doc.page_content` to `context_parts` array
        #       - print `score`
        #       - print `page_content`

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = None #TODO: Assign USER_PROMPT with format for `context` and `query` parameters to the `augmented_prompt`

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        # TODO:
        #  1. Create `messages` array with such messages:
        #       - SystemMessage(content=SYSTEM_PROMPT)
        #       - HumanMessage(content=augmented_prompt)
        #  2. Call self.llm_client.invoke(messages) and assign result to `response` variable
        #  3. Return response content
        return None


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        # Step 1: Retrieval
        context = rag.retrieve_context(user_question) # Here you can play with `k` and similarity score params
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        answer = rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        # TODO:
        #  1. pass embeddings:
        #       - AzureOpenAIEmbeddings
        #       - deployment='text-embedding-3-large-1' (or 'text-embedding-3-small-1')
        #       - azure_endpoint=DIAL_URL
        #       - api_key=SecretStr(API_KEY)
        #  2. pass llm_client:
        #       - AzureChatOpenAI
        #       - temperature=0.0
        #       - azure_endpoint=DIAL_URL
        #       - api_key=SecretStr(API_KEY)
        #       - api_version="2024-05-01-preview"
    )
)