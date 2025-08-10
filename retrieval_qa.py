from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. Load the policy document
loader = TextLoader("policy.txt")
documents = loader.load()

# 2. Split into chunks for embedding
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings & store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Load a local LLM (small for CPU)
model_name = "gpt2"  # You can replace with a larger model if you have GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.3,
    device=0 if torch.cuda.is_available() else -1
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 5. Create RetrievalQA pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 6. Ask a question
query = "What is the refund policy?"
result = qa({"query": query})

print("\n=== Question ===")
print(query)
print("\n=== Answer ===")
print(result["result"])
print("\n=== Source Document ===")
print(result["source_documents"][0].page_content)
