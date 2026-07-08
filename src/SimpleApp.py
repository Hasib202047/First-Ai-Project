from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", # Cleaned up repo id if using standard HF
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)
chat = ChatHuggingFace(llm=llm)

# 2. Define an output parser to extract clean string text from the model responses
output_parser = StrOutputParser()

# ----------------------------------------------------
# CHAIN 1: Generate a company name based on a product
# ----------------------------------------------------
prompt1 = ChatPromptTemplate.from_template(
    "What is a good, catchy name for a company that makes {product}?"
)
# Construct the first chain
chain_one = prompt1 | chat | output_parser

# ----------------------------------------------------
# CHAIN 2: Generate a tagline based on the company name
# ----------------------------------------------------
prompt2 = ChatPromptTemplate.from_template(
    "Write a catchy 3-word slogan or tagline for the following company: {company_name}"
)
# Construct the second chain
chain_two = prompt2 | chat | output_parser

# ----------------------------------------------------
# THE SEQUENTIAL CHAIN: Link them together
# ----------------------------------------------------
# We use a lambda function to map the output of chain_one into the input key of chain_two
sequential_chain = RunnablePassthrough.assign(company_name=chain_one) | chain_two