from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)
chat = ChatHuggingFace(llm=llm)

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template (
    "### Summarization ###" \
    "Here’s the text I need summarized: {text} " \
    "summarize this text in {words} words"
)

summarization_chain = prompt | chat | output_parser

