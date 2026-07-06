from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
# key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

parser = JsonOutputParser()
output_parser = StrOutputParser()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512
)
chat = ChatHuggingFace(llm=llm)

TEMPLATE="""
Text{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sute the questions are not repeated and check all the questions to be confirmed the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. Strictly follow the format \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""
chain_one = ChatPromptTemplate.from_template(TEMPLATE) | chat | output_parser

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""
chain_two = ChatPromptTemplate.from_template(TEMPLATE2) | chat | output_parser

generate_evaluate_chain = (
    RunnablePassthrough.assign(quiz=chain_one) | RunnablePassthrough.assign(review=chain_two)
)