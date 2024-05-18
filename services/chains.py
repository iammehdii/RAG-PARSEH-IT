from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv('.env')

def _create_chain(prompt, llm, db):
    return (RunnableParallel(
        {"context": itemgetter("question") | db.as_retriever(search_kwargs={"k": 3}),
         'question': RunnablePassthrough()}
    ) | prompt | llm | StrOutputParser())


# def _get_prompt():
template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)


db = FAISS.load_local('db/plantix_faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
# prompt = _get_prompt()

# LLM
llm = ChatOpenAI()

# RAG chain
chain = _create_chain(prompt, llm, db)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)