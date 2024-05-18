from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function

# sys.path.insert(1, "services/")
from .index import db
from models import TopicClassifier, Question
load_dotenv('.env')


def _create_chain(prompt, llm, db):
    return (RunnableParallel(
        {"context": itemgetter("question") | db.as_retriever(search_kwargs={"k": 3}),
         'question': RunnablePassthrough()}
    ) | prompt | llm | StrOutputParser())


template = """Answer the question based only on the following context:
{context}
Question: {question}
    """
rag_prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI()

# RAG chain
rag_chain = _create_chain(rag_prompt, llm, db)

classifier_function = convert_to_openai_function(TopicClassifier)
llm_topic = llm.bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)
parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="topic"
)
classifier_chain = llm_topic | parser
general_chain = (
        ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question as accurately as you can
            Question: {question}
            Answer:""")
        | llm
        | StrOutputParser()
        )
branch = RunnableBranch(
    (lambda x: "agriculture" == x["topic"].lower(), rag_chain),
    general_chain
)
chain = (
    RunnablePassthrough.assign(topic=itemgetter("question") | classifier_chain)
    | branch
)
# Add typing for input
# class Question(BaseModel):
#     __root__: str

# chain = classifier_chain
chain = chain.with_types(input_type=Question)