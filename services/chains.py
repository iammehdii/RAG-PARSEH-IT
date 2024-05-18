from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter


def _create_chain(prompt, llm, db):
    return (RunnableParallel(
        {"context": itemgetter("question") | db.as_retriever(search_kwargs={"k": 3}),
         'question': RunnablePassthrough()}
    ) | prompt | llm)
