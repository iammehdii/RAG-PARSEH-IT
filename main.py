from fastapi import FastAPI
from services import chain, format_input, db
from models import DiseaseItem

app = FastAPI()


@app.get("/query")
def query(question: str):
    resp = chain.invoke({'question': question})

    return {"response": resp}

@app.post("/add")
def add_to_db(items: DiseaseItem, local=True):
    docs = format_input(items.dict)
    db.add_documents(docs)
    if local:
        db.save_local('db/plantix_faiss')
    return {"response": 'done'}

