from fastapi import FastAPI
from services import chain

app = FastAPI()






@app.get("/")
async def root():
    prompt = _get_prompt()
    return {"message": prompt}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/query")
def query(question: str):
    resp = chain.invoke({'question':question})

    return {"message": resp}

@app.post("/update")
def add_to_db(name: str):
