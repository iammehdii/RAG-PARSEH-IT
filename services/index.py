from langchain.document_loaders import AsyncHtmlLoader
import pandas as pd
# from langchain_community.document_transformers import BeautifulSoupTransformer
# from langchain_community.document_transformers import BeautifulSoupTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

question_templates = {
    "crops": "Which crops are affected by {disease_name}?",
    "symptoms": "What are the symptoms of {disease_name}?",
    "organic_control": "How can {disease_name} be controlled organically?",
    "chemical_control": "What are the chemical control methods for {disease_name}?",
    "trigger": "What triggers {disease_name}?",
    "preventive_measures": "What preventive measures can be taken against {disease_name}?"
}

if __name__ == "__main__":
    load_dotenv('.env')  #

    qa_pairs = []
    df = pd.read_csv('data/structured_data_from_plantix.csv')
    for index, row in df.iterrows():
        disease_name = row['name_en']
        for column, question_template in question_templates.items():
            question = question_template.format(disease_name=disease_name)
            answer = row[column]
            if isinstance(answer, str):
                qa_pair = question + '\n' + answer
            elif isinstance(answer, list):
                qa_pair = question + '\n' + ' '.join(answer)
            else:
                continue
            doc = Document(page_content=qa_pair,metadata={"source": row['url']})

            qa_pairs.append(doc)

    embedding = OpenAIEmbeddings()

    vs = FAISS.from_documents(qa_pairs, embedding)
    vs.save_local('db/plantix_faiss')
