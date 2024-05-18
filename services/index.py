import pandas as pd
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv('.env')

question_templates = {
    "crops": "Which crops are affected by {disease_name}?",
    "symptoms": "What are the symptoms of {disease_name}?",
    "organic_control": "How can {disease_name} be controlled organically?",
    "chemical_control": "What are the chemical control methods for {disease_name}?",
    "trigger": "What triggers {disease_name}?",
    "preventive_measures": "What preventive measures can be taken against {disease_name}?"
}


def load_local_db(path='db/plantix_faiss'):
    """Load local vector store"""
    load_dotenv('.env')
    return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


def format_input(items_in_dict,question_templates=question_templates):
    """Add items to db"""
    qa_pairs = []
    disease_name = items_in_dict['name_en']
    for key, question_template in question_templates.items():
        question = question_template.format(disease_name=disease_name)
        answer = items_in_dict[key]
        if isinstance(answer, str):
            qa_pair = question + '\n' + answer
        elif isinstance(answer, list):
            qa_pair = question + '\n' + ' '.join(answer)
        else:
            continue
        doc = Document(page_content=qa_pair,metadata={"source": items_in_dict.get('url', None)})

        qa_pairs.append(doc)
    return qa_pairs

db = FAISS.load_local('db/plantix_faiss', OpenAIEmbeddings(), allow_dangerous_deserialization=True)


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
