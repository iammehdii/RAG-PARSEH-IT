from langchain_core.pydantic_v1 import BaseModel
from typing import List, Optional, Literal


class DiseaseItem(BaseModel):
    disease_name: str
    crops: List[str]
    symptoms: str
    organic_control: str
    chemical_control: str
    trigger: str
    preventive_measures: str
    url: Optional[str] = None


class Question(BaseModel):
    __root__: str


class TopicClassifier(BaseModel):
    """Classify the topic of the user question"""
    topic: Literal["agriculture", "general"]
    "The topic of the user question. One of 'agriculture', or 'general'."
