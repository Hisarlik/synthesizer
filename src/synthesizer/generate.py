import os
from typing import List  # Add type hints
from pathlib import Path

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from pydantic import BaseModel, Field

# Constants should be in UPPER_CASE at module level
API_KEY = "....."
HF_TOKEN = "...."
MODEL_NAME = "gpt-4o-mini"

os.environ['HF_TOKEN'] = HF_TOKEN

def load_content() -> str:
    """Load the ACCIONA content from the text file."""
    content_path = Path(__file__).parent / "data" / "acciona_content.txt"
    with open(content_path, "r", encoding="utf-8") as f:
        return f.read()

class ExamQuestion(BaseModel):
    """Model representing a single exam question with its answer and distractors."""
    question: str = Field(
        ..., 
        description="The question text in Spanish",
        min_length=10
    )
    answer: str = Field(
        ..., 
        description="The correct answer in Spanish",
        min_length=1
    )
    distractors: List[str] = Field(
        ..., 
        description="List of incorrect but plausible answers",
        min_items=2,
        max_items=4
    )

class ExamQuestions(BaseModel):
    """Container for multiple exam questions."""
    exam: List[ExamQuestion]


PROMPT_TEMPLATE = """
You are an AI assistant that helps to generate exam questions and answers in Spanish.
Your goal is to create 1 questions and answers in spanish based on the document provided, 
and a list of distractors, that are incorrect but viable answers to the question.

Document:

{{page}}

Note: Do not use the word 'documento' in your responses.

Required JSON format:
```
[
    {
        "question": "Your question in spanish language",
        "answer": "The correct answer to the question in spanish",
        "distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"]
    },
    ... (more questions and answers as required)
]
```
""".strip()

def create_pipeline() -> Pipeline:
    """Creates and configures the exam generation pipeline."""
    with Pipeline(name="ExamGenerator") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[{"page": load_content()}]
        )

        text_generation = TextGeneration(
            name="exam_generation",
            template=PROMPT_TEMPLATE,
            llm=OpenAILLM(
                model=MODEL_NAME,
                api_key=API_KEY,
                structured_output={
                    "schema": ExamQuestions.model_json_schema(),
                    "format": "json"
                }
            ),
            columns=["page"],
            output_mappings={"model_name": "generation_model"},
            input_batch_size=1
        )

        load_dataset >> text_generation
        return pipeline

def main():
    """Main execution function."""
    pipeline = create_pipeline()
    distiset = pipeline.run(
        parameters={
            "exam_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 4000,
                    }
                }
            }
        },
        use_cache=False,
    )
    
    project_name = "exam"
    distiset.push_to_hub(f"amentaphd/{project_name}")

if __name__ == "__main__":
    main()

