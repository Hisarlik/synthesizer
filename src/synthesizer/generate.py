import os

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from pydantic import BaseModel, Field

os.environ['HF_TOKEN'] = "...."


page = '''
    
    ACCIONA desarrolla infraestructuras que contribuyan a responder a los problemas globales a través de soluciones sostenibles diseñadas para hacer posible la regeneración del planeta y promover sociedades más prósperas.

ACCIONA busca redefinir el papel de las infraestructuras, convirtiéndolas en catalizadores del progreso global y en herramientas clave para impulsar la regeneración del planeta. Esta ambición ha llevado a la compañía a expandir su actividad en sectores clave, donde genera un impacto sistémico y transformador. Con esta aproximación, ACCIONA se posiciona como la proveedora de soluciones de infraestructura sostenible más completa, capaz de desafiar la manera de hacer las cosas en numerosos sectores y, de esta manera, tratar de redefinir el futuro.

ACCIONA entiende que alinear sus modelos de negocio con los objetivos ambientales y sociales es clave para prosperar como compañía. Sus soluciones y proyectos favorecen la transformación de los entornos y medios de vida de sus grupos de interés. La compañía responde a los grandes retos ambientales y sociales desde una doble perspectiva:

- ACCIONA dirige su actividad hacia el desarrollo de infraestructuras que contribuyen a responder a los problemas globales a través de soluciones sostenibles, destacándose como referencia en numerosos ámbitos.

- ACCIONA trabaja para asegurar que la forma en la que diseña, construye y opera estas infraestructuras se desarrolle de acuerdo con las técnicas más avanzadas y las últimas innovaciones, lo que constituye una ventaja competitiva para el negocio además de una mejora desde el punto de vista sostenible.

ACCIONA en 100 palabras

ACCIONA es una de las principales empresas españolas del IBEX 35, con presencia en más de 42 países alrededor de todo el mundo. A través de su actividad, la Compañía ofrece respuesta a las necesidades de infraestructura básica, agua y energía, mediante soluciones innovadoras y generadoras de progreso e impacto positivo, en una nueva manera de hacer negocios orientada a diseñar un planeta mejor para todos.

La compañía desarrolla su actividad con más de cincuenta y siete mil profesionales, unas ventas que alcanzan los 17.021 millones de euros y un resultado bruto de explotación de 1.980 millones de euros (EBITDA) en 2023. \nACCIONA desarrolla soluciones de infraestructura que generan un impacto sostenible y promueven la regeneración en la vida de las personas y las comunidades. La Compañía dispone de soluciones en los siguientes ámbitos:

- **Energía**: Posee y opera activos de energía renovable, incluyendo eólica terrestre, fotovoltaica, biomasa, hidroeléctrica, termosolar y fabricación de tecnologías de energía renovable.

- **Transporte**: Construye y opera infraestructuras para el transporte de pasajeros y mercancías, como carreteras, puentes, vías férreas y túneles.

- **Agua**: Diseña, construye y opera plantas potabilizadoras, depuradoras de aguas residuales, procesos terciarios para reutilización y plantas desalinizadoras por ósmosis inversa.

- **Ciudades**: Responde a diversos retos en las ciudades, como la gestión de residuos, la movilidad eléctrica y compartida, la revitalización de espacios urbanos y el aumento de zonas verdes.

- **Social**: Desarrolla soluciones de infraestructuras sanitarias, educativas y culturales, así como para la preservación y limpieza del entorno natural.

- **Inmobiliaria**: La actividad inmobiliaria de ACCIONA se centra en el desarrollo y la gestión de complejos inmobiliarios. \nLas actividades de Acciona se observan desde el prisma de la taxonomía europea de actividades sostenibles y los objetivos de desarrollo sostenible (ODS). La tabla presenta una clasificación de las actividades de Acciona en diferentes ámbitos, soluciones y actividades específicas, junto con su alineación con la taxonomía europea de actividades sostenibles y su impacto sobre los ODS.
    

    
'''



class ExamQuestion(BaseModel):
    question: str = Field(..., description="The question to be answered in spanish")
    answer: str = Field(..., description="The correct answer to the question in spanish")
    distractors: list[str] = Field(
        ..., description="A list of incorrect but viable answers to the question"
    )

class ExamQuestions(BaseModel):  # 
    exam: list[ExamQuestion]


PROMPT = """\
You are an AI assistant that helps to generate exam questions and answers in Spanish.
Your goal is to create 1 questions and answers in spanish based on the document provided, and a list of distractors, that are incorrect but viable answers to the question.
Document:\n\n{{page}}
You can't use the word documento.
Your answer must adhere to the following format:
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
""".strip() # 


with Pipeline(name="ExamGenerator") as pipeline:

        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "page": page  # 
                }
            ],
        )

        text_generation = TextGeneration(  # 
            name="exam_generation",
            template=PROMPT,
            llm = OpenAILLM(
                model="gpt-4o-mini",
                api_key=".....",
                structured_output={
                    "schema": ExamQuestions.model_json_schema(),
                    "format": "json"
                },
                generation_kwargs={"max_new_tokens": 4000},
        ),
        columns=["page"],
        output_mappings={"model_name": "generation_model"},
        input_batch_size=1
        )

        load_dataset >> text_generation

  

if __name__ == "__main__":
     
    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                    
                    }
                }
            }
        },
        use_cache=False,
    )

    distiset.push_to_hub("amentaphd/exam_questions")
