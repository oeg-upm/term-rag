import subprocess
from datetime import datetime
import os
import pandas as pd
from generator import RAG

prev_directory = os.path.dirname(os.getcwd())
question_dataset_path=rf"{prev_directory}\resources\Cuatrecasas-OEG-Spanish Workers Statute Eval Dataset.xlsx - 1st set.csv"
llms = ["llama3.2", "mistral"]
embeddings = ["PlanTL-GOB-ES/roberta-base-bne", "PlanTL-GOB-ES/RoBERTalex"]

#se carga el dataset de preguntas para realizar la evaluación
question_dataset = pd.read_csv(
    question_dataset_path,
    usecols=['Question Spanish', 'Answer Spanish (highlight paragraph)\nBLACK BOLD']
)

for llm in llms:
    for embedding in embeddings:
        print("Ha comenzado la ejecución del sistema con llm: ", llm, " y embedding: ", embedding)
        print("Time: ", datetime.now().time())
        for i in [2]: #indica los diferentes modos. 0 es sin RAG, 1 es con RAG y 2 es con RAG y expansion

            # creamos nuestro RAG dataset para realizar la evaluación
            rag_dataset = []
            rag = RAG(
                llm_model=llm,
                embedding_model=embedding,
                pdf_file_path=fr"{prev_directory}\resources\BOE-A-2015-11430-consolidado.pdf"
            )

            for j in range(len(question_dataset)):
                question = question_dataset.iloc[j]['Question Spanish']
                ground_truth = question_dataset.iloc[j]['Answer Spanish (highlight paragraph)\nBLACK BOLD']

                if i == 0:
                    answer = rag.ask_model(question)
                elif i == 1:
                    answer, contexts = rag.ask(question)
                else:
                    answer, contexts = rag.ask(question, True)

                response_dict = {
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth
                }
                rag_dataset.append(response_dict)

            if i == 0:
                path_csv = fr"{prev_directory}\answers\no_rag\{llm}.csv"
            elif i == 1:
                if embedding == "PlanTL-GOB-ES/roberta-base-bne":
                    path_csv = fr"{prev_directory}\answers\rag\roberta-base-bne\{llm}.csv"
                else:
                    path_csv = fr"{prev_directory}\answers\rag\robertalex\{llm}.csv"
            else:
                if embedding == "PlanTL-GOB-ES/roberta-base-bne":
                    path_csv = fr"{prev_directory}\answers\expanded\roberta-base-bne\{llm}.csv"
                else:
                    path_csv = fr"{prev_directory}\answers\expanded\robertalex\{llm}.csv"
            df = pd.DataFrame(rag_dataset)
            df.to_csv(path_csv)
            print("Ha finalizado la ejecución del sistema con llm: ", llm, " y embedding: ", embedding)
            print("Time: ", datetime.now().time())

subprocess.run("shutdown -s")