import os
from datetime import datetime

import pandas as pd
from src.code.evaluator import Evaluator

llms = ["llama3.2", "mistral"]
embeddings = ["roberta-base-bne", "robertalex"]

current_directory = os.getcwd()
prev_directory = os.path.dirname(current_directory)

for llm in llms:
    for embedding in embeddings:
        print("Ha comenzado la evaluación del sistema con llm: ", llm, " y embedding: ", embedding)
        print("Time: ", datetime.now().time())
        for i in [2]: #indica los diferentes modos. 0 es sin RAG, 1 es con RAG y 2 es con RAG y expansion
            if i == 0:
                answer_dataset = pd.read_csv(
                    fr"{prev_directory}\answers\no_rag\{llm}.csv",
                    usecols=['question', 'answer', 'ground_truth']
                )
                #se define el directorio en el que se almacenaran los resultados
                path_csv = fr"{prev_directory}\evaluation\no_rag\{llm}.csv"

            elif i == 1:
                answer_dataset = pd.read_csv(
                    fr"{prev_directory}\answers\rag\{embedding}\{llm}.csv",
                    usecols=['question', 'answer', 'ground_truth']
                )
                path_csv = fr"{prev_directory}\evaluation\rag\{embedding}\{llm}.csv"

            else:
                answer_dataset = pd.read_csv(
                    fr"{prev_directory}\answers\expanded\{embedding}\{llm}.csv",
                    usecols=['question', 'answer', 'ground_truth']
                )
                path_csv = fr"{prev_directory}\evaluation\expanded\{embedding}\{llm}.csv"

            response = []
            for j in range(len(answer_dataset)):
                question = answer_dataset.iloc[j]['question']
                answer = answer_dataset.iloc[j]['answer']
                ground_truth = answer_dataset.iloc[j]['ground_truth']
                evaluator = Evaluator(question, answer, ground_truth)
                scores = evaluator.eval_answer()
                response.append(scores)

            response_df = pd.DataFrame(response)
            response_df.to_csv(path_csv)