import os

import pandas as pd

llms = ["llama3.2", "mistral", "granite3-dense"]
embs = ["roberta-base-bne", "robertalex"]
prev_directory = os.path.dirname(os.getcwd())

for llm in llms:
    for emb in embs:
        for i in range(3):  # indica los diferentes modos. 0 es sin RAG, 1 es con RAG y 2 es con RAG y expansion
            if i == 0:
                evaluation_dataset = pd.read_csv(
                    fr"{prev_directory}\evaluation\no_rag\{llm}.csv",
                    usecols=["rouge1", "rouge2", "rougeL", "f1", "sas", "bertscore"]
                )
                # se define el directorio en el que se almacenaran los resultados
                path_csv = fr"{prev_directory}\evaluation\mean_score\{llm}.csv"

            elif i == 1:
                evaluation_dataset = pd.read_csv(
                    fr"{prev_directory}\evaluation\rag\{emb}\{llm}.csv",
                    usecols=["rouge1", "rouge2", "rougeL", "f1", "sas", "bertscore"]
                )
                path_csv = fr"{prev_directory}\evaluation\mean_score\{llm}_{emb}.csv"

            else:
                evaluation_dataset = pd.read_csv(
                    fr"{prev_directory}\evaluation\expanded\{emb}\{llm}.csv",
                    usecols=["rouge1", "rouge2", "rougeL", "f1", "sas", "bertscore"]
                )
                path_csv = fr"{prev_directory}\evaluation\mean_score\expanded_{llm}_{emb}.csv"

            rouge1_mean = 0
            rouge2_mean = 0
            rougeL_mean = 0
            f1_mean = 0
            sas_mean = 0
            bertscore_mean = 0
            for j in range(len(evaluation_dataset)):
                scores = evaluation_dataset.iloc[j]
                rouge1 = float(scores['rouge1'])
                rouge2 = float(scores['rouge2'])
                rougeL = float(scores['rougeL'])
                f1 = float(scores['f1'])
                sas = float(scores['sas'])
                bertscore = float(scores['bertscore'])
                rouge1_mean += rouge1
                rouge2_mean += rouge2
                rougeL_mean += rougeL
                f1_mean += f1
                sas_mean += sas
                bertscore_mean += bertscore

            rouge1_mean = round(rouge1_mean/len(evaluation_dataset), 2)
            rouge2_mean = round(rouge2_mean / len(evaluation_dataset), 2)
            rougeL_mean = round(rougeL_mean / len(evaluation_dataset), 2)
            f1_mean = round(f1_mean / len(evaluation_dataset), 2)
            sas_mean = round(sas_mean / len(evaluation_dataset), 2)
            bertscore_mean = round(bertscore_mean / len(evaluation_dataset), 2)

            dict_response = {
                "rouge1": [rouge1_mean],
                "rouge2": [rouge2_mean],
                "rougeL": [rougeL_mean],
                "f1": [f1_mean],
                "sas": [sas_mean],
                "bertscore": [bertscore_mean]
            }
            df = pd.DataFrame(dict_response)
            df.to_csv(path_csv)