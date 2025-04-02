import re
import unicodedata
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from transformers import BertTokenizer, BertModel
import numpy as np


def _normalize_text(text:str):
    #elimina saltos de linea y espacios adicionales
    normalized = text.replace("\n", " ").strip()
    #normaliza el texto para separar los caracteres base de los acentos
    normalized = unicodedata.normalize('NFKD', normalized)
    #filtra solo los caracteres que no son marcas de acento
    normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
    #reemplaza multiples espacios por uno solo
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized


class Evaluator:

    def __init__(self, question: str, generated_answer: str, ground_truth: str):
        """
        :param generated_answer: Respuesta generada por el modelo generativo.
        :param ground_truth: Respuesta referencia.
        """

        self.generated_answer = generated_answer
        self.ground_truth = ground_truth
        self.question = question

    def eval_answer(self):
        """
        :return: dict con los mean_score para las métricas ROUGE-1, ROUGE-2, ROUGE-L, F1-score, SAS y BERTScore
        """
        def rouge():
            """
           :return: Score obtenido para Rouge-1, Rouge-2 y Rouge-L
           """
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(self.ground_truth, self.generated_answer)
            return scores

        def f1_score():
            """
           :return: Un valor en el intervalo [0,1] que indica el f1 score de cada pregunta
           """
            pred_tokens = _normalize_text(self.generated_answer).split()
            truth_tokens = _normalize_text(self.ground_truth).split()

            if len(pred_tokens) == 0 or len(truth_tokens) == 0:
                return int(pred_tokens == truth_tokens)

            common_tokens = set(pred_tokens) & set(truth_tokens)

            if len(common_tokens) == 0:
                return 0

            prec = len(common_tokens) / len(pred_tokens)
            rec = len(common_tokens) / len(truth_tokens)
            return 2 * (prec * rec) / (prec + rec)

        def sas():
            """
            :return: Score obtenido para Semantic Answer Similarity
            El modelo utilizado para el cálculo de esta métrica es cross-encoder/stsb-roberta-large
            """
            model = CrossEncoder('cross-encoder/stsb-roberta-large')
            scores = model.predict([[self.generated_answer, self.ground_truth]])
            return scores[0]

        def bert_score():
            """
            :return: Score obtenido para BERTScore
            El modelo utilizado para el cálculo de esta métrica es bert-base-uncased
            """
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")

            #calculo de los tokens
            inputs1 = tokenizer(self.generated_answer, return_tensors="pt", padding=True, truncation=True)
            inputs2 = tokenizer(self.ground_truth, return_tensors="pt", padding=True, truncation=True)
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

            #calculo de los embeddings
            embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

            #calculo de cosine similarity entre los embeddings obtenidos
            similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
            return similarity[0][0]

        rouges = rouge()
        score = {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.generated_answer,
            "rouge1": round(rouges['rouge1'].fmeasure, 2),
            "rouge2": round(rouges['rouge2'].fmeasure, 2),
            "rougeL": round(rouges['rougeL'].fmeasure, 2),
            "f1": round(f1_score(), 2),
            "sas": sas(),
            "bertscore": round(bert_score(), 2)
        }
        return score