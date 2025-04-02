import os
import pandas as pd
import torch.cuda
from dotenv import load_dotenv
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MultiQueryRetriever:

    def __init__(self, vector_store):

        load_dotenv()
        token = os.getenv('TOKEN')
        self.vector_db = vector_store
        self.docs = []
        self.results = []
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name, token=token)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name, token=token)
        self.reranker_model.to(self.device)

    def _add_documents(self, document: tuple[Document, float]):
            self.docs.append(document[0])
            self.results.append(document)

    def _rerank_documents(self, query: str, documents: list[Document], top_k: int = 5):
        pairs = []

        for doc in documents:
            pairs.append((query, doc.page_content))

        features = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            scores = self.reranker_model(**features).logits.squeeze().cpu().numpy()

        doc_score_pairs = list(zip(documents, scores))
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return ranked_docs[:top_k]

    def run(self, query):
        """
        :param query: Query para la que se desea obtener los documentos relevantes.
        :return: Devuelve un diccionario con la lista de documentos relevantes.
        La clave del diccionario es 'documents'.
        """
        queries = [query]

        prev_directory = os.path.dirname(os.getcwd())
        expanded_questions_dataset = pd.read_csv(
            fr"{prev_directory}\resources\expanded_questions_aux.csv",
            usecols=['original', 'expandida']
        )

        for i in range(len(expanded_questions_dataset)):
            aux = expanded_questions_dataset.iloc[i]
            if aux['original'] == query and aux['expandida'] not in queries:
                queries.append(aux['expandida'])

        for q in queries:
            result = self.vector_db.similarity_search_with_relevance_scores(q, 3)
            for doc in result:
                self._add_documents(doc)

        self.results = self._rerank_documents(query, self.docs, 3)
        return self.results