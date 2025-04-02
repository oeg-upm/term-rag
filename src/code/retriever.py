import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Retriever:

    def __init__(self, pdf_file_path: str, embedding_model: str | None):
        """
        :param pdf_file_path: Ruta en la que se encuentra el fichero a leer.
        :param embedding_model: Modelo de embeddings que se desea emplear.
        """

        load_dotenv()
        token = os.getenv('TOKEN')
        self.pdf_file_path = pdf_file_path
        self.embedding_model = embedding_model

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100
        )

        #inicializamos el modelo para el cálculo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu', 'use_auth_token': token},
                encode_kwargs={'normalize_embeddings': True}
        )


    def retriever(self):
        """
        El proceso llevado a cabo es el siguiente:
            1. Se obtiene el texto obtenido de la ruta pdf_file_path.
            2. El texto se divide en chunks según lo especificado en la variable text_splitter.
            3. Se calcula el embedding de los chunks.
            4. Los vectores calculados se almacenan en una base de datos vectorial.
        :return: Se devuelve la base de datos en la que se almacenan los vectores
        """
        docs = PyPDFLoader(file_path=self.pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, self.embeddings)

        return db