from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from retriever import Retriever
from src.code.multi_query_retriever import MultiQueryRetriever


class RAG:

    def __init__(self, llm_model: str, embedding_model: str, pdf_file_path: str):
        """
        :param llm_model: Modelo generativo que se desea emplear.
        :param embedding_model: Modelo de embeddings que se desea emplear.
        :param pdf_file_path: Ruta en la que se encuentra el fichero PDF a leer.
        """

        self.pdf_file_path = pdf_file_path
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        retriever = Retriever(self.pdf_file_path, self.embedding_model)
        self.vector_db = retriever.retriever()

        #inicializamos el modelo generativo
        self.model = ChatOllama(model=llm_model)

        #creamos un template para el prompt que se le va a pasar al LLM
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] Eres un asistente para tareas de respuesta a preguntas. Utiliza los siguientes elementos
            contextuales para responder a la pregunta. Si no sabes la respuesta, di simplemente que no lo sabes.
            Sé conciso en tu respuesta. [/INST] </s>
            [INST]Pregunta: {question}
            Contexto: {context}
            Respuesta: [/INST]
            """
        )

    def ask(self, query, expanded: bool = False):
        """
        Se realiza la pregunta deseada al modelo con RAG
        :param query: Pregunta que se quiere realizar
        :param expanded: Flag que indica si se desea expandir la pregunta o no
        :return: Devuelve la respuesta generada por el modelo
        """

        if expanded:
            multi_retriever = MultiQueryRetriever(vector_store=self.vector_db)
            contexts = multi_retriever.run(query)[:3]
        else:
            contexts = self.vector_db.similarity_search_with_relevance_scores(query, 3, score_threshold=0.5)

        contexts_text = "\n\n".join([context[0].page_content for context in contexts])
        prompt = self.prompt_template.format(context=contexts_text, question=query)
        answer = self.model.invoke(prompt)

        return answer.content, contexts

    def ask_model(self, query):
        """
        Se realiza la pregunta deseada al modelo sin RAG
        :param query: Pregunta que se quiere realiza
        :return: Devuelve la respuesta generada por el modelo
        """
        return self.model.invoke(query).content
