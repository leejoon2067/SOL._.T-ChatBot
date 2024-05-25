from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.docstore.document import Document

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer

import numpy as np
import csv
import faiss
import tiktoken


# 벡터스토어 db 인스턴스 생성

class TextEmbedding_Faiss:
    def __init__(self, model_name = "jhgan/ko-sbert-nli"):
        # 텍스트 토큰으로 분할을 위한 초기화
        self.model = SentenceTransformer(model_name)
        self.index = None
    
    def text_split(self, file_path):
        with open(file_path, encoding = 'utf-8') as f:
            state_of_the_union = f.read()

        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 100,
            chunk_overlap = 10,
            length_function = len,
        )
        texts = text_splitter.split_text(state_of_the_union)
        texts = texts[1:]
        
        return texts
    
    def get_text_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return np.array(embeddings)
    
    def create_faiss_index(self, embeddings):
        # FAISS 인덱스 생성 및 벡터 추가
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

class Customdb_Chroma:
    def __init__(self):
        self.user_input = None
        self.hf = HuggingFaceEmbeddings()

    def create_db(self, csv_path):
        sentence_texts = []
        with open(csv_path, new_line ='', encoding = 'utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                sentence_texts.append(row[1])
       
        sentence_embeddings = self.hf.embed_documents(sentence_texts)
        documents = [Document(page_content=text) for text in sentence_texts]
        self.db = Chroma.from_documents(documents, self.hf, persist_directory= "./chroma_db")

        return self.db

    def input_db(self, user_input):
        self.user_input = user_input
        query_embedding = self.hf.embed_documents([user_input])[0]

        db3 = Chroma(persist_directory="./chroma_db", embedding_function= self.hf)
        
        return db3.similarity_search(user_input, k =1)
        
    def retrieve_most_similar_sentence(doc):
        if doc:
            most_similar_doc = doc[0]
            return most_similar_doc.page_content
        else:
            return None
    
    def create_retriever(self, file_path):
        db = self.create_db(file_path)
        embedding = OpenAIEmbeddings()
        retriever = db.as_retriever()

        bm25_retriever = BM25Retriever.from_texts(
            self.sentences
        )
        faiss_vectorstore = FAISS.from_texts(
            self.sentences,
            embedding
        )
        faiss_retriever = faiss_vectorstore.as_retriever(
            search_k = 20,
            k = 1
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            search_type = "mmr"
        )
        # ensemble_result = ensemble_retriever.get_relevant_documents(user_input)
        # return ensemble_result
        return ensemble_retriever
