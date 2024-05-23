from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tiktoken
from langchain_community.document_loaders.csv_loader import CSVLoader

import pandas as pd 
import numpy as np
import faiss

# 벡터스토어 db 인스턴스를 생성
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

class CustomRetriever_Faiss:
    def __init__(self):
        # 텍스트 토큰으로 분할을 위한 초기화
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.model = "jhgan/ko-sbert-nli"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': True}

        self.hf = HuggingFaceEmbeddings(
            model_name = self.model,
            model_kwargs = self.model_kwargs,
            encode_kwargs = self.encode_kwargs
        )

    def tiktoken_len(self, text):
        """텍스트를 토큰으로 변환 후 길이를 반환하는 함수"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
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
        embeddings = self.hf.embed_documents(texts)
        return np.array(embeddings)
    
    def create_faiss_index_and_search(self, embeddings, k = 5):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype = np.float32))

        nearest_neighbors = []
        for i in range(embeddings.shape[0]):
            D, I = self.index.search(np.expand_dims(embeddings[i], axis=0), k + 1)
            neighbors = I[0][1:k+1]  # 자기 자신 제외
            distances = D[0][1:k+1]
            nearest_neighbors.append((neighbors, distances))

        return nearest_neighbors

    def create_retriever(self, file_path):
        """텍스트 파일을 로드하고 리트리버를 생성하는 함수"""
        texts = self.text_split(file_path)
        embeddings = self.get_text_embeddings(texts)
        # nearest_neighbors = self.create_faiss_index_and_search(embeddings)

        vectorestore = FAISS.from_texts(texts, embeddings)
        faiss_retriever = vectorestore.as_retriever(search_k = 20, k = 1)

        return faiss_retriever

