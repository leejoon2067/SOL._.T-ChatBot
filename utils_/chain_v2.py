# document, retriever 받은 후 수정할 code

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA, LLMChain

from utils_.retriever import TextEmbedding_Faiss, Customdb_Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import warnings 
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()  # dotenv 파일에서 API KEY 불러오기

# 사용자 입력을 받습니다.
def get_response2(query):
    # 모델 불러오기
    chat_model = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.7
    )

    # 템플릿 정의
    template = """
    너는 연애에 대해 조언해 줄 수 있는 상담사 SOL-T(쏠티)야. 
    내가 연인과 처한 문제 상황을 듣고, 연인에게 공감하여 해결책을 제시해 줘.
    내가 처한 문제 상황은 다음과 같아.

    내가 연인과 처한 문제 상황을 듣고, 내 연인에게 공감하여 해결책을 제시해 줘야 해.
    이모티콘을 사용해서 상냥한 말투를 사용해 줘. 대신 존댓말은 유지해야 해. 
    내가 처한 문제 상황은 다음과 같아.

    {situation}

    만약, 상황을 다 파악했다면 내가 처한 상황을 분석해서 내가 어떻게 해야 하는지 메세지 예시를 포함하여 설명해줘.
    그렇지 않다면, 내 상황을 파악할 수 있는 추가 질문을 해도 좋아.
    
    """

    # 시스템 메시지 프롬프트 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # 사용자 메시지 프롬프트 생성
    human_template = "{situation}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # ChatPromptTemplate 생성
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 프롬프트 메시지 형식 설정
    # formatted_messages = chat_prompt.format_messages(situation=query)

    # Retrieval 체인을 구성하여 프롬프트와 모델을 묶기
    faiss_embed = TextEmbedding_Faiss()    
    csv_path = r'data/filtered_emotion.csv'

    texts = faiss_embed.text_split(csv_path)
    embeddings = faiss_embed.get_text_embeddings(texts)

    db3 = Customdb_Chroma.create_db(csv_path)
    ### 여기서부터 수정    

    retriever = Customdb_Chroma.create_retriever(csv_path)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm = chat_model,
        prompt = chat_prompt,
        retriever = retriever,
        output_parser = StrOutputParser()
    )
    
    result = qa_chain.run({"situation" : query})
    return result
