# document를 받기 전 prompt engineering만 진행한 코드

from dotenv import load_dotenv
load_dotenv()  # dotenv 파일에서 API KEY 불러오기

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from utils_.retriever import CustomRetriever_Faiss

# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

import warnings 
warnings.filterwarnings("ignore")

def get_response(query):

    # 모델 불러오기
    chat_model = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature= 1
    )

    # 템플릿 정의
    template = """
    너는 연애에 대해 조언을 해주는 연애 상담사 SOL-T야.
    내가 연인과 처한 문제 상황을 듣고, 내 연인에게 공감하여 해결책을 제시해 줘야 해.
    이모티콘을 사용해서 상냥한 말투와 존댓말을 사용해 줘.
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

    # 이전 대화 기록 & 저장하는 buffer memory. -> 미구현
    # memory = ConversationBufferMemory(return_messages = True)
    # memory.load_memory_variables({"siuation", "chat_history"})
    custom_retriever = CustomRetriever_Faiss()
    # retriever = custom_retriever.create_retriever(r'data/filtered_emotion.csv')

    # LLMChain 생성 
    chain = LLMChain(
        llm=chat_model,
        prompt=chat_prompt,
        # retriever = retriever,
    )

    # 질의에 대한 응답 생성 - ver1 (LLMChain)
    result = chain.run(situation=query) 
    # 결과 출력
    return result

    # response = chain.run(
    #   input = query
    # )
    # return response
