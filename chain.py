from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document


SYSTEM_PROMPT = """당신은 KBO 리그의 공식 규정 및 경기 운영 전문 AI 어시스턴트입니다.
질문에 정확하고 친절하게 답변하세요.

[답변 규칙]
1. 근거 규정 확인 : 질문과 관련된 규정 ID가 [참고 문서]에 있는지 먼저 확인하세요.
1-1. 참고한 규정의 category, sub_category, title, content 순서로 질문과 관련이 있는지 판별하세요.
2. 단계적 추론 :
    - 날짜 계산 시 : [말소일], [마지막 경기일], [소급 가능일], [의무 결장 기간]을 순서대로 따지세요.
    - 복합 상황 시 : 각 규정(ABS, 강우, 서스펜디드)의 적용 우선순위를 먼저 판별하세요.
3. 불확실성 : 문서에 해당 내용이 없거나 관련 없는 ID만 있다면 "죄송합니다. 관련 규정을 찾을 수 없습니다."라고 답변하세요.
4. 참고한 규정 ID를 명시하세요.

[참고 문서]
{context}"""

def build_rag_chain(vectorstore):
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.1-8b-instant"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 53
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])


    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def format_docs(docs: list[Document]) -> str:

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    sections = []

    for i, doc in enumerate(docs, 1):
        category = doc.metadata.get("category", "일반")
        question = doc.metadata.get("question", "")
        section = f"{i}. 카테고리: {category}"
        if question:
            section += f"  - 관련 질문: {question}"
        section += f"\n{doc.page_content}"
        sections.append(section)

    return "\n\n".join(sections)