from vectorstore import load_vector_from_local
from chain import build_rag_chain

# init_vectorstore()

vectorstore = load_vector_from_local()
chain = build_rag_chain(vectorstore=vectorstore)

q1 = "피치클락 위반 시 타자와 투수에게 각각 어떤 페널티가 부여되나요?"
result = chain.invoke(q1)

print(f"---사용자의 질문: {q1}\n{result}\n")

q2 = "A선수가 3일 전 경기에 나갔고 오늘 부상으로 말소됐다면, 소급 적용을 포함해 언제 복귀 가능한가요?"
result = chain.invoke(q2)

print(f"---사용자의 질문: {q2}\n{result}\n")

q3 = "경기 중 ABS가 고장 났고, 5회초에 비가 내려 경기가 중단된 상황이라면?"
result = chain.invoke(q3)

print(f"---사용자의 질문: {q3}\n{result}\n")