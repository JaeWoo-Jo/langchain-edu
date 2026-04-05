from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Document 리스트를 청크로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # 청크 인덱스를 메타데이터에 추가
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"청킹 완료: {len(documents)}페이지 → {len(chunks)}청크 (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
