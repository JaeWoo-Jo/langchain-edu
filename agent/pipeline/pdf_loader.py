from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """PDF 파일을 로드하여 페이지별 Document 리스트를 반환한다."""
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    print(f"  '{file_path.name}' 로드 완료: {len(documents)}페이지")
    return documents
