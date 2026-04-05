import subprocess
from pathlib import Path

from langchain_core.documents import Document


def load_hwp(file_path: str | Path) -> list[Document]:
    """HWP 파일을 pyhwp(hwp5txt)로 텍스트 추출하여 Document로 반환한다."""
    file_path = Path(file_path)
    result = subprocess.run(
        ["hwp5txt", str(file_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  '{file_path.name}' 파싱 실패: {result.stderr.strip()}")
        return []

    text = result.stdout.strip()
    if not text:
        print(f"  '{file_path.name}' 내용 없음")
        return []

    doc = Document(
        page_content=text,
        metadata={"source": str(file_path), "page": 0},
    )
    print(f"  '{file_path.name}' 로드 완료: {len(text)}자")
    return [doc]
