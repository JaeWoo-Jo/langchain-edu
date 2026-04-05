import argparse
from pathlib import Path
import urllib3

from langchain_core.documents import Document
from json_loader import load_recipe_json, load_nutrition_json
from pdf_loader import load_pdf
from hwp_loader import load_hwp
from chunker import chunk_documents
from embedder import embed_texts
from es_client import index_documents
from index_mapping import get_es_client, create_index
from config import ES_INDEX

# self-signed 인증서 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".json", ".txt"}


def collect_files(data_dir: str, source: str = "all") -> list[Path]:
    """지원하는 확장자의 파일을 재귀적으로 수집한다."""
    from config import RECIPES_DIR, NUTRITION_DIR, INGREDIENTS_DIR
    if source == "recipes":
        search_dir = RECIPES_DIR
    elif source == "nutrition":
        search_dir = NUTRITION_DIR
    elif source == "ingredients":
        search_dir = INGREDIENTS_DIR
    else:
        # all: 가격 도메인 데이터 디렉토리만 스캔 (의료 문서 제외)
        files = []
        for d in [RECIPES_DIR, NUTRITION_DIR, INGREDIENTS_DIR]:
            if d.exists():
                for ext in SUPPORTED_EXTENSIONS:
                    files.extend(d.rglob(f"*{ext}"))
        return sorted(files)

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(search_dir.rglob(f"*{ext}"))
    return sorted(files)


def load_file(file_path: Path) -> list:
    """확장자에 따라 적절한 로더를 선택하여 파일을 로드한다."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".hwp":
        return load_hwp(file_path)
    elif ext == ".json":
        # 상위 디렉토리명으로 로더 판별
        if "recipes" in str(file_path):
            return load_recipe_json(file_path)
        elif "nutrition" in str(file_path):
            return load_nutrition_json(file_path)
        else:
            return load_recipe_json(file_path)
    elif ext == ".txt":
        text = file_path.read_text(encoding="utf-8")
        return [Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "source_type": "ingredient",
                "category": "",
                "page": 0,
            },
        )]
    else:
        print(f"지원하지 않는 파일 형식: {ext}")
        return []


def process_file(es, file_path: Path, chunk_size: int, chunk_overlap: int) -> int:
    """단일 파일을 파싱 → 청킹 → 임베딩 → ES 적재한다."""
    documents = load_file(file_path)
    if not documents:
        print(f"  내용 없음, 건너뜀")
        return 0

    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print(f"  청크 없음, 건너뜀")
        return 0

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts(texts)
    return index_documents(es, chunks, embeddings)


def run_pipeline(files: list[Path], chunk_size: int = 500, chunk_overlap: int = 100, recreate_index: bool = False):
    """파일 리스트를 하나씩 처리하는 파이프라인"""

    # ES 클라이언트 연결 & 인덱스 준비
    print("=" * 60)
    print("Elasticsearch 연결")
    es = get_es_client()
    print(f"  ES 연결 확인: {es.info()['version']['number']}")
    create_index(es, delete_existing=recreate_index)

    print(f"\n처리 대상: {len(files)}개 파일")
    print("=" * 60)

    # 파일별 처리
    total_success = 0
    failed_files: list[str] = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path}")
        try:
            success = process_file(es, file_path, chunk_size, chunk_overlap)
            total_success += success
        except Exception as e:
            print(f"  [오류] {e}")
            failed_files.append(str(file_path))

    # 결과 확인
    print("\n" + "=" * 60)
    es.indices.refresh(index=ES_INDEX)
    count = es.count(index=ES_INDEX)["count"]
    print(f"전체 완료: {total_success}건 적재, 인덱스 내 총 문서 수: {count}")
    if failed_files:
        print(f"\n실패한 파일 ({len(failed_files)}개):")
        for f in failed_files:
            print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description="PDF/HWP to Elasticsearch 파이프라인")
    parser.add_argument("files", nargs="*", help="처리할 파일 경로 (지정하지 않으면 --data-dir 전체 처리)")
    parser.add_argument("--data-dir", default="data", help="파일이 있는 디렉토리 경로 (기본: data)")
    parser.add_argument("--chunk-size", type=int, default=500, help="청크 크기 (기본: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="청크 오버랩 (기본: 100)")
    parser.add_argument("--recreate-index", action="store_true", help="기존 인덱스를 삭제하고 재생성")
    parser.add_argument("--source", default="all",
                        choices=["all", "recipes", "nutrition", "ingredients"],
                        help="처리할 데이터 소스")
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = collect_files(args.data_dir, source=args.source)

    run_pipeline(
        files=files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate_index=args.recreate_index,
    )


if __name__ == "__main__":
    main()
