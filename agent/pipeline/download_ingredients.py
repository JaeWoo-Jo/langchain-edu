"""농촌진흥청 농식품올바로에서 식재료 가이드 문서를 크롤링한다.

사용법:
    uv run python download_ingredients.py
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

from config import INGREDIENTS_DIR

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 농촌진흥청 농식품올바로
BASE_URL = "https://koreanfood.rda.go.kr"
# 식재료 정보 게시판 목록 URL
LIST_URL = f"{BASE_URL}/kfi/foodSel/foodSelList.do"

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; edu-pipeline/1.0)"
})


def get_ingredient_pages(max_pages: int = 5) -> list[dict]:
    """식재료 정보 페이지 목록을 수집한다."""
    items: list[dict] = []
    for page_no in range(1, max_pages + 1):
        print(f"  페이지 {page_no} 스캔 중...")
        try:
            resp = SESSION.get(LIST_URL, params={"pageNo": page_no}, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            links = soup.select("a[href*='foodSelView']")
            if not links:
                break

            for link in links:
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if href and title:
                    full_url = BASE_URL + href if href.startswith("/") else href
                    items.append({"title": title, "url": full_url})
        except Exception as e:
            print(f"  페이지 {page_no} 오류: {e}")
        time.sleep(0.5)

    return items


def download_ingredient_page(url: str, title: str, save_dir: Path) -> bool:
    """식재료 상세 페이지의 본문을 텍스트 파일로 저장한다."""
    safe_title = "".join(c if c.isalnum() or c in "가-힣 _-" else "_" for c in title)
    output_path = save_dir / f"{safe_title}.txt"

    if output_path.exists():
        print(f"  [스킵] {title} (이미 존재)")
        return False

    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # 본문 영역 추출 (사이트 구조에 따라 selector 조정 필요)
        content_area = soup.select_one(".view_cont, .cont_area, #contents, article")
        if content_area:
            text = content_area.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        output_path.write_text(f"[{title}]\n\n{text}", encoding="utf-8")
        print(f"  [저장] {title}")
        return True
    except Exception as e:
        print(f"  [오류] {title}: {e}")
        return False


def main():
    INGREDIENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("농촌진흥청 식재료 정보 수집 시작")
    items = get_ingredient_pages(max_pages=5)
    print(f"발견된 식재료 정보: {len(items)}건")

    success = 0
    for item in items:
        if download_ingredient_page(item["url"], item["title"], INGREDIENTS_DIR):
            success += 1
        time.sleep(0.3)

    print(f"\n수집 완료: {success}/{len(items)}건")


if __name__ == "__main__":
    main()
