"""
Playwright 기반 웹앱 스크린샷 캡처 스크립트.
Django(:8000) + Flask(:5000)이 실행 중인 상태에서 실행합니다.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

BASE_URL = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).parent.parent / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESKTOP_W, DESKTOP_H = 1440, 900
MOBILE_W, MOBILE_H = 390, 844


def shot(page: Page, name: str, full: bool = False) -> None:
    path = str(OUT_DIR / name)
    page.screenshot(path=path, full_page=full)
    print(f"  ✓ {name}")


def shot_element(page: Page, selector: str, name: str) -> None:
    el = page.query_selector(selector)
    if el:
        el.screenshot(path=str(OUT_DIR / name))
        print(f"  ✓ {name}")
    else:
        shot(page, name)


def wait_done(page: Page, timeout: int = 45000) -> None:
    """status-pill 이 '완료' 또는 오류 문구가 될 때까지 대기."""
    try:
        page.wait_for_function(
            """() => {
                const el = document.getElementById('status-pill');
                if (!el) return false;
                const t = el.textContent.trim();
                return t === '완료' || t.includes('오류') || t.includes('Error') || t.includes('실패');
            }""",
            timeout=timeout,
        )
    except Exception:
        pass
    time.sleep(0.8)


def open_page(browser, width: int = DESKTOP_W, height: int = DESKTOP_H) -> Page:
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(BASE_URL, wait_until="domcontentloaded")
    # Tailwind + Chart.js 로드 대기
    page.wait_for_function(
        "document.querySelectorAll('.panel').length > 0",
        timeout=12000,
    )
    time.sleep(1.2)
    return page


def fill_yfinance(page: Page) -> None:
    """yfinance 소스로 삼성전자(005930.KS) 세팅."""
    page.fill("#ticker", "005930.KS")
    page.fill("#tickers", "005930.KS,000660.KS,035420.KS,051910.KS,068270.KS")
    page.select_option("#source", "yfinance")
    page.fill("#pages", "30")
    page.fill("#period", "3y")


def click_btn(page: Page, action: str) -> None:
    page.click(f'button[data-action="{action}"]')


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage", "--force-color-profile=srgb"],
        )

        # ── 01: 메인 홈 (초기 로드 상태) ─────────────────────────────
        print("\n[01] 메인 홈")
        page = open_page(browser)
        shot(page, "01_main_home.png")

        # ── 02: 풀페이지 ─────────────────────────────────────────────
        print("[02] 풀페이지")
        shot(page, "02_main_fullpage.png", full=True)

        # ── 03: 입력 패널 (왼쪽 aside 전체) ──────────────────────────
        print("[03] 입력 패널")
        shot_element(page, "aside.space-y-5", "03_input_panel.png")

        # ── 04: 결과 패널 초기 상태 (오른쪽 section) ─────────────────
        print("[04] 결과 패널 초기")
        shot_element(
            page,
            "section.workspace-grid > section.space-y-5",
            "04_result_panel_initial.png",
        )
        page.close()

        # ── 05: 크롤 실행 중 상태 캡처 ───────────────────────────────
        print("[05] 크롤 로딩 상태")
        page = open_page(browser)
        click_btn(page, "crawl")
        time.sleep(0.5)   # "실행 중" 상태를 캡처
        shot(page, "05_crawl_loading.png")
        wait_done(page, timeout=20000)
        page.close()

        # ── 06: ML 예측 결과 (yfinance, 삼성전자 RF) ─────────────────
        print("[06] ML 예측 결과")
        page = open_page(browser)
        fill_yfinance(page)
        click_btn(page, "ml")
        wait_done(page, timeout=60000)
        time.sleep(0.5)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
        shot(page, "06_ml_result.png")
        page.close()

        # ── 07: DL 예측 결과 (yfinance, MLP) ────────────────────────
        print("[07] DL 예측 결과")
        page = open_page(browser)
        fill_yfinance(page)
        click_btn(page, "dl")
        wait_done(page, timeout=60000)
        time.sleep(0.5)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
        shot(page, "07_dl_result.png")
        page.close()

        # ── 08: Forecast 결과 (차트 포함, yfinance) ──────────────────
        print("[08] Forecast (주가 예측 리포트 + 차트)")
        page = open_page(browser)
        fill_yfinance(page)
        click_btn(page, "forecast")
        wait_done(page, timeout=60000)
        time.sleep(1.2)   # 차트 렌더링 대기
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
        shot(page, "08_cluster_result.png")
        page.close()

        # ── 09: MongoDB CRUD 콘솔 ─────────────────────────────────────
        print("[09] MongoDB CRUD 콘솔")
        page = open_page(browser)
        # 사용자 목록 조회 → 결과 확인
        page.click('[data-mongo-action="mongo-list-users"]')
        wait_done(page, timeout=10000)
        # 헤더까지 포함한 전체 화면 캡처
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.4)
        shot(page, "09_mongodb_section.png")
        page.close()

        # ── 10: 모바일 뷰 (390px) ────────────────────────────────────
        print("[10] 모바일 뷰")
        page = open_page(browser, width=MOBILE_W, height=MOBILE_H)
        shot(page, "10_mobile_view.png")
        page.close()

        browser.close()
        print(f"\n완료 → {OUT_DIR}")


if __name__ == "__main__":
    run()
