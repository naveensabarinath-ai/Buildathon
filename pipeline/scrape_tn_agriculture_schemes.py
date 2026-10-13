"""
Scrape Tamil Nadu Agriculture scheme listings and scheme-detail modal tables.

Approach:
- Playwright opens the portal and runs the site's own getSchemeDet(id) function.
- BeautifulSoup parses listing/detail HTML.
- JSON/report files are saved after each scheme so progress is not lost.

Install once:
    python -m pip install playwright beautifulsoup4
    python -m playwright install chromium

Run:
    python pipeline\scrape_tn_agriculture_schemes.py
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
URL = "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"
OUTPUT_JSON = DATA_DIR / "all_tn_agriculture_schemes.json"
OUTPUT_REPORT = DATA_DIR / "scheme_scrape_report.txt"

PAGE_TIMEOUT_MS = 15000
MODAL_TIMEOUT_MS = 12000
MAX_DETAIL_PAGES = 100


def clean_text(value):
    return " ".join((value or "").split())


def save_progress(records):
    """Persist progress after each scheme."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    success = sum(x.get("detail_status") == "success" for x in records)
    empty = sum(x.get("detail_status") == "empty" for x in records)
    errors = sum(x.get("detail_status") == "error" for x in records)

    lines = [
        f"Total schemes processed: {len(records)}",
        f"With detail records: {success}",
        f"Empty detail tables: {empty}",
        f"Errors: {errors}",
        "",
        "SCHEME-BY-SCHEME REPORT",
    ]
    for item in records:
        lines.append(
            f"{item.get('scheme_name', '')} | "
            f"ID={item.get('scheme_id', '')} | "
            f"Details={item.get('detail_record_count', 0)} | "
            f"Status={item.get('detail_status', '')}"
        )
        if item.get("error"):
            lines.append(f"    Error: {item['error']}")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def set_main_page_size(page):
    """Choose the largest common page-size option, if the listing provides one."""
    for dropdown in page.locator("select").all():
        try:
            options = [
                clean_text(x)
                for x in dropdown.locator("option").all_inner_texts()
            ]
            numeric = []
            for option in options:
                match = re.search(r"\d+", option)
                if match:
                    numeric.append((int(match.group()), option))
            if numeric:
                largest_value, largest_label = max(numeric)
                if largest_value >= 50:
                    dropdown.select_option(label=largest_label)
                    page.wait_for_timeout(500)
                    print(f"Selected main-list page size: {largest_label}")
                    return
        except Exception:
            continue


def get_main_table_soup(page):
    soup = BeautifulSoup(page.content(), "html.parser")
    for table in soup.find_all("table"):
        headers = [
            clean_text(th.get_text(" ", strip=True))
            for th in table.find_all("th")
        ]
        if "Scheme" in headers and "Scheme Description" in headers:
            return table, headers
    raise RuntimeError("Could not find the main scheme listing table.")


def get_main_records(page):
    """Read listing columns and preserve PDF links as absolute URLs."""
    table, headers = get_main_table_soup(page)
    records = []

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != len(headers):
            continue

        values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        record = dict(zip(headers, values))
        if not record.get("Scheme"):
            continue

        pdf_links = []
        for cell in cells:
            for anchor in cell.find_all("a", href=True):
                href = anchor.get("href", "").strip()
                if ".pdf" in href.lower():
                    pdf_links.append({
                        "link_text": clean_text(anchor.get_text(" ", strip=True)),
                        "url": urljoin(URL, href),
                    })
        record["PDF Links"] = pdf_links
        records.append(record)

    return records


def get_scheme_links(page):
    """Return stable values rather than retaining DOM locators during modal activity."""
    found = []
    anchors = page.locator('a[onclick*="getSchemeDet"]')
    for i in range(anchors.count()):
        anchor = anchors.nth(i)
        try:
            name = clean_text(anchor.inner_text(timeout=3000))
            handler = anchor.get_attribute("onclick") or ""
            match = re.search(r"getSchemeDet\s*\(\s*['\"]?(\d+)", handler)
            if not match:
                match = re.search(r"getSchemeDet.*?(\d+)", handler)
            if name and match:
                found.append({"scheme_name": name, "scheme_id": match.group(1)})
        except Exception:
            continue

    # Remove duplicate IDs while preserving order.
    unique = []
    seen = set()
    for item in found:
        if item["scheme_id"] not in seen:
            unique.append(item)
            seen.add(item["scheme_id"])
    return unique


def find_detail_modal(page):
    """Find the visible dialog containing the expected title."""
    selectors = [
        "[role='dialog']",
        ".ui-dialog",
        ".modal.show",
        ".modal",
    ]
    for selector in selectors:
        locator = page.locator(selector)
        for i in range(locator.count()):
            element = locator.nth(i)
            try:
                if element.is_visible() and "Scheme Details View" in element.inner_text(timeout=1500):
                    return element
            except Exception:
                continue
    raise RuntimeError("The 'Scheme Details View' dialog did not appear.")


def wait_for_detail_modal(page):
    """Wait briefly for the portal's AJAX-loaded detail modal."""
    deadline = time.time() + MODAL_TIMEOUT_MS / 1000
    last_error = None
    while time.time() < deadline:
        try:
            return find_detail_modal(page)
        except Exception as exc:
            last_error = exc
            page.wait_for_timeout(200)
    raise RuntimeError(f"Detail modal did not appear: {last_error}")


def parse_detail_table(modal):
    """Parse the detail table currently rendered in the modal."""
    soup = BeautifulSoup(modal.inner_html(), "html.parser")

    for table in soup.find_all("table"):
        headers = [
            clean_text(th.get_text(" ", strip=True))
            for th in table.find_all("th")
        ]
        if not any("Input Type" in header for header in headers):
            continue

        rows = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != len(headers):
                continue
            values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
            if any(values):
                rows.append(dict(zip(headers, values)))
        return rows

    return []


def detail_table_signature(modal):
    """A short signature used to detect whether pagination changed the table."""
    rows = parse_detail_table(modal)
    return json.dumps(rows[:3], ensure_ascii=False, sort_keys=True), len(rows)


def click_detail_next(page, modal):
    """
    Click a Next control scoped to the modal using DOM click (not Playwright's
    stability checks, which were timing out on the animated jQuery UI dialog).
    Returns True if a plausible enabled Next control was clicked.
    """
    try:
        result = modal.evaluate(
            """root => {
                const candidates = Array.from(root.querySelectorAll('a, button, input[type="button"]'));
                const next = candidates.find(el => {
                    const text = ((el.innerText || el.value || el.getAttribute('aria-label') || '') + '').trim();
                    return /^next$/i.test(text) || /next page/i.test(text);
                });
                if (!next) return {clicked: false, reason: 'no-next-control'};
                const cls = (next.className || '').toString().toLowerCase();
                const parentCls = next.parentElement
                    ? (next.parentElement.className || '').toString().toLowerCase()
                    : '';
                if (next.disabled ||
                    next.getAttribute('aria-disabled') === 'true' ||
                    cls.includes('disabled') || parentCls.includes('disabled')) {
                    return {clicked: false, reason: 'disabled'};
                }
                next.click();
                return {clicked: true, reason: 'clicked'};
            }"""
        )
        return bool(result and result.get("clicked"))
    except Exception:
        return False


def extract_all_detail_pages(page, modal):
    """Collect every page of the modal detail table and deduplicate rows."""
    all_rows = []
    seen_rows = set()

    for page_number in range(1, MAX_DETAIL_PAGES + 1):
        # Re-find the modal after a table redraw; the old locator may go stale.
        modal = wait_for_detail_modal(page)
        rows = parse_detail_table(modal)

        for row in rows:
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if key not in seen_rows:
                seen_rows.add(key)
                all_rows.append(row)

        before_signature = detail_table_signature(modal)
        if not click_detail_next(page, modal):
            break

        # Wait for pagination/table redraw without long default click timeouts.
        changed = False
        deadline = time.time() + 4
        while time.time() < deadline:
            page.wait_for_timeout(150)
            try:
                refreshed = wait_for_detail_modal(page)
                after_signature = detail_table_signature(refreshed)
                if after_signature != before_signature:
                    changed = True
                    break
            except Exception:
                pass

        if not changed:
            # A control that did not change the rows is not safe to keep clicking.
            break
    else:
        print(f"WARNING: reached MAX_DETAIL_PAGES={MAX_DETAIL_PAGES}")

    return all_rows


def close_modal(page, modal):
    """Close the jQuery UI dialog with native DOM click to avoid animation timeouts."""
    closed = False
    try:
        result = modal.evaluate(
            """root => {
                const close = root.querySelector(
                    'button.ui-dialog-titlebar-close, button[title="Close"], .ui-dialog-titlebar-close'
                );
                if (close) {
                    close.click();
                    return true;
                }
                return false;
            }"""
        )
        closed = bool(result)
    except Exception:
        pass

    if not closed:
        # Fallback: find the visible dialog wrapper globally and click its close button.
        try:
            page.evaluate(
                """() => {
                    const dialogs = Array.from(document.querySelectorAll('.ui-dialog, [role="dialog"]'));
                    const dialog = dialogs.reverse().find(el =>
                        el.offsetParent !== null && (el.innerText || '').includes('Scheme Details View')
                    );
                    const close = dialog && dialog.querySelector(
                        'button.ui-dialog-titlebar-close, button[title="Close"], .ui-dialog-titlebar-close'
                    );
                    if (close) close.click();
                }"""
            )
        except Exception:
            pass

    # Do not wait for animations or use a Playwright click on the moving close button.
    page.wait_for_timeout(150)


def scrape_scheme_details(page, scheme_id, expected_name):
    """Call the site's own JS handler, parse all modal pages, and close the dialog."""
    try:
        # This is the same function used by the Scheme link's onclick handler.
        page.evaluate(
            """id => {
                if (typeof window.getSchemeDet !== 'function') {
                    throw new Error('window.getSchemeDet is not available');
                }
                window.getSchemeDet(Number(id));
            }""",
            str(scheme_id),
        )

        modal = wait_for_detail_modal(page)
        details = extract_all_detail_pages(page, modal)
        close_modal(page, modal)

        return {
            "scheme_name": expected_name,
            "scheme_id": str(scheme_id),
            "detail_record_count": len(details),
            "details": details,
            "detail_status": "success" if details else "empty",
        }
    except Exception:
        # Best-effort cleanup before the caller records the failure.
        try:
            modal = find_detail_modal(page)
            close_modal(page, modal)
        except Exception:
            pass
        raise


def match_listing_record(main_records, scheme_name):
    target = clean_text(scheme_name).casefold()
    for item in main_records:
        if clean_text(item.get("Scheme", "")).casefold() == target:
            return item
    # Fallback for punctuation/spacing differences (never match empty names).
    if not target:
        return {}
    for item in main_records:
        name = clean_text(item.get("Scheme", "")).casefold()
        if name and (target in name or name in target):
            return item
    return {}


def main():
    all_schemes = []

    print("Opening Tamil Nadu Agriculture portal...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)
            set_main_page_size(page)

            main_records = get_main_records(page)
            scheme_links = get_scheme_links(page)

            print(f"Listing rows: {len(main_records)}")
            print(f"Scheme detail links: {len(scheme_links)}")

            if not scheme_links:
                raise RuntimeError(
                    "No scheme detail links found. Confirm the portal loaded correctly."
                )

            # Match records by scheme name. On this portal, selecting 100 rows
            # normally exposes all 52 schemes on the same main listing page.
            for index, link in enumerate(scheme_links, start=1):
                scheme_name = link["scheme_name"]
                scheme_id = link["scheme_id"]
                print(f"\n[{index}/{len(scheme_links)}] {scheme_name} (ID: {scheme_id})")

                try:
                    detail = scrape_scheme_details(page, scheme_id, scheme_name)
                    listing = match_listing_record(main_records, scheme_name)
                    item = {
                        "scheme_name": scheme_name,
                        "scheme_id": scheme_id,
                        "listing": listing,
                        "detail_record_count": detail["detail_record_count"],
                        "details": detail["details"],
                        "detail_status": detail["detail_status"],
                    }
                    print(
                        f"  Detail rows collected: {item['detail_record_count']} "
                        f"| Status: {item['detail_status']}"
                    )
                except Exception as exc:
                    listing = match_listing_record(main_records, scheme_name)
                    item = {
                        "scheme_name": scheme_name,
                        "scheme_id": scheme_id,
                        "listing": listing,
                        "detail_record_count": 0,
                        "details": [],
                        "detail_status": "error",
                        "error": str(exc),
                    }
                    print(f"  ERROR: {exc}")

                all_schemes.append(item)
                save_progress(all_schemes)

            print("\n========== FINAL REPORT ==========")
            print(OUTPUT_REPORT.read_text(encoding="utf-8"))
            print(f"\nJSON saved to: {OUTPUT_JSON.resolve()}")
            print(f"Report saved to: {OUTPUT_REPORT.resolve()}")

        finally:
            input("\nPress Enter to close the browser...")
            browser.close()


if __name__ == "__main__":
    main()
