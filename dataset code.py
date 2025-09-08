import requests, time
import pandas as pd
import re

API_KEY = "4389de09-33bb-4b11-99f2-5b1531844ce1"
QUERY = '"responsible investment"'


FROM_DATE = "2018-01-01"
TO_DATE   = "2025-08-10"

BASE_URL = "https://content.guardianapis.com/search"
PAGE_SIZE = 50
SLEEP_SEC = 20.0
RETRY_MAX = 3
RETRY_BACKOFF = 3

MAX_PAGES = 10000
MAX_TOTAL = 2500  # Upper limit: prioritize high relevance (order-by=relevance)

def fetch_page(page, from_date, to_date):
    params = {
        "api-key": API_KEY,
        "q": QUERY,
        "page": page,
        "page-size": PAGE_SIZE,
        "order-by": "relevance",
        "type": "article",
        "show-fields": "headline,trailText,byline,bodyText",
        "show-tags": "keyword",
        "from-date": from_date,
        "to-date": to_date,
    }
    # Simple retry mechanism
    last_err = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            last_err = e
            if attempt < RETRY_MAX:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                raise last_err

def normalize(item):
    f = item.get("fields", {}) or {}
    tags = item.get("tags", []) or []
    return {
        "title": item.get("webTitle"),
        "date": item.get("webPublicationDate"),
        "url": item.get("webUrl"),
        "section": item.get("sectionName"),
        "byline": f.get("byline"),
        "trailText": f.get("trailText"),
        "bodyText": f.get("bodyText"),
        "keywords": [t.get("webTitle") for t in tags if t.get("type") == "keyword"],
    }

def _soft_norm_title(t):
    if pd.isna(t): return ""
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def main():
    all_rows, seen = [], set()

    # Probe the first page
    probe = fetch_page(1, FROM_DATE, TO_DATE)
    total_pages = probe.get("pages", 1)
    last_page = min(total_pages, MAX_PAGES)

    # page 1
    for it in probe.get("results", []):
        url = it.get("webUrl")
        if not url or url in seen:
            continue
        seen.add(url)
        row = normalize(it)
        all_rows.append(row)
        if len(all_rows) >= MAX_TOTAL:
            break

    # pages 2..last_page
    page = 2
    while page <= last_page and len(all_rows) < MAX_TOTAL:
        resp = fetch_page(page, FROM_DATE, TO_DATE)
        for it in resp.get("results", []):
            url = it.get("webUrl")
            if not url or url in seen:
                continue
            seen.add(url)
            row = normalize(it)
            all_rows.append(row)
            if len(all_rows) >= MAX_TOTAL:
                break
        page += 1
        if len(all_rows) < MAX_TOTAL:
            time.sleep(SLEEP_SEC)

    print(f"[RAW] collected (pre-dedup by URL) ≈ {len(all_rows)} "
          f"(pages used: ≤{last_page}/{total_pages})")

    # —— Safe deduplication: use URL as primary key; 
    # only fall back to title+date when URL is missing ——
    df = pd.DataFrame(all_rows)
    before = len(df)

    # 1) Deduplicate by URL first (most reliable)
    df_url = df[df["url"].notna()].drop_duplicates(subset=["url"]).copy()

    # 2) For records without URL, use “soft title + date” as fallback dedup key
    df_no_url = df[df["url"].isna()].copy()
    if not df_no_url.empty:
        df_no_url["title_soft"] = df_no_url["title"].map(_soft_norm_title)
        df_no_url["date_only"] = pd.to_datetime(df_no_url["date"], errors="coerce").dt.date.astype(str)
        # Only keep valid rows where both title and date are present
        valid = (df_no_url["title_soft"].astype(bool)) & (df_no_url["date_only"].astype(bool))
        df_no_url = df_no_url[valid].drop_duplicates(subset=["title_soft", "date_only"])

    # 3) Merge both parts
    df = pd.concat([df_url, df_no_url], ignore_index=True)
    after = len(df)
    print(f"[DEDUP] removed {before - after} duplicates; kept {after} unique docs "
          f"(no-url rows kept: {len(df_no_url) if 'df_no_url' in locals() else 0})")

    # —— CSV output (keep url as unique ID, useful for later merges) ——
    csv_rows = [{
        "date": r.get("date"),
        "title": r.get("title"),
        "url": r.get("url"),
        "bodyText": r.get("bodyText"),
        "keywords": "; ".join((r.get("keywords") or [])) if isinstance(r.get("keywords"), list) else "",
        "section": r.get("section"),  # Keep section for potential later analysis
    } for _, r in df.iterrows()]

    df_csv = pd.DataFrame(csv_rows)
    # Filter out rows with empty body text
    df_csv = df_csv[df_csv["bodyText"].astype(bool)].reset_index(drop=True)

    csv_path = "guardian_responsibleinvestment_2018_2025_analytics.csv"
    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_csv)} rows to {csv_path}")

if __name__ == "__main__":
    main()
