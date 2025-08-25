#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy Portal (korea.kr) 정책/인사이트 RSS에서 AI 관련 기사만 수집해 CSV/JSONL로 저장
- 기본 소스(기본값): https://www.korea.kr/rss/policy.xml, https://www.korea.kr/rss/insight.xml
- .env 최상단에 RSS_URLS="https://www.korea.kr/rss/policy.xml,https://www.korea.kr/rss/insight.xml" 형식으로 재정의 가능
- 필터: 다국어 AI 키워드(한국어/영어 혼합)
- 저장: data/policy_ai_news.csv, data/policy_ai_news.jsonl
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from dotenv import load_dotenv  # type: ignore
load_dotenv()

USE_FEEDPARSER = True
import feedparser  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# optional requests for robust fetching
import requests  # type: ignore 

# .env 최상단에 RSS_URLS="https://www.korea.kr/rss/policy.xml,https://www.korea.kr/rss/insight.xml" 형식으로 저장
RSS_URLS = [
    u.strip() for u in os.getenv(
        "RSS_URLS",
        "https://www.korea.kr/rss/policy.xml,https://www.korea.kr/rss/insight.xml",
    ).split(",") if u.strip()
]
suffix = datetime.now().strftime("%Y%m%d%H")
OUT_DIR = os.getenv("NEWS_OUT_DIR", "data")
CSV_PATH = os.getenv("NEWS_CSV_PATH", os.path.join(OUT_DIR, "ai_news.csv"))
JSONL_PATH = os.getenv("NEWS_JSONL_PATH", os.path.join(OUT_DIR, "ai_news.jsonl"))

# 날짜 범위 설정: 기본 30일(한 달)
DAYS_BACK = int(os.getenv("DAYS_BACK", "30"))

def reset_output_files() -> None:
    """기존 출력 파일(CSV/JSONL)이 있으면 삭제하여 새로 작성하도록 준비."""
    try:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
    except Exception:
        pass
    try:
        if os.path.exists(JSONL_PATH):
            os.remove(JSONL_PATH)
    except Exception:
        pass

# URL별 source 명 생성
def source_alias(url: str) -> str:
    u = url.lower()
    if "insight" in u:
        return "korea.kr_insight"
    if "policy" in u:
        return "korea.kr_policy"
    if "blog.google" in u:
        return "google_ai_blog"
    if "openai.com" in u:
        return "openai_news"
    return "korea.kr"

# ✅ AI 키워드 사전 (한국어/영어 혼합, 정규식 안전 처리)
AI_KEYWORDS = [
    # Korean
    r"\b인공지능\b", r"\b생성형\s*AI\b", r"\b초거대\s*AI\b", r"\b메가\s*모델\b",
    r"\b국가\s*AI\s*전략\b", r"\bAI\s*윤리\b", r"\bAI\s*거버넌스\b",
    r"\bAI\s*규제\b", r"\bAI\s*샌드박스\b", r"\b메가샌드박스\b", r"\b주권\s*AI\b",
    r"\b온디바이스\s*AI\b", r"\b엣지\s*AI\b", r"\bAI\s*반도체\b", r"\b뉴럴\s*프로세서\b",
    r"\bAI\s*안전\b", r"\bAI\s*검증\b", r"\bAI\s*평가\b", r"\bAI\s*허가\b",
    r"\b데이터\s*거버넌스\b", r"\b데이터\s*플랫폼\b",
    # English
    r"\bAI\b", r"\bAIOps\b", r"\bGenAI\b", r"\bgenerative\s*AI\b",
    r"\bfoundation\s*model(s)?\b", r"\blarge\s*language\s*model(s)?\b", r"\bLLM(s)?\b",
    r"\bAI\s*safety\b", r"\bAI\s*governance\b", r"\bAI\s*ethic(s)?\b",
    r"\bAI\s*sandbox\b", r"\bsovereign\s*AI\b",
    r"\bRAG\b", r"\bvector\s*(DB|database)\b", r"\bAI\s*chip(s)?\b", r"\bNPU\b",
]

KEYWORD_REGEX = re.compile("|".join(AI_KEYWORDS), flags=re.IGNORECASE)

def sanitize_html(text: str) -> str:
    if not text:
        return ""
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    # Fallback: 매우 단순한 태그 제거
    return re.sub(r"<[^>]+>", " ", text).strip()

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def fetch_rss(url: str, max_retries: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
    """
    RSS 피드 파싱(기본: feedparser). 실패 시 예외 발생.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if USE_FEEDPARSER:
                # Prefer fetching bytes ourselves to avoid encoding pitfalls
                ua = os.getenv(
                    "HTTP_USER_AGENT",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                )
                headers = {
                    "User-Agent": ua,
                    "Accept": "application/rss+xml, application/xml, text/xml, */*",
                }
                content_bytes = None
                # Try requests if available
                if requests is not None:
                    resp = requests.get(url, headers=headers, timeout=15)
                    resp.raise_for_status()
                    content_bytes = resp.content
                else:
                    import urllib.request
                    req = urllib.request.Request(url)
                    req.add_header("User-Agent", ua)
                    req.add_header("Accept", "application/rss+xml, application/xml, text/xml, */*")
                    with urllib.request.urlopen(req, timeout=15) as r:
                        content_bytes = r.read()

                # First try parsing bytes directly
                feed = feedparser.parse(content_bytes)
                if not getattr(feed, "bozo", False):
                    return feed

                # Retry with utf-8 decoded text ignoring errors
                text_utf8 = content_bytes.decode("utf-8", errors="ignore")
                feed2 = feedparser.parse(text_utf8)
                if not getattr(feed2, "bozo", False):
                    return feed2

                # If still failing, raise to trigger fallback path below
                raise RuntimeError(
                    f"Feed parse error: {getattr(feed2, 'bozo_exception', None) or getattr(feed, 'bozo_exception', None)}"
                )
            else:
                # 표준 라이브러리 폴백(간단 파서)
                import urllib.request
                import xml.etree.ElementTree as ET
                
                # User-Agent 헤더 추가 (403 Forbidden 방지)
                req = urllib.request.Request(url)
                req.add_header('User-Agent', os.getenv(
                    'HTTP_USER_AGENT',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                ))
                req.add_header('Accept', 'application/rss+xml, application/xml, text/xml, */*')
                
                with urllib.request.urlopen(req, timeout=15) as resp:
                    xml_bytes = resp.read()
                # Best effort: try declared encoding, else utf-8
                try:
                    root = ET.fromstring(xml_bytes)
                except Exception:
                    root = ET.fromstring(xml_bytes.decode('utf-8', errors='ignore'))
                channel = root.find("channel")
                items = channel.findall("item") if channel is not None else []
                # feedparser와 유사 형태로 맞춰 반환
                entries = []
                for it in items:
                    entry = {
                        "title": (it.findtext("title") or "").strip(),
                        "link": (it.findtext("link") or "").strip(),
                        "description": (it.findtext("description") or "").strip(),
                        "published": (it.findtext("pubDate") or "").strip(),
                        "guid": (it.findtext("guid") or "").strip(),
                    }
                    entries.append(entry)
                return {"entries": entries}
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise last_err  # 불도달

def to_iso8601(dt_str: str) -> str:
    """
    pubDate 등 다양한 날짜 포맷을 최대한 ISO8601로 정규화.
    """
    if not dt_str:
        return ""
    # feedparser 사용 시 이미 parsed가 제공될 수 있음
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    # 포맷 추론 실패 시 원문 반환(유지)
    return dt_str

def _parse_iso_to_date(iso_str: str):
    """ISO8601 문자열을 date로 변환. 실패 시 None 반환."""
    if not iso_str:
        return None
    try:
        if "T" in iso_str:
            # Z 처리
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.date()
        # 날짜만 있는 경우
        return datetime.strptime(iso_str, "%Y-%m-%d").date()
    except Exception:
        return None

def filter_recent(records: List[Dict[str, Any]], days_back: int = 30) -> List[Dict[str, Any]]:
    """오늘 기준 최근 days_back 일에 해당하는 레코드만 남긴다."""
    if days_back <= 0:
        return records
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_back)
    out = []
    for r in records:
        d = _parse_iso_to_date(r.get("published", ""))
        if d is None or d >= start_date:
            out.append(r)
    return out

def entry_to_record(entry: Any, source_alias_val: str) -> Dict[str, Any]:
    # feedparser 형태와 폴백 형태 모두 대응
    title = normalize_space(entry.get("title", ""))
    link = entry.get("link", "") or entry.get("id", "")
    summary = entry.get("summary", "") or entry.get("description", "")
    summary = sanitize_html(summary)
    published = entry.get("published", "") or entry.get("updated", "")
    published_iso = to_iso8601(published)
    guid = entry.get("guid", "") or entry.get("id", "") or link

    # 카테고리/태그
    tags = []
    if "tags" in entry and isinstance(entry["tags"], list):
        for t in entry["tags"]:
            term = t.get("term") if isinstance(t, dict) else None
            if term:
                tags.append(term)

    return {
        "source": source_alias_val,
        "title": title,
        "link": link,
        "published_raw": published,
        "published": published_iso,
        "guid": guid,
        "summary": summary,
        "tags": ";".join(tags) if tags else "",
        "fetched_at": datetime.utcnow().isoformat(),
    }

def is_ai_related(text_blob: str) -> bool:
    return bool(KEYWORD_REGEX.search(text_blob))

def filter_ai(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for e in entries:
        blob = " ".join([
            e.get("title", ""),
            e.get("summary", ""),
            e.get("tags", ""),
            e.get("link", ""),
        ])
        if is_ai_related(blob):
            results.append(e)
    return results

def dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in records:
        key = r.get("guid") or r.get("link") or r.get("title")
        if not key:
            # 해시 키가 전혀 없으면 제목+발행일로 구성
            key = f"{r.get('title','')}|{r.get('published','')}"
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def ensure_outdir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_outdir(path)
    fieldnames = [
        "source", "title", "link", "published_raw", "published",
        "guid", "summary", "tags", "fetched_at"
    ]
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_outdir(path)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> int:
    # 시작 전에 출력 파일 초기화
    reset_output_files()
    # 다중 RSS URL 처리 (.env의 RSS_URLS 사용)
    all_entries = []
    total_fetched = 0
    for url in RSS_URLS:
        try:
            feed = fetch_rss(url)
        except Exception as e:
            print(f"[ERROR] RSS fetch failed: {url} :: {e}", file=sys.stderr)
            continue
        print(f'url: {url}')
        raw_entries = feed.entries if (USE_FEEDPARSER and hasattr(feed, "entries")) else feed.get("entries", [])
        total_fetched += len(raw_entries)
        src = source_alias(url)
        normalized = [entry_to_record(e, src) for e in raw_entries]
        all_entries.extend(normalized)
    # URL별 통계 출력
    url_stats = {}
    for entry in all_entries:
        src = entry['source']
        pub_date = entry.get('published')
        if src not in url_stats:
            url_stats[src] = {
                'count': 0,
                'min_date': None,
                'max_date': None
            }
        url_stats[src]['count'] += 1
        
        if pub_date:
            if not url_stats[src]['min_date'] or pub_date < url_stats[src]['min_date']:
                url_stats[src]['min_date'] = pub_date
            if not url_stats[src]['max_date'] or pub_date > url_stats[src]['max_date']:
                url_stats[src]['max_date'] = pub_date

    print("\n=== URL별 수집 통계 ===")
    for src, stats in url_stats.items():
        date_range = f"{stats['min_date']} ~ {stats['max_date']}" if stats['min_date'] else "날짜 정보 없음"
        print(f"- {src}: {stats['count']}건 ({date_range})")
    print("=" * 30 + "\n")

    # 최근 N일(기본 30일) 필터
    before_filter = len(all_entries)
    all_entries = filter_recent(all_entries, DAYS_BACK)
    print(f"[INFO] 최근 {DAYS_BACK}일 필터링: {before_filter} -> {len(all_entries)}")

    # AI 필터
    ai_only = filter_ai(all_entries)

    # 중복 제거
    ai_only = dedupe(ai_only)

    # 저장
    append_csv(CSV_PATH, ai_only)
    append_jsonl(JSONL_PATH, ai_only)

    # 요약 출력
    print(f"[INFO] total_entries={total_fetched} from {len(RSS_URLS)} feeds | ai_filtered={len(ai_only)} saved_csv='{CSV_PATH}' saved_jsonl='{JSONL_PATH}'")

    #for i, r in enumerate(ai_only[:10], 1):
    #    print(f"{i:02d}. [{r['source']}] {r['title']} | {r['published']} | {r['link']}")

    return 0

if __name__ == "__main__":
    sys.exit(main())