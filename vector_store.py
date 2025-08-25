#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vector_store.py
- scrape_news.py가 만든 JSONL/CSV를 읽어 Vector DB(FAISS)에 적재하고 검색합니다.
- Cosine similarity (정규화 + IndexFlatIP) 기본값
- 청킹 옵션으로 긴 텍스트를 여러 청크로 분할 가능
- .env:
    EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    FAISS_INDEX_PATH=data/news_faiss_cos.index
    FAISS_META_PATH=data/news_faiss_meta.json
- 사용 예:
    # 인덱스 구축(JSONL) - 청킹 없음
    python vector_store.py build 
    # 인덱스 구축(JSONL) - 청킹 적용
    python vector_store.py build 
    # 인덱스 구축(CSV)
    python vector_store.py build
    # 검색
    python vector_store.py search --query "메가샌드박스 전환 영향"
"""

import os
import csv
import json
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from dotenv import load_dotenv  # type: ignore
load_dotenv()


import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Optional: 설정 파일을 env로 주입

from config_loader import load_and_apply  # type: ignore
# -------- 기본 경로/모델 설정 --------

# scrape_news.py 출력 필드와 호환
DEFAULT_FIELDS = [
    "source", "title", "link", "published_raw", "published",
    "guid", "summary", "tags", "fetched_at"
]

# -------- 공통 유틸 --------
def ensure_outdir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (r.get(k, "") or "") for k in reader.fieldnames})
    return rows

def normalize_space(text: str) -> str:
    return " ".join((text or "").split())

def build_corpus_text(r: Dict[str, Any]) -> str:
    parts = [
        r.get("title", ""),
        r.get("summary", ""),
        r.get("tags", ""),
        r.get("source", ""),
        r.get("published", ""),
        r.get("link", ""),
    ]
    return normalize_space(" \n ".join([p for p in parts if p]))

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norm

# -------- 청킹 기능 --------
def split_text_into_chunks(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    RecursiveCharacterTextSplitter를 사용하여 텍스트를 청크로 분할합니다.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 최대 문자 수
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        청크 리스트
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    # RecursiveCharacterTextSplitter 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    # 텍스트 분할
    chunks = text_splitter.split_text(text)
    
    return chunks

def create_chunked_records(records: List[Dict[str, Any]], 
                          chunk_size: Optional[int] = None, 
                          chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    레코드들을 청킹하여 새로운 레코드 리스트를 생성합니다.
    
    Args:
        records: 원본 레코드 리스트
        chunk_size: 청크 크기 (None이면 청킹하지 않음)
        chunk_overlap: 청크 간 겹치는 문자 수
    
    Returns:
        청킹된 레코드 리스트
    """
    if chunk_size is None:
        return records
    
    chunked_records = []
    
    for i, record in enumerate(records):
        text = build_corpus_text(record)
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        
        for j, chunk in enumerate(chunks):
            # 원본 레코드 복사
            chunked_record = record.copy()
            
            # 청크 정보 추가
            chunked_record["_chunk_id"] = f"{i}_{j}"
            chunked_record["_chunk_text"] = chunk
            chunked_record["_total_chunks"] = len(chunks)
            chunked_record["_chunk_index"] = j
            
            chunked_records.append(chunked_record)
    
    return chunked_records

# -------- 메타데이터 로드/저장 --------
def load_or_create_index(d: int) -> Tuple[faiss.Index, Dict[str, Dict[str, Any]]]:
    """
    Cosine 유사도: L2 정규화 + IndexFlatIP
    """
    if os.path.exists(os.getenv("faiss_index_path")):
        index = faiss.read_index(os.getenv("faiss_index_path"))
    else:
        index = faiss.IndexFlatIP(d)
    if os.path.exists(os.getenv("faiss_meta_path")):
        with open(os.getenv("faiss_meta_path"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    return index, meta

def save_index(index: faiss.Index, meta: Dict[str, Dict[str, Any]]) -> None:
    ensure_outdir(os.getenv("faiss_index_path"))
    faiss.write_index(index, os.getenv("faiss_index_path"))
    with open(os.getenv("faiss_meta_path"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# -------- 인덱싱(Upsert) --------
def gen_id(record: Dict[str, Any]) -> str:
    rid = record.get("guid") or record.get("link") or ""
    if rid:
        # 청킹된 경우 청크 ID 추가
        chunk_id = record.get("_chunk_id")
        if chunk_id:
            return f"{rid}_{chunk_id}"
        return rid
    # fallback: 제목|발행일
    title_pub = f"{record.get('title','')}|{record.get('published','')}"
    chunk_id = record.get("_chunk_id")
    if chunk_id:
        return f"{title_pub}_{chunk_id}"
    return title_pub

def filter_by_date(records: List[Dict[str, Any]], 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    날짜 범위로 레코드를 필터링합니다.
    
    Args:
        records: 필터링할 레코드 리스트
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
    
    Returns:
        필터링된 레코드 리스트
    """
    if not start_date and not end_date:
        return records
    
    from datetime import datetime
    
    filtered = []
    for record in records:
        published = record.get("published", "")
        if not published:
            continue
            
        try:
            # ISO 형식 날짜 파싱
            if "T" in published:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
            else:
                pub_date = datetime.strptime(published, "%Y-%m-%d")
            
            # 날짜 범위 체크
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if pub_date.date() < start_dt.date():
                    continue
                    
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if pub_date.date() > end_dt.date():
                    continue
                    
            filtered.append(record)
            
        except (ValueError, TypeError):
            # 날짜 파싱 실패 시 포함
            filtered.append(record)
    
    return filtered

def upsert(index: faiss.Index,
           meta: Dict[str, Dict[str, Any]],
           ids: List[str],
           vecs: np.ndarray,
           records: List[Dict[str, Any]]) -> int:
    """
    이미 존재하는 id는 스킵, 신규만 추가
    """
    to_add = []
    skipped = 0
    updated = 0
    
    for i, rid in enumerate(ids):
        if rid in meta:
            # 기존 레코드와 내용 비교하여 업데이트 여부 결정
            existing = meta[rid]
            current = records[i]
            
            # 내용이 변경되었는지 확인 (제목, 요약, 태그 등)
            if (existing.get("title") != current.get("title") or
                existing.get("summary") != current.get("summary") or
                existing.get("tags") != current.get("tags")):
                # 내용이 변경된 경우 기존 레코드 제거하고 새로 추가
                del meta[rid]
                to_add.append(i)
                updated += 1
            else:
                skipped += 1
        else:
            to_add.append(i)

    if not to_add:
        print(f"[INFO] 모든 레코드가 이미 존재합니다. (skipped={skipped})")
        return 0

    # Cosine: 정규화된 벡터를 IP 인덱스에 추가
    vecs = l2_normalize(vecs.astype("float32"))
    index.add(vecs[to_add])

    # 메타 저장 (추가 순서를 유지하기 위해 dict insertion order 사용)
    for i in to_add:
        rid = ids[i]
        record = records[i]
        
        # 기본 필드
        meta_record = {
            "id": rid,
            "source": record.get("source", ""),
            "title": record.get("title", ""),
            "link": record.get("link", ""),
            "published": record.get("published", ""),
            "summary": record.get("summary", ""),
            "tags": record.get("tags", ""),
            "fetched_at": record.get("fetched_at", "")
        }
        
        # 청킹 정보 추가
        if "_chunk_id" in record:
            meta_record.update({
                "_chunk_id": record["_chunk_id"],
                "_chunk_text": record["_chunk_text"],
                "_total_chunks": record["_total_chunks"],
                "_chunk_index": record["_chunk_index"]
            })
        
        meta[rid] = meta_record
    
    print(f"[INFO] 인덱스 업데이트: added={len(to_add)}, updated={updated}, skipped={skipped}")
    return len(to_add)

# -------- 검색 --------
def search(index: faiss.Index,
           meta: Dict[str, Dict[str, Any]],
           qvec: np.ndarray,
           topk: int = 5) -> List[Dict[str, Any]]:
    if index.ntotal == 0:
        return []
    qvec = l2_normalize(qvec.astype("float32"))
    D, I = index.search(qvec, topk)
    # 포지션 -> id 매핑 (dict insertion order)
    ids_in_order = list(meta.keys())
    results = []
    for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
        if idx < 0 or idx >= len(ids_in_order):
            continue
        rid = ids_in_order[idx]
        rec = dict(meta[rid])
        rec["_rank"] = rank
        rec["_score_ip"] = float(score)  # cosine에 해당 (정규화한 IP)
        results.append(rec)
    return results

# -------- 파이프라인 --------
def build_index() -> None:
    # 0) 입력 경로 결정: 설정/환경변수 및 일반적인 기본 후보 자동 탐색
    path = os.getenv("NEWS_INPUT_PATH")

    # 환경 변수에서 파라미터 로드
    model_name = os.getenv("embedding_model")
    chunk_size = int(os.getenv("chunk_size")) if os.getenv("chunk_size") else 512
    chunk_overlap = int(os.getenv("chunk_overlap")) if os.getenv("chunk_overlap") else 50
    start_date = os.getenv("start_date")
    end_date = os.getenv("end_date")

    print(f"[INFO] 입력 경로: {path}")
    # 1) 데이터 읽기
    rows = read_jsonl(path)
    
    print(f"[INFO] 로드된 레코드 수: {len(rows)}")

    # 2) 날짜 필터링 (선택사항)
    if start_date or end_date:
        original_count = len(rows)
        rows = filter_by_date(rows, start_date, end_date)
        print(f"[INFO] 날짜 필터링: {original_count} → {len(rows)} 레코드")
        if start_date:
            print(f"   시작 날짜: {start_date}")
        if end_date:
            print(f"   종료 날짜: {end_date}")

    # 3) 청킹 처리
    if chunk_size is not None:
        print(f"[INFO] 청킹 적용: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        rows = create_chunked_records(rows, chunk_size, chunk_overlap)
        print(f"[INFO] 청킹 후 레코드 수: {len(rows)}")
    else:
        print("[INFO] 청킹 미적용")

    # 4) 텍스트 준비
    if chunk_size is not None:
        # 청킹된 경우 청크 텍스트 사용
        corpus = [r.get("_chunk_text", build_corpus_text(r)) for r in rows]
    else:
        corpus = [build_corpus_text(r) for r in rows]
    
    ids = [gen_id(r) for r in rows]

    # 5) 임베딩
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    emb = model.encode(corpus, batch_size=32, convert_to_numpy=True, show_progress_bar=False)

    # 6) 인덱스 로드/업서트/저장
    d = emb.shape[1]
    index, meta = load_or_create_index(d)
    print(f"[INFO] 기존 인덱스: {index.ntotal} 벡터, {len(meta)} 메타데이터")
    
    n_new = upsert(index, meta, ids, emb, rows)
    save_index(index, meta)
    print(f"[INFO] 최종 인덱스: ntotal={index.ntotal}, meta_records={len(meta)}")
    print(f"[INFO] Index saved to: {os.getenv('faiss_index_path')}")
    print(f"[INFO] Meta  saved to: {os.getenv('faiss_meta_path')}")

def query_index(query: str) -> None:
    # 1) 모델/인덱스/메타 로드
    if not (os.path.exists(os.getenv("faiss_index_path")) and os.path.exists(os.getenv("faiss_meta_path"))):
        raise FileNotFoundError("인덱스/메타 파일이 없습니다. 먼저 build를 수행하세요.")
    model_name = os.getenv("embedding_model")
    try:
        topk = int(os.getenv("topk", "5"))
    except Exception:
        topk = 5
    model = SentenceTransformer(model_name)
    index = faiss.read_index(os.getenv("faiss_index_path"))
    with open(os.getenv("faiss_meta_path"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2) 쿼리 임베딩 + 검색
    qvec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    results = search(index, meta, qvec, topk=topk)

    # 3) 출력
    print(f"[SEARCH] query='{query}' topk={topk}")
    for r in results:
        print(f"- ({r['_rank']}) {r.get('title','')}")
        print(f"  link: {r.get('link','')}")
        print(f"  pub : {r.get('published','')}  score(cos)={r.get('_score_ip',0):.4f}")
        print(f"  src : {r.get('source','')}")
        
        # 청킹된 경우 청크 정보 표시
        if "_chunk_id" in r:
            print(f"  chunk: {r.get('_chunk_id','')} ({r.get('_chunk_index',0)+1}/{r.get('_total_chunks',1)})")
            chunk_text = r.get("_chunk_text", "")
            print(f"  chunk_text: {chunk_text[:150]}{'...' if len(chunk_text)>150 else ''}")
        else:
            s = r.get("summary","")
            print(f"  sum : {s[:180]}{'...' if len(s)>180 else ''}")

# -------- CLI --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector DB builder/search for AI news")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build or update FAISS index from JSONL/CSV")
    p_build.add_argument("--config", default="config/config-example.yaml", help="YAML/JSON 설정 파일 경로")

    p_search = sub.add_parser("search", help="Search from FAISS index")
    p_search.add_argument("--query", required=True, help="Search query text")
    p_search.add_argument("--config", help="YAML/JSON 설정 파일 경로")

    return p.parse_args()

def main():
    args = parse_args()
    # 설정 파일 적용
    cfg = getattr(args, 'config', None)
    if cfg and load_and_apply is not None:
        try:
            load_and_apply(cfg)
            print(f"[INFO] 설정 파일 적용: {cfg}")
        except Exception as e:
            print(f"[WARN] 설정 파일 로드 실패: {e}")

    if args.cmd == "build":
        build_index()
    elif args.cmd == "search":
        query_index(args.query)
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
