"""
streamlit run app.py --server.port 30674 --server.address 0.0.0.0
"""

import os
import json
import time
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from config_loader import load_and_apply

# --- LangChain & Vector Store ---
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document

# --- BM25 Hybrid ---
from rank_bm25 import BM25Okapi

# ---------- 환경 설정 ----------
load_dotenv()
st.set_page_config(
    page_title="AI Policy/Tech/Issues QA",
    page_icon="📰",
    layout="wide"
)

# ---------- 유틸 ----------
@st.cache_resource
def load_faiss_index(
    index_dir: str = "./data",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[FAISS, Any]:
    """
    FAISS 인덱스를 로드합니다. (vector_store.py 형식)
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.docstore.in_memory import InMemoryDocstore
    
    # Sentence Transformers 임베딩 사용
    # 모델명은 환경변수 우선 적용 (config에서 주입)
    model_name_env = os.getenv("EMBED_MODEL") or os.getenv("embedding_model") or embedding_model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_env,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # vector_store.py에서 생성한 FAISS 파일 로드
    import faiss
    import json
    
    # 경로는 환경변수 우선 적용 (config에서 주입)
    index_file = (
        os.getenv("faiss_index_path")
        or os.getenv("FAISS_INDEX_PATH")
        or os.path.join(index_dir, "news_faiss_cos.index")
    )
    
    meta_file = (
        os.getenv("faiss_meta_path")
        or os.getenv("FAISS_META_PATH")
        or os.path.join(index_dir, "news_faiss_meta.json")
    )
    
    if os.path.exists(index_file) and os.path.exists(meta_file):
        # FAISS 인덱스 로드
        index = faiss.read_index(index_file)
        # 차원 일치 여부 사전 점검
        try:
            probe_vec = embeddings.embed_query("probe")
            if hasattr(index, 'd') and len(probe_vec) != index.d:
                raise RuntimeError(
                    f"임베딩 차원 불일치: index.d={getattr(index, 'd', 'unknown')} vs embed_dim={len(probe_vec)}. "
                    f"EMBED_MODEL='{model_name_env}'이 인덱스를 만든 모델과 다를 수 있습니다."
                )
        except Exception as _e:
            # 차원 체크 중 오류도 사용자에게 알림
            raise
        
        # 메타데이터 로드
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 메타데이터를 Document 객체로 변환
        documents = []
        doc_ids = list(meta.keys())
        
        for i, doc_id in enumerate(doc_ids):
            doc_meta = meta[doc_id]
            doc = Document(
                page_content=doc_meta.get('summary', ''),
                metadata={
                    'id': doc_id,
                    'title': doc_meta.get('title', ''),
                    'url': doc_meta.get('link', ''),
                    'source': doc_meta.get('source', ''),
                    'date': doc_meta.get('published', ''),
                    'tags': doc_meta.get('tags', '').split(',') if doc_meta.get('tags') else []
                }
            )
            documents.append(doc)
        
        # InMemoryDocstore 생성 (순차적 인덱스를 키로 사용)
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        
        # FAISS 인덱스 크기와 메타데이터 크기 확인 및 조정
        print(f"FAISS 인덱스 크기: {index.ntotal}")
        print(f"메타데이터 크기: {len(documents)}")
        
        # FAISS 인덱스 크기에 맞춰 docstore와 매핑 조정
        if index.ntotal > len(documents):
            # FAISS 인덱스가 더 큰 경우, 부족한 부분을 빈 문서로 채움
            for i in range(len(documents), index.ntotal):
                empty_doc = Document(
                    page_content="",
                    metadata={'id': f'empty_{i}', 'title': '', 'url': '', 'source': '', 'date': '', 'tags': []}
                )
                documents.append(empty_doc)
                docstore._dict[str(i)] = empty_doc
        
        # index_to_docstore_id 매핑 생성
        index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}
        
        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        return vs, embeddings
    else:
        raise FileNotFoundError(f"FAISS 파일을 찾을 수 없습니다: {index_file}, {meta_file}")


@st.cache_resource
def load_bm25_corpus(corpus_path: str = "./data/corpus.jsonl") -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
    """
    BM25용 말뭉치를 로드합니다. 
    corpus.jsonl 형식 예시:
    {"id": "doc_001", "title": "...", "text": "...", "url": "...", "source": "...", "date": "2025-08-25", "tags": ["policy","regulation"]}
    """
    # 환경변수/설정 적용
    corpus_path_env = os.getenv("CORPUS_PATH") or os.getenv("corpus_path") or corpus_path
    if not os.path.exists(corpus_path_env):
        return None, []

    corpus = []
    with open(corpus_path_env, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj)

    tokenized = [ (c.get("title","") + " " + c.get("text","")).split() for c in corpus ]
    bm25 = BM25Okapi(tokenized)
    return bm25, corpus


def reciprocal_rank_fusion(results: List[List[Tuple[str, float]]], k: float = 60.0, top_k: int = 20):
    """
    RRF: 여러 랭킹 리스트를 doc_id 기준으로 결합.
    results: [[(doc_id, score), ...], [(doc_id, score), ...], ...]
    반환: [(doc_id, fused_score), ...] 점수 내림차순
    """
    rrfs = {}
    for ranking in results:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrfs[doc_id] = rrfs.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(rrfs.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def filter_by_metadata(
    docs: List[Document],
    source: Optional[str],
    date_from: Optional[dt.date],
    date_to: Optional[dt.date],
    tags: List[str]
) -> List[Document]:
    out = []
    for d in docs:
        meta = d.metadata or {}
        ok = True
        if source and (meta.get("source") != source):
            ok = False
        if date_from or date_to:
            try:
                ddate = pd.to_datetime(meta.get("date")).date() if meta.get("date") else None
            except Exception:
                ddate = None
            if date_from and (ddate is None or ddate < date_from):
                ok = False
            if date_to and (ddate is None or ddate > date_to):
                ok = False
        if tags:
            mtags = set(meta.get("tags", []))
            if not mtags.issuperset(set(tags)):
                ok = False
        if ok:
            out.append(d)
    return out


def docs_to_dataframe(docs: List[Document]) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        rows.append({
            "rank": i,
            "title": m.get("title") or d.page_content[:80],
            "source": m.get("source"),
            "date": m.get("date"),
            "url": m.get("url"),
            "tags": ", ".join(m.get("tags", [])),
            "content": d.page_content
        })
    return pd.DataFrame(rows)


def get_sources_and_tags(docs: List[Document]) -> Tuple[List[str], List[str]]:
    src_set, tag_set = set(), set()
    for d in docs:
        m = d.metadata or {}
        if m.get("source"):
            src_set.add(m["source"])
        for t in m.get("tags", []):
            tag_set.add(t)
    return sorted(src_set), sorted(tag_set)


def vector_search(vs: FAISS, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
    """
    Vector 검색 결과를 (doc_id, score) 리스트로 리턴.
    LangChain-FAISS는 내부적으로 id를 유지. score는 유사도 역수 가정으로 변환.
    """
    docs = vs.similarity_search_with_score(query, k=top_k)
    out = []
    for d, dist in docs:
        # dist는 L2 거리일 수 있음 → 점수화(작을수록 유사)를 위해 -dist 사용
        doc_id = d.metadata.get("id") or d.metadata.get("doc_id") or d.metadata.get("url") or d.page_content[:30]
        out.append((str(doc_id), -float(dist)))
    return out


def bm25_search(bm25: BM25Okapi, corpus: List[Dict[str, Any]], query: str, top_k: int = 15) -> List[Tuple[str, float]]:
    if bm25 is None:
        return []
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    idxs = np.argsort(scores)[::-1][:top_k]
    out = []
    for i in idxs:
        doc = corpus[int(i)]
        doc_id = doc.get("id") or doc.get("url") or doc.get("title", "")[:30] + f"_{i}"
        out.append((str(doc_id), float(scores[i])))
    return out


def gather_docs_by_ids(
    ids: List[str],
    vs: FAISS,
    corpus: List[Dict[str, Any]]
) -> List[Document]:
    """
    RRF로 뽑힌 doc_id에 해당하는 실제 Document 객체를 수집.
    - 우선순위: FAISS 메타데이터 id 매칭 → corpus id 매칭 → url 매칭
    """
    # 만들어진 인덱스에서 전체 docs를 직접 꺼내기 어렵기 때문에,
    # 간단히 FAISS 결과 재검색 + corpus 매핑으로 복원
    found = []
    # 1) Vector store에서 유사 매칭
    for _id in ids:
        docs = vs.similarity_search(_id, k=1)  # id를 질의로 근사 검색
        if docs:
            found.append(docs[0])

    # 2) corpus에서 누락 보완
    idset = set([d.metadata.get("id") or d.metadata.get("url") for d in found])
    for _id in ids:
        if _id in idset:
            continue
        for c in corpus:
            cid = c.get("id") or c.get("url")
            if str(cid) == str(_id):
                found.append(Document(
                    page_content=c.get("text",""),
                    metadata={
                        "id": cid,
                        "title": c.get("title"),
                        "url": c.get("url"),
                        "source": c.get("source"),
                        "date": c.get("date"),
                        "tags": c.get("tags", [])
                    }
                ))
                break
    # 중복 제거
    uniq = []
    seen = set()
    for d in found:
        key = (d.metadata.get("id"), d.metadata.get("url"), d.metadata.get("title"))
        if key not in seen:
            uniq.append(d)
            seen.add(key)
    return uniq


def build_llm(provider: str, model_name: str, temperature: float = 0.2):
    """
    LLM Provider 구성: OpenAI를 기본으로 하되, 필요 시 교체 지점.
    """
    if provider.lower() == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
    # 필요 시 Upstage, Vertex 등 추가 분기
    raise ValueError("지원하지 않는 provider 입니다. (OpenAI 사용 권장)")


def synthesize_answer(llm, question: str, passages: List[Document], language: str = "ko") -> str:
    """
    RAG 생성기: 상위 패시지로 답변 생성.
    - 한국어(Korean)로 답하되 **중요 키워드(English)** 병기.
    - 출처는 하단 '주요 기사' 카드로 별도 표기.
    """
    context = []
    for i, d in enumerate(passages[:8], 1):
        m = d.metadata or {}
        title = m.get("title") or f"Doc{i}"
        src = m.get("source") or ""
        date = m.get("date") or ""
        url = m.get("url") or ""
        context.append(f"[{i}] {title} | {src} | {date} | {url}\n{d.page_content}")

    system_prompt = (
        "너는 AI 정책/기술/이슈 전문 분석가다. 사용자의 질의에 대해 신뢰할 수 있는 근거를 바탕으로 "
        "간결하고 정확하게 한국어로 답하되, 중요한 기술 키워드는 'Korean (English)' 표기법으로 작성한다. "
        "절대 근거 없는 추정은 하지 말고, 모르면 모른다고 말한다. "
        "출처 링크는 본문에 직접 나열하지 말고, 사용자가 볼 수 있도록 UI 하단의 기사 카드 영역에 맡긴다."
    )

    user_prompt = f"""[질의]
{question}

[검색 문맥]
{chr(10).join(context)}

[지시]
- 핵심만 bullet로.
- 정책/규제(Policy/Regulation), 기술(Technology), 이슈(Issue)로 구분.
- 가능한 경우 날짜(YYYY-MM-DD) 명시.
- 모호하면 가정하지 말고 '추가 자료 필요'로 구분.
"""

    resp = llm.invoke([{"role":"system","content":system_prompt},
                       {"role":"user","content":user_prompt}])
    return resp.content


# ---------- 사이드바 ----------
with st.sidebar:
    st.header("⚙️ 설정 (Settings)")
    provider = st.selectbox("LLM Provider", ["OpenAI"], index=0)
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k_vec = st.slider("Vector top_k", 5, 50, 15, 1)
    top_k_bm25 = st.slider("BM25 top_k", 0, 50, 15, 1)
    rrf_k = st.slider("RRF k (합산 안정화)", 10, 120, 60, 5)
    max_show = st.slider("표시할 기사 수", 5, 30, 10, 1)
    st.caption("BM25 top_k=0 으로 두면 순수 Vector 검색만 사용합니다.")

# ---------- 헤더 ----------
st.title("📰 실시간 AI 정책/기술/이슈 QA 대시보드")
st.markdown("- 상단: **질의응답(QA)** · 하단: **주요 기사(출처/메타데이터)** 카드\n- 검색: **Hybrid(BM25 + Vector) + RRF**\n- 결과는 한국어로 제공하며, **중요 키워드(Korean · English)** 병기")

# ---------- 설정 파일 적용 (옵션) ----------
config_path = 'config/config-example.yaml'
if config_path:
    try:
        load_and_apply(config_path)
        st.info(f"설정 적용: {config_path}")
    except Exception as e:
        st.warning(f"설정 파일 로드 실패: {e}")

# ---------- 데이터 로드 ----------
with st.spinner("FAISS 인덱스 로드 중..."):
    vs, embeddings = load_faiss_index("./data")
bm25, corpus = load_bm25_corpus("./data/corpus.jsonl")

# 인덱스에서 샘플 문서 취득 (메타데이터 통계용)
sample_docs = vs.similarity_search("AI policy", k=200)
sources_all, tags_all = get_sources_and_tags(sample_docs)

# ---------- 필터 ----------
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.2])
with col1:
    source_filter = st.selectbox("Source 필터", ["전체"] + sources_all, index=0)
    if source_filter == "전체":
        source_filter = None
with col2:
    date_from = st.date_input("시작일", value=None)
with col3:
    date_to = st.date_input("종료일", value=None)
with col4:
    selected_tags = st.multiselect("Tags 필터", options=tags_all, default=[])

# ---------- 질의 입력 ----------
example_queries = [
    "K-정부의 생성형 AI 규제 동향과 해외 비교",
    "AI 윤리와 안전성 관련 최신 정책 동향",
    "생성형 AI 기술의 발전 현황과 주요 이슈",
    "AI 반도체와 NPU 기술 동향",
    "AI 거버넌스와 샌드박스 정책 현황",
    "온디바이스 AI와 엣지 컴퓨팅 기술",
    "AI 데이터 거버넌스와 프라이버시 보호",
    "메가모델과 초거대 AI 모델 동향",
    "AI 검증과 평가 체계 현황",
    "주권 AI와 국가 AI 전략"
]

col1, col2 = st.columns([3, 1])
with col1:
    selected_example = st.selectbox(
        "📝 예시 질의 선택 (선택사항)",
        ["직접 입력"] + example_queries,
        index=0
    )
with col2:
    if st.button("예시 적용", help="선택한 예시를 입력창에 적용"):
        if selected_example != "직접 입력":
            st.session_state["query_text"] = selected_example

query = st.text_input(
    "질의 입력",
    value=st.session_state.get("query_text", ""),
    placeholder="예: K-정부의 생성형 AI 규제 동향과 해외 비교"
)

do_search = st.button("🔎 검색 및 생성 (Search & Generate)", type="primary")

# ---------- 실행 ----------
if do_search and query.strip():
    with st.spinner("검색 및 RAG 생성 중..."):
        # 1) Vector 검색
        vec_rank = vector_search(vs, query, top_k=top_k_vec)

        # 2) BM25 검색
        bm25_rank = bm25_search(bm25, corpus, query, top_k=top_k_bm25) if top_k_bm25 > 0 else []

        # 3) RRF 결합
        fused = reciprocal_rank_fusion([vec_rank, bm25_rank], k=float(rrf_k), top_k=max_show*2)
        fused_ids = [doc_id for doc_id, _ in fused]

        # 4) 문서 복원 + 필터
        docs_all = gather_docs_by_ids(fused_ids, vs, corpus)
        docs_filtered = filter_by_metadata(docs_all, source_filter, date_from, date_to, selected_tags)

        # 5) 상위 문서로 답변 생성
        top_docs = docs_filtered[:max_show] if docs_filtered else docs_all[:max_show]
        if len(top_docs) == 0:
            st.warning("검색 결과가 없습니다. 쿼리나 필터를 조정해 보세요.")
        else:
            llm = build_llm(provider, model_name, temperature)
            answer = synthesize_answer(llm, query, top_docs, language="ko")

            # --- 상단 QA 출력 ---
            st.subheader("🧠 질의응답(Answer)")
            st.markdown(answer)

            # 피드백
            fb_col1, fb_col2, fb_col3 = st.columns([0.5, 0.5, 6])
            with fb_col1:
                if st.button("👍 도움 됨"):
                    st.toast("피드백 감사합니다!", icon="✅")
            with fb_col2:
                if st.button("👎 도움이 안 됨"):
                    st.toast("개선을 위해 의견 반영하겠습니다.", icon="⚠️")

            # --- 하단 기사 카드 ---
            st.subheader("🗂️ 주요 기사 (출처/메타데이터 전부 표시)")
            df = docs_to_dataframe(top_docs)

            # 카드 렌더링
            for i, row in df.iterrows():
                with st.container(border=True):
                    st.markdown(f"#### #{int(row['rank'])}. {row['title']}")
                    meta_cols = st.columns([1, 1, 2])
                    with meta_cols[0]:
                        st.markdown(f"- **Source**: `{row['source']}`")
                        st.markdown(f"- **Date**: `{row['date']}`")
                    with meta_cols[1]:
                        st.markdown(f"- **Tags**: `{row['tags']}`")
                    with meta_cols[2]:
                        if row["url"]:
                            st.markdown(f"- **URL**: [{row['url']}]({row['url']})")
                        else:
                            st.markdown("- **URL**: (없음)")
                    # 전체 content 노출
                    with st.expander("본문(content) 전체 보기", expanded=False):
                        st.write(row["content"])

            with st.expander("표 형태로 전체 확인"):
                st.dataframe(df, use_container_width=True)

else:
    st.info("상단 입력창에 질의어를 입력하고 **검색 및 생성** 버튼을 눌러주세요.")

# ---------- 푸터 ----------
st.caption("ⓘ 참고:*본 앱은 사전 구축된 FAISS 인덱스와 BM25 코퍼스를 사용합니다.*")
