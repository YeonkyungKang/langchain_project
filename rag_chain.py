#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG (Retrieval-Augmented Generation) 체인 구현
Retriever + Generator 결합 방식

워크플로우:
1. 사용자 질의 입력
2. Vector Store에서 관련 문서 검색 (Retriever)
3. 검색된 문서들을 컨텍스트로 사용하여 LLM으로 답변 생성 (Generator)
4. 답변과 함께 출처 문서 반환
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

# Config loader
from config_loader import load_and_apply

# Load environment variables
load_dotenv()

class E5EmbeddingsWrapper:
    """Wraps a base embeddings to add E5-style prefixes for queries and documents."""
    def __init__(self, base_embeddings: HuggingFaceEmbeddings):
        self._base = base_embeddings

    def embed_query(self, text: str):
        return self._base.embed_query(f"query: {text}")

    def embed_documents(self, texts: List[str]):
        prefixed = [f"passage: {t}" for t in texts]
        return self._base.embed_documents(prefixed)


class HybridRetriever(BaseRetriever):
    """
    하이브리드 검색기: Vector Search + BM25 결합
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        bm25_corpus: List[Dict[str, Any]],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 10
    ):
        super().__init__()
        self._vector_store = vector_store
        self._bm25_corpus = bm25_corpus
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._top_k = top_k
        
        # BM25 초기화
        if bm25_corpus:
            tokenized = [(c.get("title", "") + " " + c.get("text", "")).split() for c in bm25_corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """질의에 관련된 문서들을 검색합니다."""
        
        # 1. Vector Search
        vector_docs = self._vector_store.similarity_search_with_score(query, k=self._top_k)
        
        # 2. BM25 Search (if available)
        bm25_docs = []
        if self._bm25:
            tokenized_q = query.split()
            scores = self._bm25.get_scores(tokenized_q)
            top_indices = np.argsort(scores)[::-1][:self._top_k]
            
            for idx in top_indices:
                if scores[idx] > 0:
                    doc_data = self._bm25_corpus[int(idx)]
                    doc = Document(
                        page_content=doc_data.get("text", ""),
                        metadata={
                            "id": doc_data.get("id"),
                            "title": doc_data.get("title"),
                            "url": doc_data.get("url"),
                            "source": doc_data.get("source"),
                            "date": doc_data.get("date"),
                            "tags": doc_data.get("tags", []),
                            "score": float(scores[idx])
                        }
                    )
                    bm25_docs.append((doc, scores[idx]))
        
        # 3. 결과 결합 (간단한 방식)
        all_docs = []
        
        # Vector 검색 결과 추가
        for doc, score in vector_docs:
            doc.metadata["vector_score"] = -float(score)  # 거리를 점수로 변환
            all_docs.append(doc)
        
        # BM25 검색 결과 추가 (중복 제거)
        existing_urls = {doc.metadata.get("url") for doc in all_docs}
        for doc, score in bm25_docs:
            if doc.metadata.get("url") not in existing_urls:
                doc.metadata["bm25_score"] = float(score)
                all_docs.append(doc)
        
        return all_docs[:self._top_k]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 검색 (동기 버전과 동일)"""
        return self._get_relevant_documents(query)


class RAGChain:
    """
    RAG 체인: Retriever + Generator 결합
    """
    
    def __init__(
        self,
        vector_store_path: str = "./data",
        corpus_path: str = "./data/corpus.jsonl",
        embedding_model: str = os.getenv("embedding_model"),
        llm_model: str = os.getenv("llm_model"),
        temperature: float = 0.2,
        max_tokens: int = 1000,
        config_path: Optional[str] = None
    ):
        self.vector_store_path = vector_store_path
        self.corpus_path = corpus_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load config first so env vars are available
        if config_path:
            try:
                load_and_apply(config_path)
                print(f"[INFO] 설정 파일 적용: {config_path}")
            except Exception as e:
                print(f"[WARN] 설정 파일 로드 실패: {e}")

        # Resolve models strictly from environment (populated by config)
        self.embedding_model = os.getenv("embedding_model")
        self.llm_model = os.getenv("llm_model")
        
        # Initialize components
        self._load_components()
        self._setup_chain()

    def _build_embeddings(self):
        """Create embeddings object. Supports E5 instruct models with proper prefixes."""
        model_name = self.embedding_model
        base = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # If using an E5 instruct model, apply query/passsage prefixes
        if "e5" in model_name.lower():
            return E5EmbeddingsWrapper(base)
        return base
    
    def _load_components(self):
        """컴포넌트들을 로드합니다."""
        print("🔄 RAG 체인 컴포넌트 로드 중...")
        
        # 1. Embeddings
        self.embeddings = self._build_embeddings()
        
        # 2. Vector Store (vector_store.py 형식 직접 로드)
        try:
            import faiss
            import json
            
            # Config에서 경로 읽기
            index_file = os.getenv("faiss_index_path")
            meta_file = os.getenv("faiss_meta_path")
            
            if os.path.exists(index_file) and os.path.exists(meta_file):
                # FAISS 인덱스 로드
                index = faiss.read_index(index_file)
                
                # 메타데이터 로드
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                from langchain_community.docstore.in_memory import InMemoryDocstore
                # 메타데이터를 Document 객체로 변환
                documents = []
                for i, (doc_id, doc_meta) in enumerate(meta.items()):
                    doc = Document(
                        page_content=doc_meta.get('text', ''),
                        metadata={
                            'id': doc_id,
                            'title': doc_meta.get('title', ''),
                            'url': doc_meta.get('url', ''),
                            'source': doc_meta.get('source', ''),
                            'date': doc_meta.get('date', ''),
                            'tags': doc_meta.get('tags', [])
                        }
                    )
                    documents.append(doc)
                
                # InMemoryDocstore 생성
                docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
                
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id={i: str(i) for i in range(index.ntotal)}
                )
                
                print(f"✅ Vector Store 로드 완료 (인덱스 크기: {index.ntotal})")
            else:
                print(f"❌ FAISS 파일을 찾을 수 없습니다:")
                print(f"   - 인덱스 파일: {index_file} {'✅' if os.path.exists(index_file) else '❌'}")
                print(f"   - 메타 파일: {meta_file} {'✅' if os.path.exists(meta_file) else '❌'}")
                self.vector_store = None
        except Exception as e:
            print(f"❌ Vector Store 로드 실패: {e}")
            self.vector_store = None
        
        # 3. BM25 Corpus
        self.bm25_corpus = []
        corpus_path = os.getenv("CORPUS_PATH", self.corpus_path)
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        self.bm25_corpus.append(json.loads(line))
                print(f"✅ BM25 Corpus 로드 완료 ({len(self.bm25_corpus)} 문서)")
            except Exception as e:
                print(f"❌ BM25 Corpus 로드 실패: {e}")
        else:
            print(f"⚠️ BM25 Corpus 파일이 없습니다: {corpus_path}")
        
        # 4. LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 5. Hybrid Retriever
        if self.vector_store:
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_corpus=self.bm25_corpus,
                top_k=15
            )
            print("✅ Hybrid Retriever 초기화 완료")
        else:
            self.retriever = None
            print("❌ Retriever 초기화 실패")
    
    def _setup_chain(self):
        """RAG 체인을 설정합니다."""
        if not self.retriever:
            print("❌ Retriever가 없어 체인을 설정할 수 없습니다.")
            return
        
        # 프롬프트 템플릿
        self.prompt_template = ChatPromptTemplate.from_template("""
당신은 AI 정책/기술/이슈 전문 분석가입니다. 사용자의 질의에 대해 제공된 문서들을 기반으로 정확하고 유용한 답변을 제공하세요.

**지시사항:**
1. 제공된 문서들만을 참고하여 답변하세요.
2. 한국어로 답변하되, 중요한 기술 키워드는 '한국어 (English)' 형태로 표기하세요.
3. 답변은 구조화하여 bullet point로 작성하세요.
4. 정책/규제, 기술, 이슈로 구분하여 답변하세요.
5. 가능한 경우 날짜(YYYY-MM-DD)를 명시하세요.
6. 확실하지 않은 정보는 "추가 자료 필요"로 표시하세요.
7. 출처는 답변 하단에 별도로 나열하세요.

**질의:** {question}

**참고 문서:**
{context}

**답변:**
""")
        
        # RAG 체인 구성
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG 체인 설정 완료")
    
    def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        질의를 처리하고 답변을 생성합니다.
        
        Args:
            question: 사용자 질의
            top_k: 검색할 문서 수
            
        Returns:
            Dict containing:
            - answer: 생성된 답변
            - sources: 참고 문서 목록
            - metadata: 메타데이터
        """
        if not self.retriever:
            return {
                "answer": "❌ Retriever가 초기화되지 않았습니다.",
                "sources": [],
                "metadata": {"error": "Retriever not initialized"}
            }
        
        try:
            print(f"🔍 질의 처리 중: {question}")
            start_time = time.time()
            
            # 1. 문서 검색
            docs = self.retriever._get_relevant_documents(question)
            print(f"📄 {len(docs)}개 문서 검색됨")
            
            # 2. 컨텍스트 구성
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs[:top_k], 1):
                title = doc.metadata.get("title", f"문서 {i}")
                source = doc.metadata.get("source", "알 수 없음")
                date = doc.metadata.get("date", "")
                url = doc.metadata.get("url", "")
                
                context_parts.append(f"[{i}] {title} | {source} | {date}\n{doc.page_content}")
                
                sources.append({
                    "rank": i,
                    "title": title,
                    "source": source,
                    "date": date,
                    "url": url,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            context = "\n\n".join(context_parts)
            
            # 3. 답변 생성
            answer = self.rag_chain.invoke(question)
            
            # 4. 메타데이터 구성
            metadata = {
                "query": question,
                "search_time": time.time() - start_time,
                "documents_retrieved": len(docs),
                "model_used": self.llm_model,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ 답변 생성 완료 ({metadata['search_time']:.2f}초)")
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": metadata
            }
            
        except Exception as e:
            import traceback
            print(f"❌ 질의 처리 중 오류 발생: {e}")
            print(f"상세 오류: {traceback.format_exc()}")
            return {
                "answer": f"❌ 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계를 반환합니다."""
        stats = {
            "vector_store": {
                "loaded": self.vector_store is not None,
                "index_size": self.vector_store.index.ntotal if self.vector_store else 0
            },
            "bm25_corpus": {
                "loaded": len(self.bm25_corpus) > 0,
                "document_count": len(self.bm25_corpus)
            },
            "models": {
                "embedding": self.embedding_model,
                "llm": self.llm_model
            },
            "retriever": {
                "initialized": self.retriever is not None
            }
        }
        return stats


def main():
    """테스트용 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chain for AI News QA")
    parser.add_argument("--query", required=True, help="질의 텍스트")
    parser.add_argument("--config", default="config/config-example.yaml", help="설정 파일 경로")
    parser.add_argument("--top-k", type=int, default=10, help="검색할 문서 수")
    
    args = parser.parse_args()
    
    print("🚀 RAG 체인 시작")
    print(f"질의: {args.query}")
    print(f"설정 파일: {args.config}")
    
    # RAG 체인 초기화
    rag = RAGChain(config_path=args.config)
    
    # 통계 출력
    stats = rag.get_stats()
    print("\n📊 시스템 통계:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 질의 처리
    print(f"\n{'='*60}")
    print(f"질의: {args.query}")
    print(f"{'='*60}")
    
    result = rag.query(args.query, top_k=args.top_k)
    
    print(f"\n답변:\n{result['answer']}")
    print(f"\n참고 문서 수: {len(result['sources'])}")
    if 'search_time' in result['metadata']:
        print(f"처리 시간: {result['metadata']['search_time']:.2f}초")
    else:
        print("처리 시간: 알 수 없음")
    
    # 참고 문서 출력
    if result['sources']:
        print(f"\n📚 참고 문서:")
        for i, source in enumerate(result['sources'][:5], 1):
            print(f"{i}. {source['title']} | {source['source']} | {source['date']}")
            print(f"   {source['url']}")
            print(f"   {source['content'][:100]}...")
            print()


if __name__ == "__main__":
    main()

"""
# 기본 설정으로 질의
python rag_chain.py --query "AI 규제 정책의 최신 동향은 무엇인가요?"

# 특정 설정 파일 사용
python rag_chain.py --query "생성형 AI 기술 발전 현황" --config config/config-e5.yaml

# 검색 문서 수 조정
python rag_chain.py --query "AI 윤리 이슈" --config config/config-example.yaml --top-k 15

"""