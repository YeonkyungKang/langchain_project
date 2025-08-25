#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI News Agent - Main Pipeline
뉴스 수집부터 벡터 인덱싱까지 전체 파이프라인을 관리하는 메인 스크립트

사용 예시:
    # 전체 파이프라인 실행 (뉴스 수집 + 벡터 인덱싱)
    python main.py pipeline
    
    # 뉴스 수집만 실행
    python main.py fetch
    
    # 벡터 인덱싱만 실행
    python main.py build
    
    # 검색 실행
    python main.py search --query "AI 정책"
    
    # 청킹 적용한 전체 파이프라인
    python main.py pipeline --chunk-size 512 --chunk-overlap 50
"""

import os
import sys
import argparse
from config_loader import load_and_apply
import subprocess
from typing import Optional, List
from pathlib import Path

# .env 로드
from dotenv import load_dotenv
load_dotenv()


class NewsAgentPipeline:
    """AI 뉴스 에이전트 파이프라인 관리 클래스"""
    
    def __init__(self, news_jsonl, news_csv):
        self.news_jsonl = news_jsonl
        self.news_csv = news_csv
    
    def run_fetch_news(self) -> bool:
        """뉴스 수집 실행"""
        print("=" * 60)
        print("📰 뉴스 수집 시작")
        print("=" * 60)
        
        try:
            # fetch_news.py 실행
            result = subprocess.run([
                sys.executable, "fetch_news.py"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print("✅ 뉴스 수집 완료")
                print(result.stdout)
                return True
            else:
                print("❌ 뉴스 수집 실패")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 뉴스 수집 중 오류 발생: {e}")
            return False
    
    def run_build_index(self, 
                       config_path: Optional[str] = None) -> bool:
        """벡터 인덱스 구축 실행"""
        print("=" * 60)
        print("🔍 벡터 인덱스 구축 시작")
        print("=" * 60)
        
        # vector_store.py 실행 명령어 구성
        cmd = [sys.executable, "vector_store.py", "build"]
        if config_path:
            cmd.extend(["--config", config_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print("✅ 벡터 인덱스 구축 완료")
                print(result.stdout)
                return True
            else:
                print("❌ 벡터 인덱스 구축 실패")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 벡터 인덱스 구축 중 오류 발생: {e}")
            return False
    
    def run_search(self, query: str, config_path: Optional[str] = None) -> bool:
        """뉴스 검색 실행"""
        print("=" * 60)
        print(f"🔎 뉴스 검색: '{query}'")
        print("=" * 60)
        
        cmd = [sys.executable, "vector_store.py", "search", "--query", query]
        if config_path:
            cmd.extend(["--config", config_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print("❌ 검색 실패")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {e}")
            return False

    def run_rag_query(self, query: str, top_k: int = 10, llm_model: str = "gpt-4o-mini", config_path: Optional[str] = None) -> bool:
        """RAG 체인을 사용하여 질의응답을 실행합니다."""
        print("=" * 60)
        print(f"🤖 RAG 질의응답: '{query}'")
        print("=" * 60)
        
        try:
            from rag_chain import RAGChain
            
            print(f"🔄 RAG 체인 초기화 중...")
            rag = RAGChain(llm_model=llm_model, config_path=config_path)
            
            print(f"🔍 질의 처리 중...")
            result = rag.query(query, top_k=top_k)
            
            if result.get("metadata", {}).get("error"):
                print(f"❌ RAG 질의 실패: {result['metadata']['error']}")
                return False
            
            print("✅ RAG 질의 완료")
            print(f"\n📝 답변:\n{result['answer']}")
            print(f"\n📚 참고 문서 ({len(result['sources'])}개):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source['title']} | {source['source']} | {source['date']}")
            
            return True
            
        except Exception as e:
            print(f"❌ RAG 질의 중 오류 발생: {e}")
            return False
    
    def run_pipeline(self,
                    config_path: Optional[str] = None,
                    skip_fetch: bool = False) -> bool:
        """전체 파이프라인 실행 (뉴스 수집 + 벡터 인덱싱)"""
        print("🚀 AI 뉴스 에이전트 파이프라인 시작")
        
        # 1. 뉴스 수집 (skip_fetch가 False인 경우)
        if not skip_fetch:
            if not self.run_fetch_news():
                print("❌ 뉴스 수집 실패로 파이프라인 중단")
                return False
        
        # 2. 벡터 인덱스 구축
        if not self.run_build_index(config_path=config_path):
            print("❌ 벡터 인덱스 구축 실패로 파이프라인 중단")
            return False
        
        print("=" * 60)
        print("🎉 파이프라인 완료!")
        print("=" * 60)
        return True
    
    def check_status(self, faiss_index: str, faiss_meta: str) -> None:
        """현재 상태 확인"""
        print("=" * 60)
        print("📊 현재 상태")
        print("=" * 60)
        
        # 뉴스 파일 확인
        print(f"\n📰 뉴스 파일:")
        print(f"   JSONL: {self.news_jsonl} {'✅' if self.news_jsonl.exists() else '❌'}")
        print(f"   CSV: {self.news_csv} {'✅' if self.news_csv.exists() else '❌'}")
        
        print(f"\n🔍 벡터 인덱스:")
        print(f"   FAISS 인덱스: {faiss_index} {'✅' if faiss_index.exists() else '❌'}")
        print(f"   메타데이터: {faiss_meta} {'✅' if faiss_meta.exists() else '❌'}")
        
        if faiss_meta.exists():
            import json
            try:
                with open(faiss_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                print(f"   인덱싱된 레코드 수: {len(meta)}")
            except Exception as e:
                print(f"   메타데이터 읽기 오류: {e}")

def parse_args() -> argparse.Namespace:
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="AI News Agent - 뉴스 수집부터 벡터 검색까지 전체 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행
  python main.py pipeline

  
  # 뉴스 수집만
  python main.py fetch
  
  # 벡터 인덱싱만
  python main.py build
  
  # 검색
  python main.py search --query "AI 정책"
  
  # RAG 질의응답
  python main.py rag --query "AI 규제 정책의 최신 동향은?"
  
  # 상태 확인
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="실행할 명령")
    
    # pipeline 명령
    pipeline_parser = subparsers.add_parser("pipeline", help="전체 파이프라인 실행 (뉴스 수집 + 벡터 인덱싱)")
    pipeline_parser.add_argument("--config", default="config/config-e5.yaml", help="YAML/JSON 설정 파일 경로")
    # 파이프라인 파라미터는 설정 파일로 관리
    pipeline_parser.add_argument("--skip-fetch", action="store_true", help="뉴스 수집 단계 건너뛰기")

    
    # fetch 명령
    fetch_parser = subparsers.add_parser("fetch", help="뉴스 수집만 실행")
    fetch_parser.add_argument("--config", help="YAML/JSON 설정 파일 경로")
    
    # build 명령
    build_parser = subparsers.add_parser("build", help="벡터 인덱스 구축만 실행")
    build_parser.add_argument("--config", help="YAML/JSON 설정 파일 경로")
    # 세부 옵션은 설정 파일 사용
    
    # search 명령
    search_parser = subparsers.add_parser("search", help="뉴스 검색")
    search_parser.add_argument("--config", help="YAML/JSON 설정 파일 경로")
    search_parser.add_argument("--query", required=True, help="검색 쿼리")
    # topk, model 등은 설정 파일 사용
    
    # rag 명령
    rag_parser = subparsers.add_parser("rag", help="RAG 질의응답")
    rag_parser.add_argument("--config", help="YAML/JSON 설정 파일 경로")
    rag_parser.add_argument("--query", required=True, help="질의")
    rag_parser.add_argument("--top-k", type=int, default=10, help="검색할 문서 수 (기본값: 10)")
    rag_parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM 모델명 (기본값: gpt-4o-mini)")
    
    # status 명령
    status_parser = subparsers.add_parser("status", help="현재 상태 확인")
    status_parser.add_argument("--config", help="YAML/JSON 설정 파일 경로")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    # Apply config early so env vars are ready
    if getattr(args, 'config', None):
        try:
            cfg = load_and_apply(args.config)
            print(f"🧩 설정 적용: {args.config}")
            # 핵심 변수 환경에 명시적으로 반영
            mappings = {
                'embedding_model': 'embedding_model', 
                'llm_model': 'llm_model',
                'faiss_index_path': 'faiss_index_path',
                'faiss_meta_path': 'faiss_meta_path',
                'NEWS_INPUT_PATH': 'NEWS_INPUT_PATH',
            }
            for k, envk in mappings.items():
                if isinstance(cfg, dict) and k in cfg and cfg[k] is not None:
                    os.environ[envk] = str(cfg[k])
            # 디버그 출력
            print("🔧 ENV 설정:")
            print(f"  EMBED_MODEL       = {os.getenv('EMBED_MODEL')}")
            print(f"  LLM_MODEL         = {os.getenv('LLM_MODEL')}")
            print(f"  FAISS_INDEX_PATH  = {os.getenv('FAISS_INDEX_PATH')}")
            print(f"  FAISS_META_PATH   = {os.getenv('FAISS_META_PATH')}")
            print(f"  NEWS_INPUT_PATH   = {os.getenv('NEWS_INPUT_PATH')}")
        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패: {e}")
    
    # 파이프라인 인스턴스 생성
    pipeline = NewsAgentPipeline(news_jsonl=os.getenv('NEWS_INPUT_PATH'), news_csv=os.getenv('NEWS_CSV_PATH'))
    
    try:
        if args.command == "pipeline":
            success = pipeline.run_pipeline(
                config_path=getattr(args, 'config', None),
                skip_fetch=getattr(args, 'skip_fetch', False)
            )
            return 0 if success else 1
            
        elif args.command == "fetch":
            success = pipeline.run_fetch_news()
            return 0 if success else 1
            
        elif args.command == "build":
            success = pipeline.run_build_index(
                config_path=getattr(args, 'config', None)
            )
            return 0 if success else 1
            
        elif args.command == "search":
            success = pipeline.run_search(
                query=args.query,
                config_path=getattr(args, 'config', None)
            )
            return 0 if success else 1
            
        elif args.command == "rag":
            success = pipeline.run_rag_query(
                query=args.query,
                top_k=args.top_k,
                llm_model=args.llm_model,
                config_path=getattr(args, 'config', None)
            )
            return 0 if success else 1
            
        elif args.command == "status":
            pipeline.check_status()
            return 0
            
        else:
            print(f"❌ 알 수 없는 명령: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())


"""
전체 파이프라인: python main.py pipeline --config config/config-example.yaml
인덱스 빌드: python main.py build --config config/config-example.yaml
검색: python main.py search --query "AI 정책" --config config/config-example.yaml
RAG: python main.py rag --query "AI 규제 동향" --config config/config-example.yaml
요약:
main이 fetch_news.py, vector_store.py, rag_chain.py의 최신 CLI/설정 방식과 일치하도록 정리 완료.
설정값은 --config로 통일해 전달되며, 각 스크립트에서 env로 읽어 반영됩니다.
"""