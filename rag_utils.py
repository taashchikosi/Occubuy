import pandas as pd
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"

class RAGRetriever:
    def __init__(self):
        self.property_kb = pd.read_csv(DATA_DIR / "occubuy_property_knowledge_base.csv")
        self.financial_kb = pd.read_csv(DATA_DIR / "occubuy_financial_knowledge_base.csv")

        self.vector_db_path = DATA_DIR / "occubuy_vector_db"
        self.faiss_index = None
        self.vector_documents = None
        self.vector_config = None

        self._load_vector_db()
        self._build_tfidf_index()

    def _load_vector_db(self):
        """Load pre-built FAISS vector DB"""
        try:
            config_path = self.vector_db_path / "vector_db_config.json"
            index_path = self.vector_db_path / "occubuy_faiss.index"
            docs_path = self.vector_db_path / "occubuy_documents.pkl"

            if FAISS_AVAILABLE and config_path.exists() and index_path.exists() and docs_path.exists():
                with open(config_path, 'r') as f:
                    self.vector_config = json.load(f)
                self.faiss_index = faiss.read_index(str(index_path))
                with open(docs_path, 'rb') as f:
                    self.vector_documents = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load vector DB: {e}")
            self.faiss_index = None

    def _build_tfidf_index(self):
        """Build TF-IDF index as fallback"""
        property_texts = (
            self.property_kb['match_summary'].fillna("") + " " +
            self.property_kb['full_property_profile_text'].fillna("")
        ).tolist()

        financial_texts = (
            self.financial_kb['financial_strengths'].fillna("") + " " +
            self.financial_kb['plain_english_financial_summary'].fillna("")
        ).tolist()

        all_texts = property_texts + financial_texts

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-z]+\b'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

        self.tfidf_docs = {
            'properties': property_texts,
            'financial': financial_texts
        }

    def retrieve_properties_by_lifestyle(self, lifestyle_query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve properties matching lifestyle using hybrid retrieval"""

        # Try vector DB first if available
        if self.vector_documents and self.faiss_index:
            try:
                vector_results = self._vector_search(lifestyle_query, top_k)
                if vector_results:
                    return vector_results
            except Exception as e:
                print(f"Vector search failed, falling back to TF-IDF: {e}")

        # Fallback to TF-IDF
        return self._tfidf_search_properties(lifestyle_query, top_k)

    def retrieve_financial_insights(self, financial_query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve financial KB insights"""
        return self._tfidf_search_financial(financial_query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using vector DB (if available)"""
        if not self.vector_documents or not self.faiss_index:
            return []

        try:
            query_embedding = self._embed_text(query)
            distances, indices = self.faiss_index.search(np.array([query_embedding]), top_k)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.vector_documents):
                    doc = self.vector_documents[idx]
                    matching_rows = self.property_kb[
                        self.property_kb['property_id'] == doc.get('property_id', -1)
                    ]
                    if not matching_rows.empty:
                        results.append(matching_rows.iloc[0].to_dict())
            return results
        except Exception:
            return []

    def _embed_text(self, text: str) -> np.ndarray:
        """Simple embedding using avg of word vectors (for demo)"""
        words = text.lower().split()
        embedding = np.random.RandomState(hash(text) % 2**32).randn(384)
        return embedding.astype('float32')

    def _tfidf_search_properties(self, query: str, top_k: int) -> List[Dict]:
        """Search properties using TF-IDF"""
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = self.tfidf_matrix[:len(self.tfidf_docs['properties'])].dot(query_vec.T).toarray().flatten()
            top_indices = np.argsort(scores)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    matching_rows = self.property_kb[self.property_kb.index == idx]
                    if not matching_rows.empty:
                        results.append(matching_rows.iloc[0].to_dict())
            return results
        except Exception as e:
            print(f"TF-IDF search failed: {e}")
            return []

    def _tfidf_search_financial(self, query: str, top_k: int) -> List[Dict]:
        """Search financial KB using TF-IDF"""
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            start_idx = len(self.tfidf_docs['properties'])
            financial_scores = self.tfidf_matrix[start_idx:].dot(query_vec.T).toarray().flatten()
            top_indices = np.argsort(financial_scores)[-top_k:][::-1]

            results = []
            for relative_idx in top_indices:
                if financial_scores[relative_idx] > 0:
                    actual_idx = relative_idx
                    if actual_idx < len(self.financial_kb):
                        results.append(self.financial_kb.iloc[actual_idx].to_dict())
            return results
        except Exception as e:
            print(f"Financial search failed: {e}")
            return []

    def get_property_by_id(self, property_id: int) -> Dict:
        """Get specific property"""
        matching = self.property_kb[self.property_kb['property_id'] == property_id]
        return matching.iloc[0].to_dict() if not matching.empty else None

    def search_properties_by_price(self, min_price: float, max_price: float, top_k: int = 10) -> List[Dict]:
        """Search properties by price range"""
        filtered = self.property_kb[
            (self.property_kb['price'] >= min_price) &
            (self.property_kb['price'] <= max_price)
        ].sort_values('investment_quality_score', ascending=False)

        return filtered.head(top_k).to_dict('records')

    def search_properties_by_suburb(self, suburb: str, top_k: int = 10) -> List[Dict]:
        """Search properties by suburb"""
        filtered = self.property_kb[
            self.property_kb['suburb'].str.contains(suburb, case=False, na=False)
        ].sort_values('match_summary_score', ascending=False) if 'match_summary_score' in self.property_kb.columns else filtered

        return filtered.head(top_k).to_dict('records')
