import os
import json
import re
import enum
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import faiss
import scipy.special

# ONNX Optimum libraries for fast inference
from transformers import AutoTokenizer
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification
)


# ==========================================
# 1. DATA MODELS & UTILS
# ==========================================
class IngestionStatus(enum.Enum):
    APPROVED = "APPROVED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    DUPLICATE_REJECTED = "DUPLICATE_REJECTED"

@dataclass
class MatchCandidate:
    question_id: int
    question_text: str
    bi_encoder_score: float
    cross_encoder_score: Optional[float] = None

@dataclass
class IngestionResult:
    status: IngestionStatus
    question_id: Optional[int]
    question_text: str
    top_matches: List[MatchCandidate] = field(default_factory=list)

class MathTextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        text = str(text).lower()
        operators = [r"\+", r"-", r"–", r"=", r"\*", r"/", r"÷"]
        for op in operators:
            text = re.sub(f"({op})", r" \1 ", text)
        return re.sub(r"\s+", " ", text).strip()




# ==========================================
# 2. ONNX INFERENCE ENGINES
# ==========================================
class OnnxBiEncoder:
    """Uses optimized ONNX INT8 model for fast CPU embeddings."""
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_path)

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> np.ndarray:
        norm_texts = [MathTextNormalizer.normalize(t) for t in texts]
        inputs = self.tokenizer(norm_texts, padding=True, truncation=True, max_length=128, return_tensors="np", fix_mistral_regex=True)
        
        outputs = self.model(**inputs)
        embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        
        # L2 Normalize for FAISS Inner Product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        return norm_embeddings.astype(np.float32)


class OnnxCrossEncoder:
    """Uses optimized ONNX INT8 model for fast duplicate verification."""
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(model_path)

    def predict_duplicate_probs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)
            
        # FIX: Explicitly separate the pairs into two lists
        texts = [p[0] for p in pairs]
        text_pairs = [p[1] for p in pairs]
        
        # Pass them explicitly as text and text_pair
        inputs = self.tokenizer(
            text=texts, 
            text_pair=text_pairs, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="np",
            fix_mistral_regex=True
        )
        outputs = self.model(**inputs)
        raw_logits = outputs.logits
        
        if len(raw_logits.shape) > 1 and raw_logits.shape[1] > 1:
            # Softmax -> Extract Class 1 (Duplicate probability)
            probs = scipy.special.softmax(raw_logits, axis=1)[:, 1]
        else:
            probs = 1.0 / (1.0 + np.exp(-raw_logits))
            
        return probs
    
def generate_question_hash(question_text: str, options: list) -> str:
        # 1. Normalize the question text (your existing math normalizer)
        norm_q = MathTextNormalizer.normalize(question_text)
    
        # 2. Sort the options alphabetically so permutations become identical
        sorted_options = sorted([MathTextNormalizer.normalize(opt) for opt in options])
    
        # 3. Combine them into a single canonical string
        canonical_string = f"{norm_q} ||| {' | '.join(sorted_options)}"
    
        # 4. Generate a fast SHA-256 hash
        return hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()



# ==========================================
# 3. PERSISTENT VECTOR STORE
# ==========================================
class ExamBankVectorStore:
    def __init__(
        self, 
        dimension: int = 768, 
        bi_encoder_threshold: float = 0.60,
        cross_encoder_threshold: float = 0.85,
        review_threshold: Optional[float] = 0.65
    ):
        self.dimension = dimension
        self.bi_encoder_threshold = bi_encoder_threshold
        self.cross_encoder_threshold = cross_encoder_threshold
        self.review_threshold = review_threshold
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_question: Dict[int, str] = {}
        self.existing_hashes: set = set()  # O(1) Lookup Table
        self.next_id = 0

    def save(self, base_path: str):
        """Saves the FAISS index, JSON metadata, and Hash set to disk."""
        faiss.write_index(self.index, f"{base_path}.index")
        
        meta = {
            "next_id": self.next_id,
            "id_to_question": self.id_to_question
        }
        with open(f"{base_path}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        # Save hashes as a JSON list
        with open(f"{base_path}_hashes.json", "w", encoding="utf-8") as f:
            json.dump(list(self.existing_hashes), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, base_path: str):
        """Loads a saved vector store from disk."""
        instance = cls()
        instance.index = faiss.read_index(f"{base_path}.index")
        
        with open(f"{base_path}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            instance.next_id = meta["next_id"]
            instance.id_to_question = {int(k): v for k, v in meta["id_to_question"].items()}
            
        # Load hashes (with fallback for older databases that don't have this file yet)
        hash_path = f"{base_path}_hashes.json"
        if os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                instance.existing_hashes = set(json.load(f))
                
        return instance

    def process_new_question(
        self, 
        question_text: str, 
        embedding: np.ndarray,
        question_hash: str, 
        reranker: Optional[OnnxCrossEncoder] = None
    ) -> IngestionResult:
        
        # ---------------------------------------------------------
        # STAGE 0: EXACT MATCH HASH CHECK (0ms Compute)
        # ---------------------------------------------------------
        if question_hash in self.existing_hashes:
            return IngestionResult(
                status=IngestionStatus.DUPLICATE_REJECTED, 
                question_id=None, 
                question_text=question_text,
                top_matches=[] # Optionally flag this as an exact hash match
            )

        # ---------------------------------------------------------
        # STAGE 1: FAISS Bi-Encoder
        # ---------------------------------------------------------
        if self.index.ntotal == 0:
            self._add_to_index(question_text, embedding, question_hash)
            return IngestionResult(IngestionStatus.APPROVED, self.next_id - 1, question_text)

        scores, indices = self.index.search(embedding, 5)
        raw_candidates = [
            MatchCandidate(int(idx), self.id_to_question[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0]) if idx != -1
        ]
        
        bi_passed_candidates = [c for c in raw_candidates if c.bi_encoder_score >= self.bi_encoder_threshold]

        if not bi_passed_candidates:
            self._add_to_index(question_text, embedding, question_hash)
            return IngestionResult(IngestionStatus.APPROVED, self.next_id - 1, question_text, raw_candidates)

        # ---------------------------------------------------------
        # STAGE 2: ONNX Cross-Encoder
        # ---------------------------------------------------------
        if reranker is not None:
            ce_eval_pairs = []
            
            for c in bi_passed_candidates:
                # Bypass logic for near-exact matches to prevent OOD paradox
                if c.bi_encoder_score >= 0.99:
                    c.cross_encoder_score = 1.0
                else:
                    ce_eval_pairs.append((question_text, c.question_text))
            
            if ce_eval_pairs:
                ce_probs = reranker.predict_duplicate_probs(ce_eval_pairs)
                prob_idx = 0
                for c in bi_passed_candidates:
                    if c.bi_encoder_score < 0.99:
                        c.cross_encoder_score = float(ce_probs[prob_idx])
                        prob_idx += 1
                        
            bi_passed_candidates.sort(key=lambda c: c.cross_encoder_score, reverse=True)
            top_score = bi_passed_candidates[0].cross_encoder_score
            
            if top_score >= self.cross_encoder_threshold:
                return IngestionResult(IngestionStatus.DUPLICATE_REJECTED, None, question_text, bi_passed_candidates)

        # Passed all checks! Append to DB.
        self._add_to_index(question_text, embedding, question_hash)
        return IngestionResult(IngestionStatus.APPROVED, self.next_id - 1, question_text, bi_passed_candidates)
    def _add_to_index(self, question_text: str, embedding: np.ndarray, question_hash: str):
        self.index.add(embedding)
        self.id_to_question[self.next_id] = question_text
        self.existing_hashes.add(question_hash) # The new hash registry addition
        self.next_id += 1





# ==========================================
# 4. MULTI-BANK MANAGER
# ==========================================
class BankManager:
    """Manages lazy-loading and saving multiple subject banks."""
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.active_banks: Dict[str, ExamBankVectorStore] = {}
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def get_bank(self, subject: str) -> ExamBankVectorStore:
        """Returns a loaded subject bank, creating it if it doesn't exist."""
        subject = subject.lower().strip()
        
        # 1. Return from memory if already loaded
        if subject in self.active_banks:
            return self.active_banks[subject]
            
        base_path = os.path.join(self.data_dir, subject)
        
        # 2. Load from disk if files exist
        if os.path.exists(f"{base_path}.index") and os.path.exists(f"{base_path}_meta.json"):
            print(f"Loading '{subject}' bank from disk...")
            bank = ExamBankVectorStore.load(base_path)
        # 3. Create fresh bank if it doesn't exist
        else:
            print(f"Creating new bank for '{subject}'...")
            bank = ExamBankVectorStore()
            bank.save(base_path) # Initialize files on disk
            
        self.active_banks[subject] = bank
        return bank

    def save_bank(self, subject: str):
        """Forces a single bank to save to disk."""
        subject = subject.lower().strip()
        if subject in self.active_banks:
            base_path = os.path.join(self.data_dir, subject)
            self.active_banks[subject].save(base_path)

    def save_all(self):
        """Saves all currently active banks to disk."""
        for subject, bank in self.active_banks.items():
            base_path = os.path.join(self.data_dir, subject)
            bank.save(base_path)
        print(f"Saved {len(self.active_banks)} active banks to disk.")