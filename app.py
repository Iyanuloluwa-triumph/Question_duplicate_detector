import os
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Import components from vector_engine.py
from vector_engine import (
    OnnxBiEncoder,
    OnnxCrossEncoder,
    BankManager,
    IngestionStatus,
    IngestionResult,
    MatchCandidate,
    generate_question_hash
)

# ==========================================
# 1. GLOBAL INSTANCES (SINGLETONS)
# ==========================================
MODELS_DIR = "./models"
BI_ENCODER_PATH = os.path.join(MODELS_DIR, "bi_encoder_onnx_int8")
CROSS_ENCODER_PATH = os.path.join(MODELS_DIR, "cross_encoder_onnx_int8")
DATA_DIR = "./data"

bi_encoder: Optional[OnnxBiEncoder] = None
cross_encoder: Optional[OnnxCrossEncoder] = None
bank_manager: Optional[BankManager] = None



# ==========================================
# 2. LIFESPAN MANAGEMENT (STARTUP / SHUTDOWN)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bi_encoder, cross_encoder, bank_manager
    
    print("Starting up Vector Search Service...")
    
    # Load ONNX models into memory ONCE
    print(f"Loading ONNX Bi-Encoder from {BI_ENCODER_PATH}...")
    bi_encoder = OnnxBiEncoder(BI_ENCODER_PATH)
    
    print(f"Loading ONNX Cross-Encoder from {CROSS_ENCODER_PATH}...")
    cross_encoder = OnnxCrossEncoder(CROSS_ENCODER_PATH)
    
    # Initialize Multi-Bank Manager
    bank_manager = BankManager(data_dir=DATA_DIR)
    
    print("All ONNX models and Bank Manager initialized successfully.")
    
    yield  # Server serves HTTP requests here
    
    # Shutdown logic
    print("Server shutting down: Flushing all vector banks to disk...")
    if bank_manager:
        bank_manager.save_all()
    print("Shutdown complete.")


app = FastAPI(
    title="Exam Question Deduplication API",
    description="2-Stage Vector Search + ONNX Cross-Encoder Ingestion Engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (fine for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ==========================================
# 3. PYDANTIC SCHEMAS
# ==========================================
class MatchCandidateSchema(BaseModel):
    question_id: int
    question_text: str
    bi_encoder_score: float
    cross_encoder_score: Optional[float] = None

class SingleIngestRequest(BaseModel):
    subject: str = Field(..., example="mathematics")
    question_text: str = Field(..., example="Find the roots of 2x^2 - 5x + 3 = 0")

class SingleIngestResponse(BaseModel):
    subject: str
    status: str
    question_id: Optional[int]
    question_text: str
    message: str
    top_matches: List[MatchCandidateSchema]

class BatchIngestRequest(BaseModel):
    subject: str = Field(..., example="english")
    questions: List[str] = Field(..., example=[
        "Identify the noun in the sentence.",
        "Which of the following is a transitive verb?"
    ])

class BatchIngestResponse(BaseModel):
    subject: str
    total_processed: int
    total_approved: int
    total_rejected: int
    total_needs_review: int
    results: List[SingleIngestResponse]

class BankStatsResponse(BaseModel):
    subject: str
    total_questions: int


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.post("/ingest/single", response_model=SingleIngestResponse, status_code=status.HTTP_200_OK)
async def ingest_single_question(payload: SingleIngestRequest):
    """
    Ingests a single question into the specified subject vector bank.
    Evaluates candidates using Stage 1 (FAISS) + Stage 2 (Cross-Encoder).
    """
    if not payload.question_text.strip():
        raise HTTPException(status_code=400, detail="Question text cannot be empty.")
        
    subject = payload.subject.lower().strip()
    bank = bank_manager.get_bank(subject)
    # Generate the Stage 0 Hash
    q_hash = generate_question_hash(payload.question_text, payload.options)
    
    # Optionally: Embed the concatenated canonical text so options are vectorized
    canonical_text = f"{payload.question_text} {' '.join(payload.options)}"
    
    # Generate 768-dim normalized embedding via ONNX
    embedding = bi_encoder.encode([payload.question_text])
    
    # Process through 2-stage pipeline
    result: IngestionResult = bank.process_new_question(
        question_text=payload.question_text,
        embedding=embedding,
        question_hash=q_hash,
        reranker=cross_encoder
    )
    
    # Save bank state to disk
    bank_manager.save_bank(subject)
    
    # Format candidates for response
    formatted_candidates = [
        MatchCandidateSchema(
            question_id=m.question_id,
            question_text=m.question_text,
            bi_encoder_score=m.bi_encoder_score,
            cross_encoder_score=m.cross_encoder_score
        )
        for m in result.top_matches
    ]
    
    status_str = result.status.value
    msg_map = {
        IngestionStatus.APPROVED.value: "Question approved and indexed successfully.",
        IngestionStatus.DUPLICATE_REJECTED.value: "Question rejected as a semantic duplicate.",
        IngestionStatus.NEEDS_HUMAN_REVIEW.value: "Question flagged for human review."
    }
    
    return SingleIngestResponse(
        subject=subject,
        status=status_str,
        question_id=result.question_id,
        question_text=result.question_text,
        message=msg_map.get(status_str, ""),
        top_matches=formatted_candidates
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse, status_code=status.HTTP_200_OK)
async def ingest_batch_questions(payload: BatchIngestRequest):
    """
    Ingests multiple questions into a subject bank in a single batch operation.
    Vectorizes all inputs simultaneously for optimum speed.
    """
    if not payload.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty.")
        
    subject = payload.subject.lower().strip()
    bank = bank_manager.get_bank(subject)
    
    # Vectorize all texts in one matrix pass for efficiency
    embeddings = bi_encoder.encode(payload.questions)
    
    total_approved = 0
    total_rejected = 0
    total_needs_review = 0
    individual_results: List[SingleIngestResponse] = []
    
    for q_text, single_emb in zip(payload.questions, embeddings):
        # Keep shape as [1, 768]
        single_emb_arr = np.expand_dims(single_emb, axis=0)
        q_hash = generate_question_hash(q_text, payload.options)
        
        
        result: IngestionResult = bank.process_new_question(
            question_text=q_text,
            embedding=single_emb_arr,
            question_hash=q_hash,
            reranker=cross_encoder
        )
        
        status_str = result.status.value
        if result.status == IngestionStatus.APPROVED:
            total_approved += 1
        elif result.status == IngestionStatus.DUPLICATE_REJECTED:
            total_rejected += 1
        elif result.status == IngestionStatus.NEEDS_HUMAN_REVIEW:
            total_needs_review += 1
            
        formatted_candidates = [
            MatchCandidateSchema(
                question_id=m.question_id,
                question_text=m.question_text,
                bi_encoder_score=m.bi_encoder_score,
                cross_encoder_score=m.cross_encoder_score
            )
            for m in result.top_matches
        ]
        
        individual_results.append(
            SingleIngestResponse(
                subject=subject,
                status=status_str,
                question_id=result.question_id,
                question_text=result.question_text,
                message=f"Status: {status_str}",
                top_matches=formatted_candidates
            )
        )
        
    # Flush batch updates to disk
    bank_manager.save_bank(subject)
    
    return BatchIngestResponse(
        subject=subject,
        total_processed=len(payload.questions),
        total_approved=total_approved,
        total_rejected=total_rejected,
        total_needs_review=total_needs_review,
        results=individual_results
    )


@app.get("/bank/{subject}/stats", response_model=BankStatsResponse)
async def get_bank_stats(subject: str):
    """Returns the total number of approved questions currently indexed in a subject bank."""
    subject_clean = subject.lower().strip()
    bank = bank_manager.get_bank(subject_clean)
    return BankStatsResponse(
        subject=subject_clean,
        total_questions=bank.index.ntotal
    )


@app.get("/bank/list")
async def list_active_banks():
    """Lists all subjects currently loaded in memory and their active status."""
    return {
        "active_in_memory": list(bank_manager.active_banks.keys()),
        "data_directory": DATA_DIR
    }


# ==========================================
# 5. ENTRY POINT FOR DIRECT EXECUTION
# ==========================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)