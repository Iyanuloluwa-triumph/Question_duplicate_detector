import app

# ==========================================
# 1. INITIALIZE COMPONENTS
# ==========================================
embedder = BiEncoderEmbedder(
    base_model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    adapter_path="/kaggle/working/question_duplicate_model"
)

cross_encoder = CrossEncoderReranker(
    model_path="/kaggle/working/cross_encoder_yoruba_stem",
)

vector_db = ExamBankVectorStore(
    dimension=768,
    bi_encoder_threshold=0.60,      # Stage 1 Gate: Minimum similarity to trigger Cross-Encoder
    cross_encoder_threshold=0.85,   # Stage 2 Final Block: Auto-reject duplicates >= 85% prob
    review_threshold=0.65          # Stage 2 Gray Zone: Flags for human review (65% - 84%)
)

# ==========================================
# 2. SEED INDEX WITH INITIAL QUESTION
# ==========================================
existing_questions = [
    "Find the roots of the quadratic equation 2x^2 - 5x + 3 = 0."
]
seed_embeddings = embedder.encode(existing_questions)
vector_db.add_existing_bank(existing_questions, seed_embeddings)

# ==========================================
# 3. TEST INCOMING QUESTIONS
# ==========================================
test_questions = [
    # Case 1: Unrelated Question -> Bi-Encoder rejects candidate straight up (< 0.60)
    "Calculate the derivative of sin(x) with respect to x.",
    
    # Case 2: Hard Negative (Sign Flip) -> Passes Bi-Encoder (~0.99), Rejected by Cross-Encoder (< 0.30)
    "Find the roots of the quadratic equation 2x^2 + 5x - 3 = 0.",
    
    # Case 3: Rephrased Duplicate -> Passes Bi-Encoder (~0.98), Confirmed by Cross-Encoder (> 0.95)
    "Determine the values of x that satisfy the quadratic equation 2x^2 - 5x + 3 = 0."
]

print("\n" + "="*80)
print("RUNNING 2-STAGE INGESTION TEST")
print("="*80)

for incoming_q in test_questions:
    q_emb = embedder.encode([incoming_q])
    
    # MUST pass reranker=cross_encoder here!
    result = vector_db.process_new_question(
        question_text=incoming_q, 
        embedding=q_emb, 
        reranker=cross_encoder
    )
    
    print(f"\nIncoming: '{result.question_text}'")
    print(f"Status:   [{result.status.value}]")
    
    if result.top_matches:
        print("Matches Evaluated:")
        for m in result.top_matches:
            ce_score_str = f"{m.cross_encoder_score:.4f}" if m.cross_encoder_score is not None else "N/A (Bypassed)"
            print(f"  - ID {m.question_id} | Stage 1 Sim: {m.bi_encoder_score:.4f} | Stage 2 Prob: {ce_score_str}")
            print(f"    Text: '{m.question_text}'")
    else:
        print("  - No candidates met Stage 1 threshold (Auto-Approved immediately)")