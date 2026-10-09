# Question Duplicate Detector

A semantic matching system for finding repeated or substantially similar questions in exam question banks. It uses sentence embeddings to retrieve likely matches efficiently, followed by a cross-encoder to assess candidate pairs in greater detail.

## Overview

Exact text matching can miss duplicate questions when their wording differs. This project uses `sentence-transformers/all-MiniLM-L6-v2` to encode questions as vectors and identify semantically similar candidates. It is intended to help exam administrators review question banks and flag duplicates before questions are used in exams such as JAMB and WAEC.

## Workflow

1. **Build the dataset:** collect JAMB past questions and Quora question-pair examples, then preprocess and combine the data.
2. **Fine-tune the embedding model:** train the sentence-transformer on question pairs to improve its representation of question similarity and technical concepts.
3. **Retrieve candidates:** encode questions and use vector similarity to shortlist likely matches rather than exhaustively comparing every pair in a large database.
4. **Rerank candidates:** pass shortlisted pairs to a cross-encoder, which evaluates both questions together for a more detailed duplicate assessment.
5. **Review flagged pairs:** use model results to support question-bank review and cleanup.

## Features

- Semantic matching that is not limited to identical wording.
- A two-stage embedding and cross-encoder approach for scalable candidate discovery and pair-level assessment.
- Fine-tuning data combining exam questions and general question-pair examples.
- An exam-question quality-control use case.

## Intended use and limitations

Treat predictions as recommendations for review, not as definitive decisions. Related questions may test different knowledge, and duplicate questions can differ substantially in wording. Human review is recommended before removing or changing questions. Accuracy depends on data quality and coverage and may vary across subjects, exam boards, and question formats.

Before distributing or deploying the project, verify the licensing, attribution, and permitted use of all datasets and pretrained models. Consult the repository's code and data documentation for implementation-specific setup and usage instructions.
