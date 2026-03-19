# RAG-based-QnA

## FastAPI Adaptive Interview Engine (No UI)

### 1. Build the vector DB

```powershell
python .\RAG\make_db.py
```

### 2. Run the API server

```powershell
uvicorn RAG.server:app --reload
```

### 3. Endpoints

- `GET /health`
- `POST /interview/start`
- `GET /interview/{session_id}/current`
- `POST /interview/{session_id}/answer`
- `GET /interview/{session_id}/report`

### 4. Example flow

1. Start interview:
```http
POST /interview/start
Content-Type: application/json

{
  "num_questions": 6
}
```

Response (includes first question):
```json
{
  "session_id": "uuid",
  "current_question": {
    "question_id": 12,
    "question": "What is overfitting and how can you prevent it?",
    "generated": false,
    "focus": "core concept"
  },
  "progress": {
    "answered": 0,
    "target": 6,
    "remaining": 6
  }
}
```

2. Submit answer:
```http
POST /interview/{session_id}/answer
Content-Type: application/json

{
  "question_id": 12,
  "student_answer": "Your answer..."
}
```

Response includes:
- strict scoring (`score`, `factuality`, `context`, `originality`, `example`, `injection`)
- strengths and improvement actions
- adaptive `next_question` generated based on previous answer quality

3. Final interview report:
```http
GET /interview/{session_id}/report
```

Report includes:
- average score
- per-dimension averages
- what candidate did well
- what to improve
- actionable next steps

Evaluation object schema:
```json
{
  "score": 0,
  "factuality": 0,
  "context": 2,
  "originality": 0,
  "example": 0,
  "injection": true,
  "feedback": "...",
  "strengths": ["..."],
  "improvements": ["..."]
}
```
