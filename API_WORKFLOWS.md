# API Workflow Diagrams - TAFMGR System

## 1. POST /api/recommendations - Get User Recommendations

```
================================================================================
                    API 1: GET USER RECOMMENDATIONS
================================================================================

REQUEST FLOW:
=============

    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   CLIENT     │          │   FastAPI    │          │  Recommend   │
    │  (Request)   │─────────▶│   Endpoint   │─────────▶│   System     │
    └──────────────┘          └──────────────┘          └──────┬───────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDATION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STEP 1: USER VALIDATION                                                            │
│  ┌─────────────────┐                                                               │
│  │ Check if user   │──Yes──▶ STEP 2                                              │
│  │ exists in DB    │                                                               │
│  └────────┬────────┘                                                               │
│           │ No                                                                     │
│           ▼                                                                        │
│  ┌─────────────────┐                                                               │
│  │ Return cold     │                                                               │
│  │ start items     │                                                               │
│  └─────────────────┘                                                               │
│                                                                                     │
│  STEP 2: FETCH USER DATA                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                 │
│  │ User        │  │ User's Past │  │   Item      │                                 │
│  │ Profile     │  │ Interactions│  │   Catalog   │                                 │
│  │ (metadata)  │  │ (history)   │  │ (candidates)│                                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                 │
│         │                │                │                                        │
│         └────────────────┴────────────────┘                                        │
│                          │                                                         │
│                          ▼                                                         │
│  STEP 3: FEATURE EXTRACTION                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                │
│  │  User       │  │   Text      │  │   Image     │                                │
│  │ Embedding   │  │  Features   │  │  Features   │                                │
│  │  (64-dim)   │  │  (1000-dim) │  │  (512-dim)  │                                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                │
│         │                │                │                                        │
│         └────────────────┴────────────────┘                                        │
│                          │                                                         │
│                          ▼                                                         │
│  STEP 4: MULTIMODAL ENCODER                                                        │
│  ┌─────────────────────────────────────────┐                                       │
│  │     Fusion Layer: Concatenate All       │                                       │
│  │     [User || Text || Image] → MLP      │                                       │
│  │     Output: 64-dim feature vector      │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 5: GNN PROPAGATION                                                         │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Message Passing on User-Item Graph    │                                       │
│  │  h_v^(l+1) = σ(Σ μ_uv · h_u^(l) · W) │                                       │
│  │  Output: Graph-aware embeddings        │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 6: TRUST SCORE CALCULATION                                                   │
│  ┌─────────────────────────────────────────┐                                       │
│  │  τ_u = Σ w_i · f_i(history, metadata)  │                                       │
│  │  4 factors: Consistency, Popularity,   │                                       │
│  │             Recency, Activity          │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 7: CANDIDATE SCORING & RANKING                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                           │
│  │ For each    │    │ Calculate   │    │ Apply       │                           │
│  │ candidate   │───▶│ base score  │───▶│ trust boost │                           │
│  │ item        │    │ s_uv        │    │ s_final     │                           │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                           │
│                                                │                                  │
│                                                ▼                                  │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Sort by final score (descending)      │                                       │
│  │  Select Top-K items                    │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 8: FORMAT RESPONSE                                                           │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Add item details: name, category,     │                                       │
│  │  image_url, rating, trust_score        │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
└─────────────────────┼──────────────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  Response    │◀─────────│   Return     │◀─────────│   Top-K      │
    │  (JSON)      │          │   JSON       │          │   Items      │
    └──────────────┘          └──────────────┘          └──────────────┘

JSON RESPONSE:
==============
{
  "user_id": "123",
  "recommendations": [
    {
      "business_id": "abc",
      "name": "Restaurant",
      "predicted_rating": 4.5,
      "trust_score": 0.85
    }
  ],
  "count": 10
}
```

---

## 2. POST /api/similar-items - Find Similar Items

```
================================================================================
                    API 2: FIND SIMILAR ITEMS
================================================================================

REQUEST FLOW:
=============

    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   CLIENT     │          │   FastAPI    │          │   Similar    │
    │  (Request)   │─────────▶│   Endpoint   │─────────▶│   Items      │
    │  item_id     │          │  Validate    │          │   Engine     │
    └──────────────┘          └──────────────┘          └──────┬───────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SIMILARITY CALCULATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STEP 1: FETCH QUERY ITEM                                                          │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Get item_id from request              │                                       │
│  │  Fetch from database:                  │                                       │
│  │    • Text features (TF-IDF)            │                                       │
│  │    • Image features (ResNet)           │                                       │
│  │    • Category, metadata                │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 2: GENERATE ITEM EMBEDDING                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                │
│  │  Item ID    │  │   Text      │  │   Image     │                                │
│  │ Embedding   │  │   MLP       │  │   MLP       │                                │
│  │  (64-dim)   │  │ 1000→64     │  │ 512→64      │                                │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                │
│         │                │                │                                        │
│         └────────────────┼────────────────┘                                        │
│                          │                                                         │
│                          ▼                                                         │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Fusion: Concatenate + MLP              │                                       │
│  │  Output: 64-dim query embedding       │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 3: COMPUTE SIMILARITIES                                                      │
│  ┌─────────────────────────────────────────┐                                       │
│  │  For all items in catalog:              │                                       │
│  │    • Get item embedding (cache or calc) │                                       │
│  │    • Compute cosine similarity:         │                                       │
│  │      sim = (q · i) / (||q|| × ||i||)    │                                       │
│  │  Filter: same category (optional)       │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 4: RANK & SELECT                                                             │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Sort by similarity score (descending)  │                                       │
│  │  Select Top-N most similar items       │                                       │
│  │  (exclude query item itself)           │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 5: FORMAT RESPONSE                                                           │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Add item details for each result      │                                       │
│  │  Include similarity score              │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
└─────────────────────┼──────────────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  Response    │◀─────────│   Return     │◀─────────│  Top-N       │
    │  (JSON)      │          │   JSON       │          │  Similar     │
    └──────────────┘          └──────────────┘          └──────────────┘

JSON RESPONSE:
==============
{
  "item_id": "abc",
  "similar_items": [
    {
      "business_id": "def",
      "name": "Similar Restaurant",
      "similarity": 0.92,
      "category": "Food"
    }
  ],
  "count": 5
}
```

---

## 3. POST /api/interaction - Update User Interactions

```
================================================================================
                    API 3: UPDATE USER INTERACTIONS
================================================================================

REQUEST FLOW:
=============

    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   CLIENT     │          │   FastAPI    │          │  Interaction │
    │  (Request)   │─────────▶│   Endpoint   │─────────▶│   Handler    │
    │ user_id,     │          │  Validate    │          │              │
    │ item_id,     │          │  Data Types  │          │              │
    │ rating       │          │              │          │              │
    └──────────────┘          └──────────────┘          └──────┬───────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INTERACTION UPDATE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STEP 1: VALIDATE INPUT                                                            │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Check:                                │                                       │
│  │    • user_id exists in system          │                                       │
│  │    • item_id exists in catalog         │                                       │
│  │    • rating in valid range [1-5]       │                                       │
│  └────────┬────────┬────────┬────────────┘                                       │
│           │        │        │                                                      │
│          Fail   Fail    Success                                                    │
│           │        │        │                                                      │
│           ▼        ▼        ▼                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐                                        │
│  │ 400 Error│ │ 400 Error│ │ Continue to  │                                        │
│  │ Bad User │ │ Bad Item │ │ Step 2       │                                        │
│  └──────────┘ └──────────┘ └──────┬───────┘                                        │
│                                    │                                               │
│                                    ▼                                               │
│  STEP 2: UPDATE DATABASE                                                           │
│  ┌─────────────────────────────────────────┐                                       │
│  │  SQL INSERT/UPDATE:                    │                                       │
│  │  • interactions table                │                                       │
│  │  • user_item_ratings table           │                                       │
│  │  • Timestamp: CURRENT_TIMESTAMP      │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 3: UPDATE USER PROFILE                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                │
│  │ Increment │  │ Update Last │  │ Recalculate │                                │
│  │ Interaction│  │  Active Time│  │ Avg Rating  │                                │
│  │   Count    │  │             │  │             │                                │
│  └─────────────┘  └─────────────┘  └─────────────┘                                │
│         │                │                │                                        │
│         └────────────────┴────────────────┘                                        │
│                          │                                                         │
│                          ▼                                                         │
│  STEP 4: UPDATE TRUST SCORE (Async)                                                │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Recalculate τ_u for this user:        │                                       │
│  │    • Rating Consistency                │                                       │
│  │    • Item Category Popularity           │                                       │
│  │    • Interaction Recency                │                                       │
│  │    • User Activity Level                │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 5: CACHE UPDATE                                                              │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Clear recommendation cache for user   │                                       │
│  │  (force fresh recommendations)         │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 6: FEDERATED UPDATE (Async)                                                  │
│  ┌─────────────────────────────────────────┐                                       │
│  │  If using FL:                          │                                       │
│  │    • Queue for next training round     │                                       │
│  │    • Update local client dataset       │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
└─────────────────────┼──────────────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────┐          ┌──────────────┐
    │  Response    │◀─────────│   Return     │
    │  (JSON)      │          │   Success    │
    └──────────────┘          └──────────────┘

JSON RESPONSE:
==============
{
  "success": true,
  "message": "Interaction recorded",
  "user_id": "123",
  "item_id": "abc",
  "rating": 4.5,
  "trust_score_updated": true
}
```

---

## 4. GET /api/system-info - Get System Status

```
================================================================================
                    API 4: GET SYSTEM STATUS
================================================================================

REQUEST FLOW:
=============

    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   CLIENT     │          │   FastAPI    │          │   System     │
    │  (Request)   │─────────▶│   Endpoint   │─────────▶│   Monitor    │
    └──────────────┘          └──────────────┘          └──────┬───────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM STATUS COLLECTION                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STEP 1: COLLECT MODEL METRICS                                                     │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Check Model Status:                   │                                       │
│  │    • Encoder loaded?                   │                                       │
│  │    • GNN loaded?                       │                                       │
│  │    • Trust mechanism active?           │                                       │
│  │    • Model version/timestamp           │                                       │
│  └──────────────┬──────────────────────────┘                                       │
│                 │                                                                  │
│                 ├────────────────────────────────┐                                │
│                 │                                │                                │
│                 ▼                                ▼                                │
│  STEP 2: COLLECT DATA METRICS                    STEP 3: COLLECT FL METRICS        │
│  ┌─────────────────────────────────┐            ┌─────────────────────────────┐   │
│  │  Dataset Statistics:            │            │  Federated Status:          │   │
│  │    • Total users: 827           │            │    • FL enabled: true       │   │
│  │    • Total items: 760           │            │    • Num clients: 3         │   │
│  │    • Total interactions: ~2,000 │            │    • Last round: #15        │   │
│  │    • Avg ratings: 3.8           │            │    • Privacy budget: ε=1.2  │   │
│  │    • Data freshness             │            │    • Aggregation: FedAvg    │   │
│  └──────────────┬──────────────────┘            └─────────────┬───────────────┘   │
│                 │                                            │                    │
│                 └────────────────────┬───────────────────────┘                    │
│                                      │                                            │
│                                      ▼                                            │
│  STEP 4: CHECK SYSTEM HEALTH                                                      │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Health Checks:                        │                                       │
│  │    • Database connection: ✓              │                                       │
│  │    • Model inference: ✓                  │                                       │
│  │    • Memory usage: 45%                 │                                       │
│  │    • Response time: <100ms             │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
│                     ▼                                                              │
│  STEP 5: COMPILE RESPONSE                                                          │
│  ┌─────────────────────────────────────────┐                                       │
│  │  Aggregate all metrics into JSON         │                                       │
│  │  Add timestamp and version               │                                       │
│  └──────────────────┬────────────────────┘                                       │
│                     │                                                              │
└─────────────────────┼──────────────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────┐          ┌──────────────┐
    │  Response    │◀─────────│   Return     │
    │  (JSON)      │          │   JSON       │
    └──────────────┘          └──────────────┘

JSON RESPONSE:
==============
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "models": {
    "encoder": "loaded",
    "gnn": "loaded",
    "trust": "active"
  },
  "data": {
    "users": 827,
    "items": 760,
    "interactions": 2156,
    "avg_rating": 3.8
  },
  "federated": {
    "enabled": true,
    "clients": 3,
    "round": 15,
    "epsilon": 1.2
  },
  "health": {
    "database": "connected",
    "memory": "45%",
    "response_time_ms": 85
  }
}
```

---

## API SUMMARY TABLE

| API | Method | Endpoint | Purpose | Key Steps |
|-----|--------|----------|---------|-----------|
| **Recommendations** | POST | `/api/recommendations` | Get personalized items | User validation → Feature extraction → GNN → Trust scoring → Ranking |
| **Similar Items** | POST | `/api/similar-items` | Find similar items | Fetch item → Embedding → Similarity calc → Rank |
| **Interaction** | POST | `/api/interaction` | Record user feedback | Validation → DB update → Trust recalc → Cache clear |
| **System Info** | GET | `/api/system-info` | Check system status | Model status → Data stats → FL status → Health check |
