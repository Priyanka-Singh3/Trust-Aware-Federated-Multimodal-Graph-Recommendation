# API Workflow Diagrams - Concise Version

## 1. POST /api/recommendations

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Client │───▶│ Validate│───▶│ Feature │───▶│  GNN  │───▶│ Trust │───▶│ Score │───▶│ Return │
│ Request│    │  User   │    │ Extract│    │ Prop  │    │ Calc  │    │ & Rank│    │ Top-K  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘    └────────┘    └────────┘
                              │
                              ▼
                         [Text, Image, User]
                              │
                              ▼
                         [Fusion: 64-dim]
```

**Key Steps:** Validate → Extract Features (Text+Image+User) → GNN → Trust Score → Rank → Return

---

## 2. POST /api/similar-items

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Client │───▶│  Fetch │───▶│ Embed  │───▶│ Cosine │───▶│ Return │
│ item_id│    │  Item  │    │  Item  │    │ Similar│    │ Top-N  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
                              │
                              ▼
                    [Query embedding: 64-dim]
                              │
                              ▼
                    [sim = cos(q, i)]
```

**Key Steps:** Fetch Item → Generate Embedding → Cosine Similarity → Rank → Return Similar

---

## 3. POST /api/interaction

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Client │───▶│Validate│───▶│ Update │───▶│ Recalc │───▶│ Return │
│(user,  │    │(user,  │    │   DB   │    │ Trust  │    │ Success│
│ item,  │    │ item,  │    │        │    │ Score  │    │        │
│ rating)│    │ rating)│    │        │    │        │    │        │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
                              │                │
                              ▼                ▼
                    [Tables: interactions]   [τ_u = Σ w·f]
                    [user_profile]           [4 factors]
```

**Key Steps:** Validate → Update DB → Recalculate Trust → Return Success

---

## 4. GET /api/system-info

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Client │───▶│  Get   │───▶│  Get   │───▶│ Return │
│ Request│    │ Models │    │  Stats │    │  JSON  │
└────────┘    └────────┘    └────────┘    └────────┘
                  │              │
                  ▼              ▼
            [Encoder, GNN]  [Users: 827]
            [Trust]         [Items: 760]
                            [Interactions: 2156]
```

**Key Steps:** Check Models → Get Stats → Return Status JSON

---

## Summary Table

| API | Input | Core Processing | Output |
|-----|-------|-----------------|--------|
| `/recommendations` | user_id | Features→GNN→Trust→Rank | Top-K items |
| `/similar-items` | item_id | Embedding→Cosine Similarity | Top-N similar |
| `/interaction` | (user, item, rating) | Validate→Update→Recalc Trust | Success/Error |
| `/system-info` | - | Check Models + Stats | System status JSON |
