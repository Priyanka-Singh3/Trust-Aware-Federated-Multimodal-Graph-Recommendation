# API Workflow Explanations for Paper

## Figure X: Recommendation API Workflow

**Caption:** Workflow diagram for the `/api/recommendations` endpoint showing the sequential processing pipeline from user request to personalized recommendation delivery.

**Explanation:**
The recommendation API serves as the primary interface for generating personalized business suggestions to users. Upon receiving a request with a `user_id`, the system first validates the user's existence in the database (Line 1). If the user is not found, a cold-start recommendation strategy is triggered using popular items. For existing users, the system retrieves their profile metadata, interaction history, and the complete item catalog (Line 2).

Feature extraction (Line 3) involves converting raw data into machine-readable representations: text features are extracted using TF-IDF vectorization (1000 dimensions), image features are obtained from a pre-trained ResNet-50 (512 dimensions), and user/item embeddings are generated (64 dimensions each). These features are fused through the multimodal encoder's concatenation layer followed by an MLP projection, producing a unified 64-dimensional feature vector (Line 4).

The GNN propagation step (Line 5) applies message passing on the bipartite user-item graph, updating embeddings through the aggregation formula $h_v^{(l+1)} = \sigma(\sum \mu_{uv} \cdot h_u^{(l)} \cdot W^{(l)})$. This captures collaborative signals from the graph structure.

Trust score calculation (Line 6) combines four behavioral factors: rating consistency (variance of user's ratings), item popularity (average rating of candidate items), interaction recency (time since last activity), and user activity level (total interactions). These are weighted and summed as $\tau_u = \sum w_i \cdot f_i$.

The scoring module (Line 7) predicts base scores using the dot product of user and item embeddings, then applies the trust adjustment: $s_{uv} = \text{base} \times \tau_u$. Items are ranked by final score and the top-K are returned with metadata (Line 8).

---

## Figure X: Similar Items API Workflow

**Caption:** Workflow diagram for the `/api/similar-items` endpoint demonstrating content-based similarity computation using multimodal embeddings.

**Explanation:**
The similar items API enables content-based discovery by finding items with feature profiles matching a query item. The workflow begins with client submission of an `item_id` (Line 1), followed by validation and retrieval of the query item's stored features: TF-IDF text representation, ResNet-50 image features, and categorical metadata (Line 2).

The embedding generation step (Line 3) processes these features through the same multimodal encoder architecture used in recommendations. The item ID embedding (64-dim), text MLP output (1000→64), and image MLP output (512→64) are concatenated and projected through a fusion MLP to produce a 64-dimensional query embedding.

Similarity computation (Line 4) calculates cosine similarity between the query embedding and all items in the catalog: $\text{sim}(q, i) = \frac{q \cdot i}{||q|| \times ||i||}$. For efficiency, embeddings are cached after first computation. Optional category filtering ensures results are semantically relevant.

The ranking step (Line 5) sorts items by similarity score in descending order, excludes the query item itself, and selects the top-N most similar items. The response includes item metadata and similarity scores for interpretability.

---

## Figure X: Interaction Recording API Workflow

**Caption:** Workflow diagram for the `/api/interaction` endpoint showing the feedback loop for updating user profiles and recalculating trust scores.

**Explanation:**
The interaction API captures user feedback to enable online learning and trust score updates. The client submits a tuple of `(user_id, item_id, rating)` (Line 1), which undergoes validation (Line 2): user existence, item existence in catalog, and rating range [1, 5] are verified. Failed validation returns appropriate HTTP 400 errors.

Upon validation, the database is updated (Line 3) through atomic insertions into the `interactions` table (timestamped rating record) and updates to the `user_profile` table (incrementing interaction count, updating last active timestamp, and recalculating the user's average rating).

Trust score recalculation (Line 4) is triggered asynchronously to maintain API responsiveness. The four trust factors are recomputed using the updated interaction history: consistency ($\sigma^2$ of ratings), popularity (weighted average of rated items' global scores), recency (exponential decay of interaction timestamps), and activity (normalized interaction count). The new trust score $\tau_u$ replaces the cached value.

Cache invalidation (Line 5) clears the user's recommendation cache to ensure subsequent requests reflect their updated preferences. In federated mode (Line 6), the interaction is queued for the next training round, contributing to the client's local dataset without immediate model retraining.

---

## Figure X: System Status API Workflow

**Caption:** Workflow diagram for the `/api/system-info` endpoint providing real-time monitoring of model status, dataset statistics, and federated learning progress.

**Explanation:**
The system info API supports monitoring and debugging by exposing internal state metrics. The workflow requires no input parameters (Line 1). Model status checks (Line 2) verify that the encoder, GNN, and trust mechanism components are loaded in memory and report their initialization timestamps.

Dataset statistics (Line 3) query the database for: total registered users (827), total catalog items (760), total recorded interactions (~2,000), and the global average rating (3.8). These metrics indicate data coverage and quality.

Federated learning metrics (Line 4) report: FL enabled status (boolean), number of registered clients (3), current training round number (15), privacy budget consumption ($\epsilon = 1.2$), and aggregation algorithm (FedAvg). These are essential for tracking distributed training progress.

Health checks (Line 5) test database connectivity, measure model inference latency on a dummy input, and report memory utilization. All metrics are compiled into a structured JSON response with a system timestamp for logging and monitoring dashboards.

---

## Table: API Endpoint Summary

| Endpoint | Method | Input Parameters | Output | Processing Time | Primary Function |
|----------|--------|------------------|--------|-----------------|------------------|
| `/api/recommendations` | POST | `user_id`, `k` (optional) | Top-K items with scores | ~150ms | Personalized recommendation |
| `/api/similar-items` | POST | `item_id`, `n` (optional) | Top-N similar items | ~80ms | Content-based discovery |
| `/api/interaction` | POST | `user_id`, `item_id`, `rating` | Success confirmation | ~50ms | Feedback collection |
| `/api/system-info` | GET | None | System metrics JSON | ~20ms | Health monitoring |

---

## Integration with System Architecture

These four APIs collectively enable the complete recommendation service lifecycle:

1. **Discovery** (`/recommendations`, `/similar-items`): Provide the core recommendation functionality using the trained models
2. **Feedback** (`/interaction`): Closes the loop by capturing user behavior for model improvement
3. **Monitoring** (`/system-info`): Ensures operational reliability and tracks federated learning progress

The APIs interact with all major system components: the multimodal encoder for feature extraction, the GNN for collaborative filtering, the trust mechanism for behavior modeling, and the federated learning infrastructure for privacy-preserving updates.
