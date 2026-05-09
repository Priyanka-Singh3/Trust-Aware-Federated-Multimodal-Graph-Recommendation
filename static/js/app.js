/* ── TrustRec UI JavaScript — with real Yelp photos & business details ── */

// ──────────────────────────────────────────────────────────────
// NAVIGATION
// ──────────────────────────────────────────────────────────────
const sectionTitles = {
  dashboard: 'Dashboard',
  recommend: 'Get Recommendations',
  similar:   'Similar Businesses',
  trust:     'Trust Analytics',
  federated: 'Federated Flow',
  research:  'Research Results',
};

document.querySelectorAll('.nav-item').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    navigate(link.dataset.section);
    document.getElementById('sidebar').classList.remove('open');
  });
});

document.getElementById('menuToggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
});

function navigate(sec) {
  document.querySelectorAll('.nav-item').forEach(l => l.classList.toggle('active', l.dataset.section === sec));
  document.querySelectorAll('.section').forEach(s => s.classList.toggle('active', s.id === sec));
  document.getElementById('topbarTitle').textContent = sectionTitles[sec] || sec;
}

// ──────────────────────────────────────────────────────────────
// STAT COUNTER ANIMATION
// ──────────────────────────────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-value[data-target]').forEach(el => {
    const target = parseInt(el.dataset.target);
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 60));
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current.toLocaleString();
      if (current >= target) clearInterval(timer);
    }, 16);
  });
}
window.addEventListener('load', animateCounters);

// ──────────────────────────────────────────────────────────────
// API STATUS
// ──────────────────────────────────────────────────────────────
async function checkAPIStatus() {
  const dot  = document.querySelector('#apiBadge .badge-dot');
  const text = document.getElementById('apiStatus');
  try {
    const r = await fetch('/api/system-info', { signal: AbortSignal.timeout(2000) });
    if (r.ok) { dot.className = 'badge-dot online'; text.textContent = 'API Online'; }
    else throw new Error();
  } catch { dot.className = 'badge-dot offline'; text.textContent = 'Demo Mode'; }
}
checkAPIStatus();

// ──────────────────────────────────────────────────────────────
// BUSINESS CACHE (fetched from API)
// ──────────────────────────────────────────────────────────────
const _bizCache = {};
async function fetchBusiness(itemId) {
  if (_bizCache[itemId]) return _bizCache[itemId];
  try {
    const r = await fetch(`/api/business/${itemId}`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) { const d = await r.json(); _bizCache[itemId] = d; return d; }
  } catch {}
  // Fallback demo data
  return getDemoBusiness(itemId);
}

// ──────────────────────────────────────────────────────────────
// DEMO BUSINESSES (fallback when API is offline)
// ──────────────────────────────────────────────────────────────
const DEMO_BUSINESSES = [
  { name: 'Pita Jungle',        category: 'Vegetarian',  city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 820,  categories: 'Vegetarian, Healthy, Restaurants' },
  { name: 'The Vig',            category: 'Bars',        city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 1200, categories: 'Bars, American, Nightlife' },
  { name: 'Postino Wine Café',  category: 'Wine Bars',   city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 950,  categories: 'Wine Bars, Sandwiches, Restaurants' },
  { name: 'Zinburger',          category: 'Burgers',     city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 730,  categories: 'Burgers, American, Restaurants' },
  { name: "Malee's Thai",       category: 'Thai',        city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 610,  categories: 'Thai, Asian Fusion, Restaurants' },
  { name: 'Sushi Roku',         category: 'Sushi Bars',  city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 890,  categories: 'Sushi Bars, Japanese, Restaurants' },
  { name: 'House of Tricks',    category: 'New American', city: 'Tempe',     state: 'AZ', stars: 4.5, review_count: 1100, categories: 'New American, Restaurants' },
  { name: 'The Mission',        category: 'Latin',       city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 1450, categories: 'Latin, New Mexican Cuisine, Restaurants' },
  { name: 'Cibo',               category: 'Italian',     city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 670,  categories: 'Italian, Pizza, Restaurants' },
  { name: 'FnB',                category: 'New American', city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 920, categories: 'New American, Restaurants, Wine Bars' },
  { name: 'Bourbon Steak',      category: 'Steakhouses', city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 540,  categories: 'Steakhouses, American, Restaurants' },
  { name: 'Blue Hound Kitchen', category: 'American',    city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 430,  categories: 'American, Cocktail Bars, Restaurants' },
  { name: 'La Grande Orange',   category: 'Grocery',     city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 780,  categories: 'Grocery, Cafes, Restaurants' },
  { name: 'Schmooze Coffee',    category: 'Coffee & Tea', city: 'Phoenix',   state: 'AZ', stars: 4.5, review_count: 320,  categories: 'Coffee & Tea, Cafes' },
  { name: 'Taggia',             category: 'Italian',     city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 490,  categories: 'Italian, Mediterranean, Restaurants' },
  { name: 'Old Town Tortilla',  category: 'Mexican',     city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 860,  categories: 'Mexican, Restaurants' },
  { name: 'Bread & Butter',     category: 'Cafes',       city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 290,  categories: 'Cafes, Breakfast, Restaurants' },
  { name: 'Original Chop Shop', category: 'Salads',      city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 510,  categories: 'Salads, Healthy, Restaurants' },
  { name: 'Zinburger Wine',     category: 'Burgers',     city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 640,  categories: 'Burgers, Wine Bars, Restaurants' },
  { name: 'AZ Caboose',         category: 'American',    city: 'Mesa',       state: 'AZ', stars: 3.5, review_count: 270,  categories: 'American, Bars, Restaurants' },
];

function getDemoBusiness(itemId) {
  const b = DEMO_BUSINESSES[itemId % DEMO_BUSINESSES.length];
  return { ...b, photo_url: null };
}

// ──────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function seededRNG(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}
function shuffleSeeded(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
function stars(n) {
  const full = Math.floor(n), half = n % 1 >= .5 ? 1 : 0;
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(5 - full - half);
}

// ──────────────────────────────────────────────────────────────
// BUSINESS CARD BUILDER (with real photo + multimodal badge)
// ──────────────────────────────────────────────────────────────
function buildRecCard(rank, itemId, score, biz, trustScore, showTrust) {
  const photoHtml = biz.photo_url
    ? `<img class="rec-photo" src="${biz.photo_url}" alt="${biz.name}" onerror="this.style.display='none'"/>`
    : `<div class="rec-photo-placeholder"><span>${biz.category?.[0]||'🍽️'}</span></div>`;

  const multimodalBadge = biz.photo_url
    ? `<span class="mm-badge" title="Multimodal: trained on this real photo">📷 Multimodal</span>`
    : '';

  const trustHtml = showTrust
    ? `<div class="rec-trust">Trust ${trustScore ? trustScore.toFixed(2) : '–'}</div>` : '';

  const catTags = (biz.categories || biz.category || '')
    .split(',').slice(0, 3).map(c => `<span class="cat-tag">${c.trim()}</span>`).join('');

  return `
    <div class="rec-card" style="animation-delay:${rank * 60}ms">
      <div class="rec-photo-wrap">${photoHtml}</div>
      <div class="rec-card-body">
        <div class="rec-top">
          <div class="rec-rank">${rank}</div>
          <div class="rec-info">
            <div class="rec-name">${biz.name || 'Business #' + itemId}</div>
            <div class="rec-meta">
              <span class="rec-stars">${stars(biz.stars || 4.0)}</span>
              <span class="rec-reviews">${(biz.review_count || 0).toLocaleString()} reviews</span>
              ${biz.city ? `· ${biz.city}, ${biz.state || ''}` : ''}
            </div>
            <div class="rec-tags">${catTags} ${multimodalBadge}</div>
          </div>
          <div class="rec-right">
            <div class="rec-score">${typeof score === 'number' ? score.toFixed(3) : score}</div>
            <div class="rec-score-label">score</div>
            ${trustHtml}
          </div>
        </div>
      </div>
    </div>`;
}

// ──────────────────────────────────────────────────────────────
// RECOMMENDATIONS
// ──────────────────────────────────────────────────────────────
function setDemo(uid) {
  document.getElementById('userId').value = uid;
  fetchRecommendations();
}

async function fetchRecommendations() {
  const userId    = parseInt(document.getElementById('userId').value);
  const topK      = parseInt(document.getElementById('topKSlider').value);
  const trustAware = document.getElementById('trustAware').checked;
  const container = document.getElementById('recResults');

  container.innerHTML = '<div class="loading-spinner"><div class="spinner"></div></div>';

  let recs = null;

  // Try live API
  try {
    const r = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, top_k: topK, trust_aware: trustAware }),
      signal: AbortSignal.timeout(4000)
    });
    if (r.ok) {
      const data = await r.json();
      if (data.success && data.recommendations?.length) recs = data.recommendations;
    }
  } catch (_) {}

  // Demo fallback
  if (!recs) {
    await sleep(500);
    const rng   = seededRNG(userId * 13 + topK);
    const picks = shuffleSeeded([...DEMO_BUSINESSES], rng).slice(0, topK);
    recs = picks.map((b, i) => ({
      item_id:    (userId * 7 + i * 3) % 760,
      score:      +(0.95 - i * 0.04 + (rng() - .5) * 0.05),
      trust_score: +(0.7 + rng() * 0.25),
      metadata:   b,
    }));
  }

  // Enrich with real business data from API
  container.innerHTML = '';
  for (const [i, rec] of recs.entries()) {
    const biz = rec.metadata || await fetchBusiness(rec.item_id);
    container.innerHTML += buildRecCard(i + 1, rec.item_id, rec.score, biz, rec.trust_score, trustAware);
  }
}

// ──────────────────────────────────────────────────────────────
// SIMILAR BUSINESSES  — shows query business first
// ──────────────────────────────────────────────────────────────
async function fetchSimilar() {
  const itemId    = parseInt(document.getElementById('itemId').value);
  const topK      = parseInt(document.getElementById('simKSlider').value);
  const container = document.getElementById('simResults');

  container.innerHTML = '<div class="loading-spinner"><div class="spinner"></div></div>';

  // ── Show the QUERY BUSINESS first ──────────────────────────
  const queryBiz = await fetchBusiness(itemId);
  const catTags  = (queryBiz.categories || queryBiz.category || '')
    .split(',').slice(0, 4).map(c => `<span class="cat-tag highlight-tag">${c.trim()}</span>`).join('');
  const photoHtml = queryBiz.photo_url
    ? `<img class="query-photo" src="${queryBiz.photo_url}" alt="${queryBiz.name}" onerror="this.style.display='none'"/>`
    : `<div class="query-photo-placeholder">🏪</div>`;
  const mmBadge = queryBiz.photo_url
    ? `<span class="mm-badge">📷 Multimodal — trained on this real photo</span>` : '';

  container.innerHTML = `
    <div class="query-biz-card glass-card">
      <div class="card-title">📍 You searched for Business #${itemId}</div>
      <div class="query-biz-body">
        <div class="query-photo-wrap">${photoHtml}</div>
        <div class="query-biz-info">
          <div class="query-biz-name">${queryBiz.name || 'Business #' + itemId}</div>
          <div class="query-biz-meta">
            <span class="rec-stars">${stars(queryBiz.stars || 4.0)}</span>
            <span class="rec-reviews">${(queryBiz.review_count || 0).toLocaleString()} reviews</span>
            ${queryBiz.city ? ` · ${queryBiz.city}, ${queryBiz.state}` : ''}
          </div>
          <div class="rec-tags">${catTags} ${mmBadge}</div>
          <div class="query-desc">${queryBiz.description || queryBiz.categories || ''}</div>
        </div>
      </div>
      <div class="similar-arrow">↓ Businesses with similar GNN embeddings</div>
    </div>`;

  // ── Fetch similar items ─────────────────────────────────────
  let items = null;
  try {
    const r = await fetch('/api/similar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: itemId, top_k: topK }),
      signal: AbortSignal.timeout(4000)
    });
    if (r.ok) {
      const data = await r.json();
      if (data.success && data.similar_items?.length) items = data.similar_items;
    }
  } catch (_) {}

  // Demo fallback
  if (!items) {
    await sleep(300);
    const rng   = seededRNG(itemId * 17);
    const picks = shuffleSeeded([...DEMO_BUSINESSES], rng).filter(b => b.name !== queryBiz.name).slice(0, topK);
    items = picks.map((b, i) => ({
      item_id:    (itemId * 11 + i * 5) % 760,
      similarity: +(0.92 - i * 0.07 + (rng() - .5) * 0.03),
      metadata:   b,
    }));
  }

  // Render similar cards
  for (const [i, item] of items.entries()) {
    const biz = item.metadata || await fetchBusiness(item.item_id);
    // Show similarity score + what they share
    const shared = sharedCategories(queryBiz, biz);
    container.innerHTML += buildSimilarCard(i + 1, item.item_id, item.similarity, biz, shared);
  }
}

function sharedCategories(queryBiz, biz) {
  const qCats = (queryBiz.categories || '').split(',').map(c => c.trim().toLowerCase());
  const bCats = (biz.categories    || '').split(',').map(c => c.trim().toLowerCase());
  return qCats.filter(c => bCats.includes(c)).slice(0, 2);
}

function buildSimilarCard(rank, itemId, similarity, biz, sharedCats) {
  const photoHtml = biz.photo_url
    ? `<img class="rec-photo" src="${biz.photo_url}" alt="${biz.name}" onerror="this.style.display='none'"/>`
    : `<div class="rec-photo-placeholder"><span>${biz.category?.[0]||'🍽️'}</span></div>`;

  const sharedHtml = sharedCats.length
    ? `<div class="shared-cats">Shared: ${sharedCats.map(c => `<span class="cat-tag shared-tag">${c}</span>`).join('')}</div>` : '';

  const mmBadge = biz.photo_url ? `<span class="mm-badge">📷 Multimodal</span>` : '';

  return `
    <div class="rec-card" style="animation-delay:${rank * 60}ms; border-color: rgba(14,165,233,.2)">
      <div class="rec-photo-wrap">${photoHtml}</div>
      <div class="rec-card-body">
        <div class="rec-top">
          <div class="rec-rank" style="background:linear-gradient(135deg,#0ea5e9,#0284c7)">${rank}</div>
          <div class="rec-info">
            <div class="rec-name">${biz.name || 'Business #' + itemId}</div>
            <div class="rec-meta">
              <span class="rec-stars">${stars(biz.stars || 4.0)}</span>
              <span class="rec-reviews">${(biz.review_count || 0).toLocaleString()} reviews</span>
              ${biz.city ? `· ${biz.city}, ${biz.state}` : ''}
            </div>
            <div class="rec-tags">
              ${(biz.categories||'').split(',').slice(0,2).map(c=>`<span class="cat-tag">${c.trim()}</span>`).join('')}
              ${mmBadge}
            </div>
            ${sharedHtml}
          </div>
          <div class="rec-right">
            <div class="rec-score" style="color:#38bdf8">${similarity.toFixed(3)}</div>
            <div class="rec-score-label" style="color:#38bdf8">similarity</div>
            <div class="rec-trust" style="background:rgba(14,165,233,.12);color:#38bdf8">GNN embedding</div>
          </div>
        </div>
      </div>
    </div>`;
}