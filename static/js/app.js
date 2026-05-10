/* ── TrustRec UI JavaScript — with real Yelp photos & business details ── */

// ──────────────────────────────────────────────────────────────
// NAVIGATION
// ──────────────────────────────────────────────────────────────
const sectionTitles = {
    dashboard: 'Dashboard',
    recommend: 'Get Recommendations',
    similar: 'Similar Businesses',
    trust: 'Trust Analytics',
    federated: 'Federated Flow',
    research: 'Research Results',
    animation: 'System Flow — Live Demo',
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
    // Sections inside <main> use .active class; animation section sits outside main and uses display
    document.querySelectorAll('main .section').forEach(s => s.classList.toggle('active', s.id === sec));
    const animSec = document.getElementById('animation');
    if (animSec) {
        if (sec === 'animation') {
            animSec.style.display = 'block';
            // give main a placeholder active so the layout stays sane
            document.querySelector('main').style.display = 'none';
            animSec.classList.add('active');
        } else {
            animSec.style.display = 'none';
            animSec.classList.remove('active');
            document.querySelector('main').style.display = '';
        }
    }
    document.getElementById('topbarTitle').textContent = sectionTitles[sec] || sec;
    if (sec === 'animation') animResetStep();
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
    const dot = document.querySelector('#apiBadge .badge-dot');
    const text = document.getElementById('apiStatus');
    try {
        const r = await fetch('/api/system-info', { signal: AbortSignal.timeout(2000) });
        if (r.ok) { dot.className = 'badge-dot online'; text.textContent = 'API Online'; }
        else throw new Error();
    } catch { dot.className = 'badge-dot offline'; text.textContent = 'Demo Mode'; }
}
checkAPIStatus();
// Start pre-warming the business cache immediately on page load
window.addEventListener('load', () => prewarmCache());

// ──────────────────────────────────────────────────────────────
// BUSINESS CACHE (fetched from API)
// ──────────────────────────────────────────────────────────────
const _bizCache = {};
async function fetchBusiness(itemId) {
  if (_bizCache[itemId]) return _bizCache[itemId];
  try {
    // 8s timeout — business lookup is a simple dict hit on the server, <100ms normally
    const r = await fetch(`/api/business/${itemId}`, { signal: AbortSignal.timeout(8000) });
    if (r.ok) {
      const d = await r.json();
      // Only cache if we got a real photo_url
      _bizCache[itemId] = d;
      return d;
    }
  } catch { }
  // Last-resort fallback: still try to get a photo from a nearby item_id
  return getDemoBusiness(itemId);
}

// Pre-warm cache for item IDs 0-19 on page load (background, no await)
function prewarmCache() {
  for (let i = 0; i < 20; i++) {
    fetch(`/api/business/${i}`, { signal: AbortSignal.timeout(5000) })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) _bizCache[i] = d; })
      .catch(() => {});
  }
}

// ──────────────────────────────────────────────────────────────
// DEMO BUSINESSES (fallback when API is offline)
// No photo_url — emoji placeholders are used instead to avoid
// duplicate-photo issues.  Real photos come from the live API.
// ──────────────────────────────────────────────────────────────
const DEMO_BUSINESSES = [
  { name: 'Pita Jungle',       category: 'Vegetarian',  city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 820,  categories: 'Vegetarian, Healthy, Restaurants',        photo_url: '/api/photo/demo_pita_jungle' },
  { name: 'The Vig',           category: 'Bars',         city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 1200, categories: 'Bars, American, Nightlife',                photo_url: '/api/photo/demo_the_vig' },
  { name: 'Postino Wine Café', category: 'Wine Bars',    city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 950,  categories: 'Wine Bars, Sandwiches, Restaurants',       photo_url: '/api/photo/demo_postino_wine' },
  { name: 'Zinburger',         category: 'Burgers',      city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 730,  categories: 'Burgers, American, Restaurants',           photo_url: '/api/photo/demo_zinburger' },
  { name: "Malee's Thai",      category: 'Thai',         city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 610,  categories: 'Thai, Asian Fusion, Restaurants',          photo_url: '/api/photo/demo_malees_thai' },
  { name: 'Sushi Roku',        category: 'Sushi Bars',   city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 890,  categories: 'Sushi Bars, Japanese, Restaurants',        photo_url: '/api/photo/demo_sushi_roku' },
  { name: 'House of Tricks',   category: 'New American', city: 'Tempe',      state: 'AZ', stars: 4.5, review_count: 1100, categories: 'New American, Restaurants',                photo_url: '/api/photo/demo_house_of_tricks' },
  { name: 'The Mission',       category: 'Latin',        city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 1450, categories: 'Latin, New Mexican Cuisine, Restaurants',  photo_url: '/api/photo/demo_the_mission' },
  { name: 'Cibo',              category: 'Italian',      city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 670,  categories: 'Italian, Pizza, Restaurants',              photo_url: '/api/photo/demo_cibo' },
  { name: 'FnB',               category: 'New American', city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 920,  categories: 'New American, Restaurants, Wine Bars',     photo_url: '/api/photo/demo_fnb' },
  { name: 'Bourbon Steak',     category: 'Steakhouses',  city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 540,  categories: 'Steakhouses, American, Restaurants',       photo_url: '/api/photo/demo_bourbon_steak' },
  { name: 'Blue Hound Kitchen',category: 'American',     city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 430,  categories: 'American, Cocktail Bars, Restaurants',     photo_url: '/api/photo/demo_blue_hound' },
  { name: 'La Grande Orange',  category: 'Grocery',      city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 780,  categories: 'Grocery, Cafes, Restaurants',              photo_url: '/api/photo/demo_la_grande_orange' },
  { name: 'Schmooze Coffee',   category: 'Coffee & Tea', city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 320,  categories: 'Coffee & Tea, Cafes',                      photo_url: '/api/photo/demo_schmooze_coffee' },
  { name: 'Taggia',            category: 'Italian',      city: 'Scottsdale', state: 'AZ', stars: 4.5, review_count: 490,  categories: 'Italian, Mediterranean, Restaurants',      photo_url: '/api/photo/demo_taggia' },
  { name: 'Old Town Tortilla', category: 'Mexican',      city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 860,  categories: 'Mexican, Restaurants',                     photo_url: '/api/photo/demo_old_town_tortilla' },
  { name: 'Bread & Butter',    category: 'Cafes',        city: 'Phoenix',    state: 'AZ', stars: 4.5, review_count: 290,  categories: 'Cafes, Breakfast, Restaurants',            photo_url: '/api/photo/demo_bread_butter' },
  { name: 'Original Chop Shop',category: 'Salads',       city: 'Scottsdale', state: 'AZ', stars: 4.0, review_count: 510,  categories: 'Salads, Healthy, Restaurants',             photo_url: '/api/photo/demo_original_chop_shop' },
  { name: 'Zinburger Wine',    category: 'Burgers',      city: 'Phoenix',    state: 'AZ', stars: 4.0, review_count: 640,  categories: 'Burgers, Wine Bars, Restaurants',          photo_url: '/api/photo/demo_zinburger_wine' },
  { name: 'AZ Caboose',        category: 'American',     city: 'Mesa',       state: 'AZ', stars: 3.5, review_count: 270,  categories: 'American, Bars, Restaurants',              photo_url: '/api/photo/demo_az_caboose' },
];

function getDemoBusiness(itemId) {
  // Returns a demo business (no photo_url) — emoji placeholder will show
  return { ...DEMO_BUSINESSES[itemId % DEMO_BUSINESSES.length] };
}

// ──────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showInputError(container, message) {
  container.innerHTML = `
  <div class="input-error-card">
    <div class="error-icon">⚠️</div>
    <div class="error-title">Invalid Input</div>
    <div class="error-msg">${message}</div>
  </div>`;
}
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
// CATEGORY → EMOJI MAP  (fixes the category[0] first-letter bug)
// ──────────────────────────────────────────────────────────────
const CAT_EMOJI = {
  sushi:'🍣', japanese:'🍱', thai:'🍜', chinese:'🥡', italian:'🍝', pizza:'🍕',
  burger:'🍔', steakhouse:'🥩', bbq:'🍖', seafood:'🦞', mexican:'🌮', indian:'🍛',
  french:'🥐', american:'🍽️', breakfast:'🍳', cafe:'☕', coffee:'☕', bakery:'🥖',
  bar:'🍺', cocktail:'🍹', wine:'🍷', vegan:'🥗', vegetarian:'🥗', sandwich:'🥪',
  noodle:'🍜', korean:'🍲', mediterranean:'🫒', greek:'🫒', salad:'🥙',
  grocery:'🛒', 'ice cream':'🍦', dessert:'🎂', diner:'🍽️', nightlife:'🎶',
};
function catEmoji(cat) {
  if (!cat) return '🍽️';
  const c = String(cat).toLowerCase();
  const match = Object.entries(CAT_EMOJI).find(([k]) => c.includes(k));
  return match ? match[1] : '🍽️';
}

// ──────────────────────────────────────────────────────────────
// BUSINESS CARD BUILDER (with real photo + multimodal badge)
// ──────────────────────────────────────────────────────────────
function buildRecCard(rank, itemId, score, biz, trustScore, showTrust) {
  const emoji = catEmoji(biz.category || biz.categories);
  const photoHtml = biz.photo_url
    ? `<img class="rec-photo" src="${biz.photo_url}" alt="${biz.name}" loading="lazy"
         onerror="this.parentElement.innerHTML='<div class=rec-photo-placeholder><span>${emoji}</span></div>'"/>`
    : `<div class="rec-photo-placeholder"><span>${emoji}</span></div>`;

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
// USER PROFILE CARD BUILDER
// ──────────────────────────────────────────────────────────────
function buildHistoryItem(h) {
  const emoji = catEmoji(h.category || h.categories);
  const photoHtml = h.photo_url
    ? `<img class="hist-photo" src="${h.photo_url}" alt="${h.name}" loading="lazy"
           onerror="this.parentElement.innerHTML='<div class=hist-photo-placeholder><span>${emoji}</span></div>'"/>`
    : `<div class="hist-photo-placeholder"><span>${emoji}</span></div>`;

  const ratingStars = '★'.repeat(Math.round(h.rating)) + '☆'.repeat(5 - Math.round(h.rating));
  const ratingColor = h.rating >= 4 ? 'var(--green)' : h.rating >= 3 ? 'var(--amber)' : 'var(--red)';

  return `
  <div class="hist-item">
    <div class="hist-photo-wrap">${photoHtml}</div>
    <div class="hist-details">
      <div class="hist-name">${h.name || 'Business #' + h.item_id}</div>
      <div class="hist-cats">${(h.categories || h.category || '').split(',').slice(0, 2).map(c => `<span class="cat-tag">${c.trim()}</span>`).join('')}</div>
      ${h.city ? `<div class="hist-loc">${h.city}, ${h.state}</div>` : ''}
    </div>
    <div class="hist-rating" style="color:${ratingColor}">
      <div class="hist-rating-num">${h.rating.toFixed(1)}</div>
      <div class="hist-rating-stars">${ratingStars}</div>
      <div class="hist-rating-label">User rating</div>
    </div>
  </div>`;
}

function buildUserProfileCard(profile) {
  const { user_id, num_interactions, avg_rating, top_categories, interactions } = profile;

  const avgColor = avg_rating >= 4 ? 'var(--green)' : avg_rating >= 3 ? 'var(--amber)' : 'var(--red)';
  const avgStars = '★'.repeat(Math.round(avg_rating)) + '☆'.repeat(5 - Math.round(avg_rating));

  const catPills = top_categories.map(c =>
    `<span class="pref-cat-pill">${catEmoji(c)} ${c}</span>`
  ).join('');

  const historyHtml = interactions.map(h => buildHistoryItem(h)).join('');

  return `
  <div class="user-profile-card glass-card">
    <div class="profile-header">
      <div class="profile-avatar">
        <div class="avatar-circle">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="8" r="4" stroke="#a78bfa" stroke-width="2"/>
            <path d="M4 20c0-4 4-6 8-6s8 2 8 6" stroke="#a78bfa" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="profile-title">User #${user_id}</div>
          <div class="profile-subtitle">Yelp User Profile</div>
        </div>
      </div>
      <div class="profile-stats-row">
        <div class="profile-stat">
          <div class="profile-stat-val">${num_interactions}</div>
          <div class="profile-stat-key">Reviews</div>
        </div>
        <div class="profile-stat">
          <div class="profile-stat-val" style="color:${avgColor}">${avg_rating.toFixed(1)}</div>
          <div class="profile-stat-key">Avg Rating</div>
        </div>
        <div class="profile-stat">
          <div class="profile-stat-val">${top_categories.length}</div>
          <div class="profile-stat-key">Categories</div>
        </div>
      </div>
    </div>

    ${top_categories.length ? `
    <div class="profile-section">
      <div class="profile-section-title">🎯 Preferred Categories</div>
      <div class="pref-cats-wrap">${catPills}</div>
    </div>` : ''}

    <div class="profile-section">
      <div class="profile-section-title">📝 Past Interactions
        <span class="profile-section-badge">${num_interactions} business${num_interactions !== 1 ? 'es' : ''}</span>
      </div>
      <div class="hist-list">${historyHtml}</div>
    </div>

    <div class="profile-arrow">↓ Personalised recommendations based on this history</div>
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
    const userId = parseInt(document.getElementById('userId').value);
    const topK = parseInt(document.getElementById('topKSlider').value);
    const trustAware = document.getElementById('trustAware').checked;
    const container = document.getElementById('recResults');

    // ── Validate input range ─────────────────────────────────────
    if (isNaN(userId) || userId < 0 || userId > 826) {
        showInputError(container, 'User ID must be between <strong>0</strong> and <strong>826</strong>. You entered: <strong>' + (document.getElementById('userId').value || '(empty)') + '</strong>');
        return;
    }

    container.innerHTML = '<div class="loading-spinner"><div class="spinner"></div></div>';

    // ── 1. Fetch user profile ────────────────────────────────────
    let profile = null;
    try {
        const pr = await fetch(`/api/user-profile/${userId}`, { signal: AbortSignal.timeout(5000) });
        if (pr.ok) profile = await pr.json();
    } catch (_) { }

    // Demo fallback profile
    if (!profile) {
        const rng = seededRNG(userId * 31);
        const numHist = 1 + Math.floor(rng() * 4);
        const demoHist = shuffleSeeded([...DEMO_BUSINESSES], rng).slice(0, numHist).map((b, i) => ({
            item_id: (userId * 7 + i * 3) % 760,
            rating: +(1 + Math.floor(rng() * 5)),
            ...b,
        }));
        const allCats = demoHist.flatMap(h => (h.categories || '').split(',').map(c => c.trim())).filter(Boolean);
        const uniqueCats = [...new Set(allCats)].slice(0, 6);
        profile = {
            user_id: userId,
            num_interactions: numHist,
            avg_rating: +(demoHist.reduce((s, h) => s + h.rating, 0) / numHist).toFixed(2),
            top_categories: uniqueCats,
            interactions: demoHist,
        };
    }

    // ── 2. Show profile card ─────────────────────────────────────
    container.innerHTML = buildUserProfileCard(profile);

    // ── 3. Fetch recommendations ─────────────────────────────────
    let recs = null;
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
    } catch (_) { }

    // Demo fallback
    if (!recs) {
        await sleep(500);
        const rng = seededRNG(userId * 13 + topK);
        const picks = shuffleSeeded([...DEMO_BUSINESSES], rng).slice(0, topK);
        recs = picks.map((b, i) => ({
            item_id: (userId * 7 + i * 3) % 760,
            score: +(0.95 - i * 0.04 + (rng() - .5) * 0.05),
            trust_score: +(0.7 + rng() * 0.25),
            metadata: b,
        }));
    }

    // ── 4. Add rec section header ────────────────────────────────
    container.innerHTML += `<div class="rec-section-header">
      <div class="rec-section-title">✦ Recommended For You</div>
      <div class="rec-section-sub">${recs.length} personalised picks · ${trustAware ? 'Trust-Aware' : 'Standard'} mode</div>
    </div>`;

    // ── 5. Render rec cards ──────────────────────────────────────
    for (const [i, rec] of recs.entries()) {
        const liveBiz = await fetchBusiness(rec.item_id);
        const biz = { ...(rec.metadata || {}), ...liveBiz };
        container.innerHTML += buildRecCard(i + 1, rec.item_id, rec.score, biz, rec.trust_score, trustAware);
    }
}

// ──────────────────────────────────────────────────────────────
// SIMILAR BUSINESSES  — shows query business first
// ──────────────────────────────────────────────────────────────
async function fetchSimilar() {
    const itemId = parseInt(document.getElementById('itemId').value);
    const topK = parseInt(document.getElementById('simKSlider').value);
    const container = document.getElementById('simResults');

    // ── Validate input range ─────────────────────────────────────
    if (isNaN(itemId) || itemId < 0 || itemId > 759) {
        showInputError(container, 'Business ID must be between <strong>0</strong> and <strong>759</strong>. You entered: <strong>' + (document.getElementById('itemId').value || '(empty)') + '</strong>');
        return;
    }

    container.innerHTML = '<div class="loading-spinner"><div class="spinner"></div></div>';

    // ── Show the QUERY BUSINESS first ──────────────────────────
    const queryBiz = await fetchBusiness(itemId);
    const catTags = (queryBiz.categories || queryBiz.category || '')
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
    } catch (_) { }

    // Demo fallback
    if (!items) {
        await sleep(300);
        const rng = seededRNG(itemId * 17);
        const picks = shuffleSeeded([...DEMO_BUSINESSES], rng).filter(b => b.name !== queryBiz.name);
        // Ensure unique item_ids in demo fallback
        const usedIds = new Set();
        items = [];
        for (let i = 0; i < picks.length && items.length < topK; i++) {
            const id = (itemId * 11 + i * 7) % 760;  // use prime step to reduce collisions
            if (id === itemId || usedIds.has(id)) continue;
            usedIds.add(id);
            items.push({
                item_id: id,
                similarity: +(0.92 - items.length * 0.07 + (rng() - .5) * 0.03),
                metadata: picks[i],
            });
        }
    }

  // Deduplicate by item_id (safety net even for API results)
  const seenIds = new Set();
  let rank = 0;
  for (const item of items) {
    if (seenIds.has(item.item_id)) continue;
    seenIds.add(item.item_id);
    rank++;
    const liveBiz = await fetchBusiness(item.item_id);
    const biz = { ...(item.metadata || {}), ...liveBiz };
    const shared = sharedCategories(queryBiz, biz);
    container.innerHTML += buildSimilarCard(rank, item.item_id, item.similarity, biz, shared);
  }
}

function sharedCategories(queryBiz, biz) {
    const qCats = (queryBiz.categories || '').split(',').map(c => c.trim().toLowerCase());
    const bCats = (biz.categories || '').split(',').map(c => c.trim().toLowerCase());
    return qCats.filter(c => bCats.includes(c)).slice(0, 2);
}

function buildSimilarCard(rank, itemId, similarity, biz, sharedCats) {
  const emoji = catEmoji(biz.category || biz.categories);
  const photoHtml = biz.photo_url
    ? `<img class="rec-photo" src="${biz.photo_url}" alt="${biz.name}" loading="lazy"
         onerror="this.parentElement.innerHTML='<div class=rec-photo-placeholder><span>${emoji}</span></div>'"/>`
    : `<div class="rec-photo-placeholder"><span>${emoji}</span></div>`;

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
              ${(biz.categories || '').split(',').slice(0, 2).map(c => `<span class="cat-tag">${c.trim()}</span>`).join('')}
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
// ──────────────────────────────────────────────────────────────
// SYSTEM FLOW ANIMATION CONTROLLER
// ──────────────────────────────────────────────────────────────
let _animCurrent = 0;
const _animTotal  = 6;
let   _animTimer  = null;
let   _animPlaying = false;

function animResetStep() {
    _animCurrent = 0;
    _animPlaying = false;
    if (_animTimer) { clearInterval(_animTimer); _animTimer = null; }
    _animRender();
    const btn = document.getElementById('animPlayBtn');
    if (btn) btn.textContent = '▶ Auto-Play';
}

function animStep(dir) {
    _animCurrent = Math.max(0, Math.min(_animTotal - 1, _animCurrent + dir));
    _animRender();
}

function animTogglePlay() {
    _animPlaying = !_animPlaying;
    const btn = document.getElementById('animPlayBtn');
    if (_animPlaying) {
        if (btn) btn.textContent = '⏸ Pause';
        _animTimer = setInterval(() => {
            if (_animCurrent >= _animTotal - 1) {
                _animCurrent = 0;
            } else {
                _animCurrent++;
            }
            _animRender();
        }, 4000);
    } else {
        if (btn) btn.textContent = '▶ Auto-Play';
        if (_animTimer) { clearInterval(_animTimer); _animTimer = null; }
    }
}

function _animRender() {
    // Show/hide panels
    for (let i = 0; i < _animTotal; i++) {
        const p = document.getElementById(`panel-${i}`);
        if (p) p.classList.toggle('active', i === _animCurrent);
    }
    // Update stepper dots
    document.querySelectorAll('.anim-step').forEach((el, i) => {
        el.classList.toggle('active', i === _animCurrent);
        el.classList.toggle('done',   i < _animCurrent);
    });
    // Update step lines
    document.querySelectorAll('.step-line').forEach((el, i) => {
        el.style.background = i < _animCurrent
            ? 'linear-gradient(90deg,#10b981,#34d399)'
            : 'rgba(255,255,255,.08)';
    });
    // Update progress text
    const pt = document.getElementById('animProgressText');
    if (pt) pt.textContent = `Step ${_animCurrent + 1} of ${_animTotal}`;
    // Update buttons
    const prev = document.getElementById('animPrev');
    const next = document.getElementById('animNext');
    if (prev) prev.disabled = _animCurrent === 0;
    if (next) next.disabled = _animCurrent === _animTotal - 1;
    // Trigger loss bar restart on step 2 (training)
    if (_animCurrent === 2) {
        const lf = document.getElementById('lossFill');
        if (lf) { lf.style.animation = 'none'; void lf.offsetWidth; lf.style.animation = ''; }
    }
}
