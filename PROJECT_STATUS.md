# Total Search - Project Status Document
**Last Updated:** 2026-01-20
**Purpose:** Resume development in a new conversation

---

## Project Overview
Multi-platform keyword research tool that aggregates search demand data from 5 platforms:
- Google (DataForSEO API)
- YouTube (Google Trends API - NEW)
- Amazon (Jungle Scout API - needs API key)
- TikTok (Apify scraper - proxy scores)
- Instagram (Apify scraper - proxy scores)

**Live URL:** https://total-search.vercel.app (or your Vercel domain)
**GitHub:** https://github.com/mohamedali170683-dotcom/Total-Search

---

## Current Deployment Issue (BLOCKING)

**Problem:** Vercel deployment failing with cached pycache error:
```
[Error: ENOENT: no such file or directory, lstat '/vercel/path0/__pycache__/six.cpython-312.pyc']
```

**Solution Required:**
1. Go to Vercel Dashboard → Total-Search project
2. Settings → General → scroll to "Build Cache"
3. Click "Clear Build Cache"
4. Then Deployments → Redeploy latest

The buildCommand in vercel.json was added but cache restore happens BEFORE build command runs.

---

## Recent Changes (This Session)

### 1. Proxy Score Calculation - DEMAND FOCUSED (Completed)
**Files Changed:**
- `src/calculators/proxy_scores.py`

**TikTok Changes:**
- Now uses hashtag VIEW COUNTS as primary demand signal (not engagement)
- Formula: `monthly_views = hashtag_views / 12 months * 0.01 + video_count`
- Trend based on posting FREQUENCY, not engagement

**Instagram Changes:**
- Now uses POST COUNTS as primary demand signal (not engagement)
- Formula: `daily_posts * 100 + post_count * 0.01`
- Trend based on posting FREQUENCY, not engagement

### 2. YouTube via Google Trends (Completed)
**New File:** `src/clients/google_trends.py`

Uses pytrends library to fetch YouTube-specific search trends:
- No API key required (uses public Google Trends data)
- Converts relative trends index (0-100) to estimated search volume
- Added to requirements.txt: `pytrends>=4.9.0`

**Pipeline Updated:** `src/pipeline/keyword_pipeline.py`
- YouTube now fetches from `self.google_trends.get_youtube_search_volume()` instead of DataForSEO clickstream

### 3. Dashboard Labels Updated
**File:** `templates/brand_dashboard.html`
- Changed "engagement score" → "est. demand/month" for proxy metrics
- Updated no-data reasons for each platform

### 4. Build Configuration
**Files:**
- `.vercelignore` - Added to exclude pycache, venv, tests
- `vercel.json` - Added buildCommand to clean pycache (but cache issue persists)

---

## API Keys Status

| Platform | API | Status | Env Variable |
|----------|-----|--------|--------------|
| Google | DataForSEO | ✅ Working | `DATAFORSEO_LOGIN`, `DATAFORSEO_PASSWORD` |
| YouTube | Google Trends | ✅ No key needed | N/A |
| Amazon | Jungle Scout | ❌ Needs API key | `JUNGLESCOUT_API_KEY` |
| TikTok | Apify | ✅ Working | `APIFY_API_TOKEN` |
| Instagram | Apify | ✅ Working | `APIFY_API_TOKEN` |

**To get Jungle Scout API:**
1. Subscribe to Growth Accelerator ($79/mo) or higher at https://www.junglescout.com/pricing/
2. Go to Account Settings → API → Generate key
3. Add to Vercel env vars as `JUNGLESCOUT_API_KEY`

---

## Database

**Type:** PostgreSQL (Neon serverless)
**Connection:** `DATABASE_URL` env variable in Vercel

**Recent Fix:** PostgreSQL upsert was failing because it used `constraint=` instead of `index_elements=`. Fixed in `src/db/repository.py`:
```python
stmt = stmt.on_conflict_do_update(
    index_elements=["keyword_id", "platform", "collected_date"],  # NOT constraint=
    set_=update_set,
)
```

---

## Key Files Reference

### Core Pipeline
- `src/pipeline/keyword_pipeline.py` - Main orchestrator
- `src/calculators/proxy_scores.py` - TikTok/Instagram demand calculation
- `src/calculators/unified_score.py` - Cross-platform scoring

### API Clients
- `src/clients/dataforseo.py` - Google search volume
- `src/clients/google_trends.py` - YouTube via Google Trends (NEW)
- `src/clients/apify.py` - TikTok/Instagram scraping
- `src/clients/junglescout.py` - Amazon search volume

### API & Frontend
- `api/main.py` - FastAPI endpoints
- `templates/brand_dashboard.html` - Brand Intelligence Dashboard
- `templates/dashboard.html` - Main keyword research dashboard

### Configuration
- `vercel.json` - Vercel deployment config
- `requirements.txt` - Python dependencies
- `src/config.py` - Settings and env vars

---

## Pending Tasks

1. **URGENT:** Clear Vercel build cache and redeploy
2. Help user configure Jungle Scout API key for Amazon data
3. Test all 5 platforms after successful deployment
4. Consider adding rate limiting for Google Trends (to avoid blocks)

---

## Architecture Notes

### Vercel Serverless Constraints
- 60 second max execution time
- Apify timeouts set to 45 seconds
- Hashtag scraping runs in parallel for speed

### Proxy Score Philosophy
User explicitly requested: "I am not interested on engagement, I am interested on measuring the demand of a brand or a product via keyword on all platforms by measuring the search volume or proxies"

Therefore:
- TikTok: Use VIEW COUNTS not likes/comments
- Instagram: Use POST COUNTS not likes/comments
- Display as "est. demand/month" not "engagement score"

---

## Quick Start for Next Session

```bash
cd /Users/moha/Desktop/original-projects/Total-search

# Check git status
git status

# View recent commits
git log --oneline -10

# Key files to review
cat src/calculators/proxy_scores.py
cat src/clients/google_trends.py
cat api/main.py
```

**First Priority:** Clear Vercel build cache and redeploy to test all changes.
