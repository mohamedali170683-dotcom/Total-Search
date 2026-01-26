"""Demo trend data generators for TikTok, Instagram, and Amazon.

Generates realistic simulated time-series data (0-100 scale) based on
the actual Google Web trend pattern. Each platform applies characteristic
offsets, lags, and noise to produce believable divergence patterns.

These are clearly labeled as demo/simulated data in the API response
to show what the prototype would look like with a KeywordTool.io license.
"""

import hashlib
import math
import random
from datetime import datetime, timedelta


def _seed_from_keyword(keyword: str, platform: str) -> int:
    """Deterministic seed so the same keyword always produces the same demo data."""
    return int(hashlib.md5(f"{keyword}:{platform}".encode()).hexdigest()[:8], 16)


def generate_demo_trend(
    google_web_series: list[dict],
    keyword: str,
    platform: str,
) -> list[dict]:
    """
    Generate a realistic demo trend line for a platform based on Google Web data.

    The generated data follows the Google Web pattern but with platform-specific
    characteristics:

    - Amazon: Lags Google by 1-2 weeks (awareness → purchase intent delay),
      amplifies peaks (purchase spikes), slightly lower baseline.
    - TikTok: Leads Google by 1-3 weeks (viral content drives search),
      more volatile/spiky, higher noise.
    - Instagram: Loosely correlated with Google, smoother curve,
      lower amplitude, independent seasonal bumps.

    Args:
        google_web_series: The real Google Web interest_over_time data.
        keyword: The search keyword (used for deterministic seeding).
        platform: One of "amazon", "tiktok", "instagram".

    Returns:
        List of dicts with "date" and platform key (0-100 scale).
    """
    if not google_web_series:
        return []

    rng = random.Random(_seed_from_keyword(keyword, platform))

    # Extract values from Google Web data
    dates = [entry["date"] for entry in google_web_series]
    web_key = next((k for k in google_web_series[0] if k != "date"), keyword)
    base_values = [float(entry.get(web_key, entry.get(keyword, 0))) for entry in google_web_series]

    n = len(base_values)
    if n == 0:
        return []

    # Platform-specific transformation parameters
    configs = {
        "amazon": {
            "lag": rng.randint(1, 3),         # Amazon lags Google (awareness → purchase)
            "amplitude": rng.uniform(0.8, 1.2),
            "noise": 0.08,
            "baseline_shift": rng.uniform(-10, -3),
            "peak_amplify": 1.3,              # Purchase spikes amplified
            "smoothing": 0.3,
        },
        "tiktok": {
            "lag": rng.randint(-3, -1),       # TikTok leads Google (viral → search)
            "amplitude": rng.uniform(0.9, 1.4),
            "noise": 0.18,                    # More volatile
            "baseline_shift": rng.uniform(-15, 5),
            "peak_amplify": 1.5,              # Viral spikes
            "smoothing": 0.15,
        },
        "instagram": {
            "lag": rng.randint(-1, 1),        # Loosely aligned
            "amplitude": rng.uniform(0.6, 0.9),
            "noise": 0.12,
            "baseline_shift": rng.uniform(-8, 8),
            "peak_amplify": 0.9,              # Smoother, less spiky
            "smoothing": 0.45,                # More smoothing
        },
    }

    config = configs.get(platform, configs["instagram"])
    lag = config["lag"]
    amp = config["amplitude"]
    noise_level = config["noise"]
    shift = config["baseline_shift"]
    peak_amp = config["peak_amplify"]
    smooth = config["smoothing"]

    # Apply lag by shifting the base values
    shifted = []
    for i in range(n):
        src_idx = i - lag
        if 0 <= src_idx < n:
            shifted.append(base_values[src_idx])
        elif src_idx < 0:
            shifted.append(base_values[0])
        else:
            shifted.append(base_values[-1])

    # Apply amplitude scaling, peak amplification, noise, and baseline shift
    raw_values = []
    for i, val in enumerate(shifted):
        # Amplify peaks (values above 70)
        if val > 70:
            val = val * peak_amp
        else:
            val = val * amp

        # Add noise
        noise = rng.gauss(0, noise_level * 30)
        val += noise + shift

        # Add platform-specific seasonal variation
        week_of_year = i % 52
        if platform == "tiktok":
            # TikTok has mini viral spikes
            if rng.random() < 0.08:
                val += rng.uniform(15, 35)
        elif platform == "instagram":
            # Instagram has gentle seasonal bumps
            val += 5 * math.sin(2 * math.pi * week_of_year / 26)
        elif platform == "amazon":
            # Amazon has Q4 holiday boost
            if week_of_year >= 44 or week_of_year <= 2:
                val *= 1.2

        raw_values.append(val)

    # Apply exponential smoothing
    smoothed = [raw_values[0]]
    for i in range(1, n):
        smoothed.append(smooth * raw_values[i] + (1 - smooth) * smoothed[-1])

    # Normalize to 0-100
    max_val = max(smoothed) if smoothed else 1
    min_val = min(smoothed) if smoothed else 0
    val_range = max_val - min_val if max_val != min_val else 1

    result = []
    for i, val in enumerate(smoothed):
        normalized = ((val - min_val) / val_range) * 100
        normalized = max(0, min(100, round(normalized)))
        result.append({
            "date": dates[i],
            platform: normalized,
        })

    return result
