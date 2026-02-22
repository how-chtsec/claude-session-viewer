"""
Cost calculation for Claude model token usage.

Dynamically fetches pricing from LiteLLM's model pricing dataset on startup.
Caches locally for offline use. Falls back to hardcoded defaults if both fail.
"""

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

LITELLM_URL = (
    'https://raw.githubusercontent.com/BerriAI/litellm/main/'
    'model_prices_and_context_window.json'
)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache'
)
_CACHE_FILE = os.path.join(_CACHE_DIR, 'litellm_pricing.json')
_CACHE_MAX_AGE = 86400  # re-fetch after 24 hours

# ---------------------------------------------------------------------------
# Hardcoded fallback pricing (used when no network AND no cache)
# ---------------------------------------------------------------------------
_FALLBACK_PRICING = {
    'claude-opus-4-6': {
        'input': 5e-6, 'output': 25e-6,
        'cache_creation': 6.25e-6, 'cache_read': 5e-7,
    },
    'claude-opus-4-5-20250918': {
        'input': 5e-6, 'output': 25e-6,
        'cache_creation': 6.25e-6, 'cache_read': 5e-7,
    },
    'claude-opus-4-20250514': {
        'input': 15e-6, 'output': 75e-6,
        'cache_creation': 18.75e-6, 'cache_read': 1.5e-6,
    },
    'claude-sonnet-4-6': {
        'input': 3e-6, 'output': 15e-6,
        'cache_creation': 3.75e-6, 'cache_read': 3e-7,
    },
    'claude-sonnet-4-5-20250929': {
        'input': 3e-6, 'output': 15e-6,
        'cache_creation': 3.75e-6, 'cache_read': 3e-7,
    },
    'claude-sonnet-4-20250514': {
        'input': 3e-6, 'output': 15e-6,
        'cache_creation': 3.75e-6, 'cache_read': 3e-7,
    },
    'claude-haiku-4-5-20251001': {
        'input': 1e-6, 'output': 5e-6,
        'cache_creation': 1.25e-6, 'cache_read': 1e-7,
    },
    'claude-3-5-sonnet-20241022': {
        'input': 3e-6, 'output': 15e-6,
        'cache_creation': 3.75e-6, 'cache_read': 3e-7,
    },
    'claude-3-5-sonnet-20240620': {
        'input': 3e-6, 'output': 15e-6,
        'cache_creation': 3.75e-6, 'cache_read': 3e-7,
    },
    'claude-3-5-haiku-20241022': {
        'input': 1e-6, 'output': 5e-6,
        'cache_creation': 1.25e-6, 'cache_read': 1e-7,
    },
    'claude-3-opus-20240229': {
        'input': 15e-6, 'output': 75e-6,
        'cache_creation': 18.75e-6, 'cache_read': 1.5e-6,
    },
}

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
_pricing: dict[str, dict] = {}
_initialized = False

# Version suffix: -v1, -v1:0, etc.
_VERSION_SUFFIX_RE = re.compile(r'-v\d+(?::\d+)?$')


# ---------------------------------------------------------------------------
# LiteLLM data handling
# ---------------------------------------------------------------------------

def _litellm_to_internal(entry: dict) -> dict:
    """Convert a LiteLLM pricing entry to our internal format."""
    return {
        'input': entry.get('input_cost_per_token', 0),
        'output': entry.get('output_cost_per_token', 0),
        'cache_creation': entry.get('cache_creation_input_token_cost', 0),
        'cache_read': entry.get('cache_read_input_token_cost', 0),
        'input_above_200k': entry.get('input_cost_per_token_above_200k_tokens', 0),
        'output_above_200k': entry.get('output_cost_per_token_above_200k_tokens', 0),
        'cache_creation_above_200k': entry.get(
            'cache_creation_input_token_cost_above_200k_tokens', 0
        ),
        'cache_read_above_200k': entry.get(
            'cache_read_input_token_cost_above_200k_tokens', 0
        ),
    }


def _normalize_litellm_key(key: str) -> str | None:
    """Convert a LiteLLM key to our internal model name.

    Examples:
        'anthropic.claude-opus-4-6-v1'        → 'claude-opus-4-6'
        'anthropic.claude-haiku-4-5@20251001'  → 'claude-haiku-4-5'
        'azure_ai/claude-opus-4-6'             → 'claude-opus-4-6'

    Returns None for regional variants or non-Claude models.
    """
    # Skip regional variants (us., eu., global., au., apac., etc.)
    if re.match(r'^[a-z]+\.anthropic[./]', key):
        return None

    # Strip provider prefix
    for prefix in ('anthropic.', 'anthropic/', 'azure_ai/', 'bedrock/'):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    if not key.startswith('claude-'):
        return None

    # Strip version suffix (-v1, -v1:0)
    key = _VERSION_SUFFIX_RE.sub('', key)

    # Strip @date suffix (claude-haiku-4-5@20251001)
    if '@' in key:
        key = key.split('@')[0]

    return key


def _extract_pricing_from_litellm(raw_data: dict) -> dict:
    """Extract Claude model pricing from raw LiteLLM JSON."""
    result = {}
    for key, entry in raw_data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get('mode') != 'chat':
            continue
        if 'input_cost_per_token' not in entry:
            continue

        model_name = _normalize_litellm_key(key)
        if not model_name:
            continue

        # Keep first occurrence (canonical entry, not regional duplicate)
        if model_name not in result:
            result[model_name] = _litellm_to_internal(entry)

    return result


# ---------------------------------------------------------------------------
# Cache & fetch
# ---------------------------------------------------------------------------

def _fetch_remote() -> dict | None:
    """Fetch raw LiteLLM pricing JSON from GitHub."""
    try:
        req = urllib.request.Request(
            LITELLM_URL, headers={'User-Agent': 'claude-viewer/1.0'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning('Failed to fetch LiteLLM pricing: %s', e)
        return None


def _save_cache(raw_data: dict):
    """Save raw LiteLLM data to local cache file."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = _CACHE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'fetched_at': time.time(), 'data': raw_data}, f)
        os.replace(tmp, _CACHE_FILE)
    except Exception as e:
        logger.warning('Failed to save pricing cache: %s', e)


def _load_cache() -> tuple[dict | None, float]:
    """Load cached LiteLLM data. Returns (data, fetched_timestamp)."""
    if not os.path.exists(_CACHE_FILE):
        return None, 0
    try:
        with open(_CACHE_FILE, 'r') as f:
            cached = json.load(f)
        return cached.get('data'), cached.get('fetched_at', 0)
    except Exception:
        return None, 0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _init_pricing():
    """Initialize pricing: remote → cache → hardcoded fallback."""
    global _pricing, _initialized
    if _initialized:
        return

    cached_data, fetched_at = _load_cache()
    cache_fresh = (time.time() - fetched_at) < _CACHE_MAX_AGE if fetched_at else False

    if not cache_fresh:
        raw = _fetch_remote()
        if raw:
            _save_cache(raw)
            _pricing = _extract_pricing_from_litellm(raw)
            _initialized = True
            logger.info(
                'Loaded pricing for %d Claude models from LiteLLM (remote)',
                len(_pricing),
            )
            return

    if cached_data:
        _pricing = _extract_pricing_from_litellm(cached_data)
        _initialized = True
        logger.info(
            'Loaded pricing for %d Claude models from cache (age: %s)',
            len(_pricing),
            f'{(time.time() - fetched_at) / 3600:.1f}h' if fetched_at else 'unknown',
        )
        return

    _pricing = dict(_FALLBACK_PRICING)
    _initialized = True
    logger.warning('Using hardcoded fallback pricing (%d models)', len(_pricing))


# ---------------------------------------------------------------------------
# Pricing lookup
# ---------------------------------------------------------------------------

def _find_pricing(model_name: str) -> dict | None:
    """Find pricing for a model name, trying exact then fuzzy match."""
    _init_pricing()

    if not model_name:
        return None

    # Exact match
    if model_name in _pricing:
        return _pricing[model_name]

    lower = model_name.lower()

    # Substring match (model name contains or is contained by a key)
    for key, pricing in _pricing.items():
        if lower in key or key in lower:
            return pricing

    # Family fallback: match the latest model in the same family
    family_defaults = {
        'opus': 'claude-opus-4-6',
        'sonnet': 'claude-sonnet-4-6',
        'haiku': 'claude-haiku-4-5-20251001',
    }
    for family, default_key in family_defaults.items():
        if family in lower and default_key in _pricing:
            return _pricing[default_key]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_cost(model_name: str,
                   input_tokens: int = 0,
                   output_tokens: int = 0,
                   cache_creation_tokens: int = 0,
                   cache_read_tokens: int = 0) -> float:
    """Calculate total USD cost for token usage using base rates only.

    Uses base (non-tiered) rates because token counts are typically pre-aggregated
    from many API calls, each of which individually falls under the 200k threshold.
    Applying tiered pricing to aggregates would overcharge. This matches ccusage
    behavior where cost is calculated per-entry then summed.

    Returns 0.0 if model is unknown.
    """
    pricing = _find_pricing(model_name)
    if not pricing:
        return 0.0

    cost = 0.0
    if input_tokens > 0 and pricing.get('input'):
        cost += input_tokens * pricing['input']
    if output_tokens > 0 and pricing.get('output'):
        cost += output_tokens * pricing['output']
    if cache_creation_tokens > 0 and pricing.get('cache_creation'):
        cost += cache_creation_tokens * pricing['cache_creation']
    if cache_read_tokens > 0 and pricing.get('cache_read'):
        cost += cache_read_tokens * pricing['cache_read']

    return cost


def calculate_session_cost(model_breakdowns: dict) -> float:
    """Calculate cost from per-model token breakdowns (from session summary).

    Each breakdown entry: { 'input': N, 'output': N, 'cache_read': N, 'cache_create': N }
    Uses base rates per model for accurate multi-model session costing.
    """
    total = 0.0
    for model_name, tokens in model_breakdowns.items():
        total += calculate_cost(
            model_name,
            input_tokens=tokens.get('input', 0),
            output_tokens=tokens.get('output', 0),
            cache_creation_tokens=tokens.get('cache_create', 0),
            cache_read_tokens=tokens.get('cache_read', 0),
        )
    return total


def get_model_pricing_display(model_name: str) -> dict | None:
    """Get human-readable pricing info for a model (per 1M tokens)."""
    pricing = _find_pricing(model_name)
    if not pricing:
        return None
    return {
        'input_per_1m': (pricing.get('input') or 0) * 1_000_000,
        'output_per_1m': (pricing.get('output') or 0) * 1_000_000,
        'cache_creation_per_1m': (pricing.get('cache_creation') or 0) * 1_000_000,
        'cache_read_per_1m': (pricing.get('cache_read') or 0) * 1_000_000,
    }


def get_pricing_source() -> str:
    """Return a string describing the current pricing data source."""
    _init_pricing()
    cached_data, fetched_at = _load_cache()
    if cached_data and fetched_at:
        from datetime import datetime
        dt = datetime.fromtimestamp(fetched_at)
        return f'LiteLLM ({dt.strftime("%Y-%m-%d %H:%M")})'
    return 'Hardcoded fallback'
