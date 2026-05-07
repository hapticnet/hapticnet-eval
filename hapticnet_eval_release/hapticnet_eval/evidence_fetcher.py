"""
evidence_fetcher.py — On-demand source document fetching via Tavily Extract API.

Ensures source documents are available for the calibrated judge by:
1. Checking the production SourceIndex for existing content files
2. Checking a local SourceIndexCache for previously fetched content
3. Fetching missing sources via Tavily Extract API (markdown format)
4. Writing fetched content to the local cache

Full resource tracking is built in: every fetch attempt is logged with
URL, status, latency, and content size for observability.

Usage:
    from hapticnet_eval.evidence_fetcher import EvidenceFetcher

    fetcher = EvidenceFetcher(cache_dir="~/.hapticnet_source_cache")
    stats = fetcher.ensure_sources(pred_claims, source_index_base="/path/to/SourceIndex")
    # stats = {"fetched": 3, "cached": 12, "pre_existing": 5, "failed": 1, ...}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Content file resolution ──────────────────────────────────────────────────

# Ordered list of content file names to try (production scraper may have
# different outputs depending on which scraper succeeded first).
CONTENT_FILE_NAMES = ("content.md", "content_tavily.md", "content_jina.md")


def resolve_content_path(
    source_uid: str,
    *search_dirs: str,
) -> Optional[str]:
    """Find the best content file for a source UID across multiple directories.

    Searches each directory in order, trying each content filename.
    Returns the first existing path with >100 bytes, or None.

    Args:
        source_uid: The MD5 hash UID of the source URL.
        *search_dirs: Directories to search (e.g. source_index_base, cache_dir).

    Returns:
        Absolute path to the best content file, or None if not found.
    """
    for base_dir in search_dirs:
        if not base_dir:
            continue
        for name in CONTENT_FILE_NAMES:
            path = os.path.join(base_dir, source_uid, "content", name)
            try:
                if os.path.exists(path) and os.path.getsize(path) > 100:
                    return path
            except OSError:
                continue
    return None


def url_to_uid(url: str) -> str:
    """Compute the SourceIndex UID for a URL (MD5 hash)."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


# ── Fetch record (for observability) ────────────────────────────────────────

@dataclass
class FetchRecord:
    """Single source fetch attempt record for observability."""
    url: str
    uid: str
    status: str  # "fetched", "cached", "pre_existing", "failed", "skipped"
    source_dir: str = ""  # which directory the content was found/written in
    content_size: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    tavily_extract_depth: str = ""
    content_file: str = ""

    def to_dict(self) -> dict:
        d = {
            "url": self.url,
            "uid": self.uid,
            "status": self.status,
        }
        if self.source_dir:
            d["source_dir"] = self.source_dir
        if self.content_size:
            d["content_size"] = self.content_size
        if self.elapsed_s:
            d["elapsed_s"] = round(self.elapsed_s, 2)
        if self.error:
            d["error"] = self.error
        if self.tavily_extract_depth:
            d["tavily_extract_depth"] = self.tavily_extract_depth
        if self.content_file:
            d["content_file"] = self.content_file
        return d


# ── Evidence Fetcher ────────────────────────────────────────────────────────

@dataclass
class EvidenceFetcher:
    """Fetches missing source documents via Tavily Extract API.

    Attributes:
        cache_dir: Local directory for cached fetched content.
        tavily_api_key: API key (defaults to TAVILY_API_KEY env var).
        max_batch_size: Max URLs per Tavily Extract call (API max = 20).
        initial_backoff_s: Initial backoff on API error (doubles each retry).
        max_retries: Max retries per batch on API errors.
    """
    cache_dir: str = ""
    tavily_api_key: str = ""
    max_batch_size: int = 20
    initial_backoff_s: float = 1.0
    max_retries: int = 3
    _records: List[FetchRecord] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.path.expanduser("~/.hapticnet_source_cache")
        self.cache_dir = os.path.expanduser(self.cache_dir)
        if not self.tavily_api_key:
            self.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    @property
    def records(self) -> List[FetchRecord]:
        return list(self._records)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate fetch statistics."""
        statuses = {}
        for r in self._records:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        total_bytes = sum(r.content_size for r in self._records)
        total_time = sum(r.elapsed_s for r in self._records)
        return {
            "total_urls": len(self._records),
            "by_status": statuses,
            "total_content_bytes": total_bytes,
            "total_fetch_time_s": round(total_time, 2),
            "cache_dir": self.cache_dir,
            "records": [r.to_dict() for r in self._records],
        }

    def ensure_sources(
        self,
        pred_claims,
        source_index_base: str = "",
    ) -> Dict[str, Any]:
        """Ensure all source documents are available for evaluation.

        Scans pred_claims for source URLs, checks existing content,
        and fetches missing sources via Tavily Extract.

        Args:
            pred_claims: List of CanonicalClaim objects.
            source_index_base: Production SourceIndex directory (checked first).

        Returns:
            Summary dict with counts and per-URL records.
        """
        self._records = []

        # 1. Collect unique (url, uid) pairs from claim provenance
        url_uid_pairs: Dict[str, str] = {}  # url → uid
        for claim in pred_claims:
            for prov in claim.provenance:
                url = prov.source_url
                uid = prov.source_uid
                if url and uid:
                    url_uid_pairs[url] = uid
                elif url and not uid:
                    uid = url_to_uid(url)
                    url_uid_pairs[url] = uid

        if not url_uid_pairs:
            logger.info("EvidenceFetcher: No source URLs found in claims.")
            return self.summary()

        logger.info("EvidenceFetcher: Found %d unique source URLs.", len(url_uid_pairs))

        # 2. Check which UIDs already have content
        urls_to_fetch: List[Tuple[str, str]] = []  # (url, uid)

        for url, uid in url_uid_pairs.items():
            content_path = resolve_content_path(uid, source_index_base, self.cache_dir)
            if content_path:
                size = os.path.getsize(content_path)
                source_dir = "source_index" if source_index_base and content_path.startswith(source_index_base) else "cache"
                self._records.append(FetchRecord(
                    url=url, uid=uid,
                    status="pre_existing" if source_dir == "source_index" else "cached",
                    source_dir=source_dir,
                    content_size=size,
                    content_file=os.path.basename(content_path),
                ))
            else:
                urls_to_fetch.append((url, uid))

        pre_existing = sum(1 for r in self._records if r.status == "pre_existing")
        cached = sum(1 for r in self._records if r.status == "cached")
        logger.info(
            "EvidenceFetcher: %d pre-existing, %d cached, %d need fetching.",
            pre_existing, cached, len(urls_to_fetch),
        )

        # 3. Fetch missing sources via Tavily Extract
        if urls_to_fetch and self.tavily_api_key:
            self._fetch_batch(urls_to_fetch)
        elif urls_to_fetch and not self.tavily_api_key:
            logger.warning(
                "EvidenceFetcher: %d sources need fetching but TAVILY_API_KEY is not set!",
                len(urls_to_fetch),
            )
            for url, uid in urls_to_fetch:
                self._records.append(FetchRecord(
                    url=url, uid=uid,
                    status="skipped",
                    error="TAVILY_API_KEY not set",
                ))

        return self.summary()

    def _fetch_batch(self, url_uid_pairs: List[Tuple[str, str]]) -> None:
        """Fetch a list of URLs via Tavily Extract API in batches."""
        try:
            import requests as _requests
        except ImportError:
            logger.error("EvidenceFetcher: 'requests' library not installed.")
            for url, uid in url_uid_pairs:
                self._records.append(FetchRecord(
                    url=url, uid=uid, status="failed",
                    error="requests library not installed",
                ))
            return

        # Process in batches of max_batch_size
        for batch_start in range(0, len(url_uid_pairs), self.max_batch_size):
            batch = url_uid_pairs[batch_start:batch_start + self.max_batch_size]
            batch_urls = [url for url, _ in batch]
            batch_uid_map = {url: uid for url, uid in batch}

            logger.info(
                "EvidenceFetcher: Fetching batch %d-%d of %d URLs...",
                batch_start + 1, batch_start + len(batch), len(url_uid_pairs),
            )

            # Try basic first, fall back to advanced on failure
            for extract_depth in ("basic", "advanced"):
                t0 = time.monotonic()
                success = self._call_tavily_extract(
                    _requests, batch_urls, batch_uid_map, extract_depth,
                )
                elapsed = time.monotonic() - t0

                if success:
                    logger.info(
                        "EvidenceFetcher: Batch fetched (%s) in %.1fs.",
                        extract_depth, elapsed,
                    )
                    break
                elif extract_depth == "basic":
                    logger.warning(
                        "EvidenceFetcher: Basic extract failed, retrying with advanced..."
                    )
                else:
                    logger.error("EvidenceFetcher: Advanced extract also failed.")

    def _call_tavily_extract(
        self,
        _requests,
        urls: List[str],
        uid_map: Dict[str, str],
        extract_depth: str,
    ) -> bool:
        """Make a single Tavily Extract API call with exponential backoff.

        Returns True if the API call succeeded (even if some URLs failed).
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_api_key}",
        }
        payload = {
            "urls": urls,
            "format": "markdown",
            "extract_depth": extract_depth,
        }

        backoff = self.initial_backoff_s
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                resp = _requests.post(
                    "https://api.tavily.com/extract",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                elapsed = time.monotonic() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    self._process_extract_response(data, uid_map, extract_depth, elapsed)
                    return True
                elif resp.status_code == 429:
                    # Rate limited — back off
                    logger.warning(
                        "EvidenceFetcher: Rate limited (429), backing off %.1fs...",
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logger.error(
                        "EvidenceFetcher: HTTP %d: %s",
                        resp.status_code, resp.text[:300],
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    # Mark all as failed
                    for url in urls:
                        uid = uid_map[url]
                        self._records.append(FetchRecord(
                            url=url, uid=uid, status="failed",
                            error=f"HTTP {resp.status_code}",
                            elapsed_s=elapsed,
                            tavily_extract_depth=extract_depth,
                        ))
                    return False

            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error("EvidenceFetcher: Request error: %s", e)
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                for url in urls:
                    uid = uid_map[url]
                    self._records.append(FetchRecord(
                        url=url, uid=uid, status="failed",
                        error=str(e), elapsed_s=elapsed,
                        tavily_extract_depth=extract_depth,
                    ))
                return False

        return False

    def _process_extract_response(
        self,
        data: dict,
        uid_map: Dict[str, str],
        extract_depth: str,
        elapsed: float,
    ) -> None:
        """Process Tavily Extract API response and write content files."""
        results = data.get("results", [])
        failed_results = data.get("failed_results", [])

        # Track which URLs we got results for
        urls_with_results = set()

        for result in results:
            url = result.get("url", "")
            raw_content = result.get("raw_content", "")

            if url not in uid_map:
                continue

            uid = uid_map[url]
            urls_with_results.add(url)

            if raw_content and len(raw_content) > 100:
                # Write to cache
                content_dir = os.path.join(self.cache_dir, uid, "content")
                os.makedirs(content_dir, exist_ok=True)
                content_path = os.path.join(content_dir, "content.md")

                with open(content_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)

                # Also write metadata
                meta_path = os.path.join(content_dir, "fetch_metadata.json")
                with open(meta_path, "w") as f:
                    json.dump({
                        "url": url,
                        "uid": uid,
                        "fetch_method": "tavily_extract",
                        "extract_depth": extract_depth,
                        "content_length": len(raw_content),
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }, f, indent=2)

                self._records.append(FetchRecord(
                    url=url, uid=uid, status="fetched",
                    source_dir="cache",
                    content_size=len(raw_content),
                    elapsed_s=elapsed / max(len(results), 1),
                    tavily_extract_depth=extract_depth,
                    content_file="content.md",
                ))
                logger.debug(
                    "EvidenceFetcher: Fetched %s → %d chars", url[:60], len(raw_content),
                )
            else:
                self._records.append(FetchRecord(
                    url=url, uid=uid, status="failed",
                    error="empty or too short content",
                    elapsed_s=elapsed / max(len(results), 1),
                    tavily_extract_depth=extract_depth,
                ))

        # Handle explicitly failed URLs
        for failed in failed_results:
            url = failed.get("url", "")
            if url in uid_map and url not in urls_with_results:
                uid = uid_map[url]
                urls_with_results.add(url)
                self._records.append(FetchRecord(
                    url=url, uid=uid, status="failed",
                    error=failed.get("error", "unknown"),
                    tavily_extract_depth=extract_depth,
                ))

        # Handle URLs not in results at all
        for url, uid in uid_map.items():
            if url not in urls_with_results:
                self._records.append(FetchRecord(
                    url=url, uid=uid, status="failed",
                    error="not in API response",
                    tavily_extract_depth=extract_depth,
                ))
