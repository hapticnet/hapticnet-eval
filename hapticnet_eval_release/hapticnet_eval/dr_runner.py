"""
dr_runner.py — Unified Deep Research API runner for benchmark Track B.

Runs a research query against one of four DR API providers:
  1. Gemini Deep Research (Interactions API)
  2. OpenAI Deep Research (Responses API, o3-deep-research)
  3. Tavily Deep Research (/research endpoint)
  4. Perplexity (sonar-deep-research)

Each run produces:
  - A raw prose report (saved as {method}_report.md)
  - A structured db_entry.json (via ReportAdapter)
  - A run_metadata.json with full resource tracking

Usage:
    from hapticnet_eval.dr_runner import DRRunner

    runner = DRRunner(provider="gemini", guided=True)
    result = runner.run(
        query_material="Memory Foam",
        query_property="thermal_conductivity",
        gt_source_urls=["https://..."],
        output_dir="./runs/memory_foam_tc/gemini_guided",
    )
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# ── Query template ──────────────────────────────────────────────────────────

UNGUIDED_QUERY = (
    "What is the {property} of {material}? "
    "Provide specific numeric values with units, measurement conditions, "
    "and cite sources with verbatim quotes."
)

GUIDED_QUERY = (
    "What is the {property} of {material}? "
    "Provide specific numeric values with units, measurement conditions, "
    "and cite sources with verbatim quotes.\n\n"
    "The following source URLs may contain relevant information:\n"
    "{urls}\n\n"
    "You may also search for additional sources."
)


# ── DR API Provider Implementations ────────────────────────────────────────

@dataclass
class DRResult:
    """Result from a single DR API call."""
    provider: str
    status: str  # "success", "failed", "timeout", "skipped"
    report_text: str = ""
    report_length: int = 0
    elapsed_s: float = 0.0
    cost_estimate_usd: float = 0.0
    sources_found: List[str] = field(default_factory=list)
    citations_found: int = 0
    error: str = ""
    api_metadata: Dict[str, Any] = field(default_factory=dict)
    # Token-level resource tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    grounding_queries: int = 0  # Gemini search grounding calls

    def to_dict(self) -> dict:
        d = {
            "provider": self.provider,
            "status": self.status,
            "report_length": self.report_length,
            "elapsed_s": round(self.elapsed_s, 1),
            "cost_estimate_usd": round(self.cost_estimate_usd, 4),
            "sources_found": self.sources_found,
            "citations_found": self.citations_found,
            "resource_tracking": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
                "cached_tokens": self.cached_tokens,
                "grounding_queries": self.grounding_queries,
            },
        }
        if self.error:
            d["error"] = self.error
        if self.api_metadata:
            d["api_metadata"] = self.api_metadata
        return d


def _extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from markdown/prose text."""
    url_pattern = r'https?://[^\s\)\]\>\"\'\,]+'
    urls = re.findall(url_pattern, text)
    # Clean trailing punctuation
    cleaned = []
    for url in urls:
        url = url.rstrip('.')
        if len(url) > 15:  # Skip very short matches
            cleaned.append(url)
    return list(dict.fromkeys(cleaned))  # Deduplicate preserving order


def run_gemini_dr(query: str, timeout: int = 300) -> DRResult:
    """Run Gemini Deep Research via Interactions API."""
    try:
        from google import genai
    except ImportError:
        return DRResult(provider="gemini", status="failed", error="google-genai not installed")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return DRResult(provider="gemini", status="skipped", error="GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    t0 = time.monotonic()

    try:
        interaction = client.interactions.create(
            input=query,
            agent="deep-research-preview-04-2026",
            background=True,
        )
        logger.info("Gemini DR: Interaction started: %s", interaction.id)

        # Poll until complete
        deadline = time.monotonic() + timeout
        poll_interval = 10
        while time.monotonic() < deadline:
            interaction = client.interactions.get(interaction.id)
            status = interaction.status
            if status == "completed":
                elapsed = time.monotonic() - t0
                report = _extract_gemini_report(interaction)
                sources = _extract_urls_from_text(report)
                usage = _extract_gemini_usage(interaction)
                return DRResult(
                    provider="gemini",
                    status="success",
                    report_text=report,
                    report_length=len(report),
                    elapsed_s=elapsed,
                    cost_estimate_usd=usage.get("cost_estimate_usd", 0.15),
                    sources_found=sources,
                    citations_found=len(sources),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=usage.get("thoughts_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    cached_tokens=usage.get("cached_tokens", 0),
                    grounding_queries=usage.get("grounding_queries", 0),
                    api_metadata={"interaction_id": interaction.id, "usage": usage},
                )
            elif status == "failed":
                elapsed = time.monotonic() - t0
                err = str(getattr(interaction, "error", "unknown"))
                return DRResult(provider="gemini", status="failed", elapsed_s=elapsed, error=err)
            else:
                logger.debug("Gemini DR: status=%s, polling in %ds...", status, poll_interval)
                time.sleep(poll_interval)

        return DRResult(
            provider="gemini", status="timeout",
            elapsed_s=time.monotonic() - t0,
            error=f"Timed out after {timeout}s",
        )

    except Exception as e:
        return DRResult(
            provider="gemini", status="failed",
            elapsed_s=time.monotonic() - t0, error=str(e),
        )


def _extract_gemini_report(interaction) -> str:
    """Extract research report text from Gemini Interactions API response."""
    report_parts = []
    if not interaction.outputs:
        return ""
    for out in interaction.outputs:
        out_type = getattr(out, "type", None)
        if out_type and str(out_type).lower() in ("thought", "thinking"):
            continue
        text = getattr(out, "text", None)
        if text:
            report_parts.append(text)
            continue
        content = getattr(out, "content", None)
        if content:
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    report_parts.append(part_text)
    return "\n".join(report_parts)


def _extract_gemini_usage(interaction) -> Dict[str, Any]:
    """Extract token usage and cost from Gemini Interactions API response.

    The Interactions API exposes usage_metadata on the completed interaction
    object with prompt_token_count, candidates_token_count, total_token_count,
    thoughts_token_count, and cached_content_token_count.
    """
    usage: Dict[str, Any] = {}
    um = getattr(interaction, "usage_metadata", None)
    if um:
        usage["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
        usage["completion_tokens"] = getattr(um, "candidates_token_count", 0) or 0
        usage["total_tokens"] = getattr(um, "total_token_count", 0) or 0
        usage["thoughts_tokens"] = getattr(um, "thoughts_token_count", 0) or 0
        usage["cached_tokens"] = getattr(um, "cached_content_token_count", 0) or 0
        # Estimate cost: Gemini 2.5 Pro pricing for deep research
        # Input: $0.1875/1K tokens, Output: $0.75/1K tokens,
        # Grounding: ~$5/1K queries
        in_cost = usage["prompt_tokens"] * 0.1875 / 1000
        out_cost = usage["completion_tokens"] * 0.75 / 1000
        think_cost = usage["thoughts_tokens"] * 0.1875 / 1000  # billed at input rate
        usage["cost_estimate_usd"] = round(in_cost + out_cost + think_cost, 4)
    else:
        # Fallback: inspect outputs for any usage data
        for out in getattr(interaction, "outputs", []) or []:
            out_um = getattr(out, "usage_metadata", None)
            if out_um:
                usage["prompt_tokens"] = (usage.get("prompt_tokens", 0)
                                          + (getattr(out_um, "prompt_token_count", 0) or 0))
                usage["completion_tokens"] = (usage.get("completion_tokens", 0)
                                              + (getattr(out_um, "candidates_token_count", 0) or 0))
                usage["total_tokens"] = (usage.get("total_tokens", 0)
                                         + (getattr(out_um, "total_token_count", 0) or 0))
        if not usage:
            usage["cost_estimate_usd"] = 0.15  # conservative fallback estimate

    # Count grounding/search queries from tool use outputs
    grounding_queries = 0
    for out in getattr(interaction, "outputs", []) or []:
        if getattr(out, "type", None) and "search" in str(getattr(out, "type", "")).lower():
            grounding_queries += 1
        # Some interactions report tool_calls
        tool_calls = getattr(out, "tool_calls", None) or []
        for tc in tool_calls:
            if "search" in str(getattr(tc, "name", "")).lower():
                grounding_queries += 1
    usage["grounding_queries"] = grounding_queries
    return usage


def run_openai_dr(query: str, timeout: int = 300) -> DRResult:
    """Run OpenAI Deep Research via Responses API."""
    try:
        from openai import OpenAI
    except ImportError:
        return DRResult(provider="openai", status="failed", error="openai not installed")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return DRResult(provider="openai", status="skipped", error="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    t0 = time.monotonic()

    try:
        response = client.responses.create(
            model="o3-deep-research",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials science research agent. Produce a rigorous "
                        "report with numeric values, units, measurement conditions, and "
                        "inline citations."
                    ),
                },
                {"role": "user", "content": query},
            ],
            tools=[{"type": "web_search_preview"}],
            background=True,
        )

        research_id = response.id
        logger.info("OpenAI DR: Response started: %s", research_id)

        # Poll
        deadline = time.monotonic() + timeout
        poll_interval = 10
        while time.monotonic() < deadline:
            check = client.responses.retrieve(research_id)
            status = check.status
            if status == "completed":
                elapsed = time.monotonic() - t0
                report = ""
                for item in (check.output or []):
                    if getattr(item, "type", None) == "message":
                        for content in getattr(item, "content", []) or []:
                            text = getattr(content, "text", None)
                            if text:
                                report += text
                if not report:
                    report = getattr(check, "output_text", "") or ""
                sources = _extract_urls_from_text(report)
                # Extract OpenAI usage metadata
                oai_usage = _extract_openai_usage(check)
                return DRResult(
                    provider="openai",
                    status="success",
                    report_text=report,
                    report_length=len(report),
                    elapsed_s=elapsed,
                    cost_estimate_usd=oai_usage.get("cost_estimate_usd", 3.50),
                    prompt_tokens=oai_usage.get("prompt_tokens", 0),
                    completion_tokens=oai_usage.get("completion_tokens", 0),
                    reasoning_tokens=oai_usage.get("reasoning_tokens", 0),
                    total_tokens=oai_usage.get("total_tokens", 0),
                    api_metadata={"response_id": research_id, "usage": oai_usage},
                    sources_found=sources,
                    citations_found=len(sources),
                )
            elif status in ("failed", "cancelled"):
                elapsed = time.monotonic() - t0
                err = str(getattr(check, "last_error", "unknown"))
                return DRResult(provider="openai", status="failed", elapsed_s=elapsed, error=err)
            else:
                logger.debug("OpenAI DR: status=%s, polling in %ds...", status, poll_interval)
                time.sleep(poll_interval)

        return DRResult(
            provider="openai", status="timeout",
            elapsed_s=time.monotonic() - t0,
            error=f"Timed out after {timeout}s",
        )

    except Exception as e:
        return DRResult(
            provider="openai", status="failed",
            elapsed_s=time.monotonic() - t0, error=str(e),
        )


def _extract_openai_usage(response) -> Dict[str, Any]:
    """Extract token usage and cost from OpenAI Responses API.

    The response object has a .usage field with:
    prompt_tokens, completion_tokens, total_tokens,
    and completion_tokens_details.reasoning_tokens.
    """
    usage: Dict[str, Any] = {}
    resp_usage = getattr(response, "usage", None)
    if resp_usage:
        usage["prompt_tokens"] = getattr(resp_usage, "prompt_tokens", 0) or 0
        usage["completion_tokens"] = getattr(resp_usage, "completion_tokens", 0) or 0
        usage["total_tokens"] = getattr(resp_usage, "total_tokens", 0) or 0
        ctd = getattr(resp_usage, "completion_tokens_details", None)
        reasoning = 0
        if ctd:
            reasoning = getattr(ctd, "reasoning_tokens", 0) or 0
        usage["reasoning_tokens"] = reasoning
        # o3-deep-research pricing: ~$15/1M input, ~$60/1M output
        in_cost = usage["prompt_tokens"] * 15.0 / 1_000_000
        out_cost = usage["completion_tokens"] * 60.0 / 1_000_000
        usage["cost_estimate_usd"] = round(in_cost + out_cost, 4)
    else:
        usage["cost_estimate_usd"] = 3.50  # fallback
    return usage


def run_tavily_dr(query: str, timeout: int = 300) -> DRResult:
    """Run Tavily Deep Research via /research endpoint.

    Ref: https://docs.tavily.com/documentation/api-reference/endpoint/research
    POST body: { input, model, stream, output_schema?, citation_format? }
    Response: { request_id, status, content, sources[], response_time }
    """
    try:
        import requests as _requests
    except ImportError:
        return DRResult(provider="tavily", status="failed", error="requests not installed")

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return DRResult(provider="tavily", status="skipped", error="TAVILY_API_KEY not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    t0 = time.monotonic()

    try:
        # NOTE: Tavily output_schema is extremely restrictive (no anyOf, no nested
        # objects, every field needs 'type'). Instead of fighting it, we let Tavily
        # return a prose report and rely on the ReportAdapter (Gemini constrained
        # decoding with the full HapticMaterialProperty schema) to extract structured
        # data. This is more robust and the adapter handles it well.
        payload = {
            "input": query,
            "model": "pro",
            "stream": False,
            "citation_format": "numbered",
        }
        logger.info("Tavily DR: POST /research (timeout=%ds)...", timeout)
        resp = _requests.post(
            "https://api.tavily.com/research",
            headers=headers, json=payload, timeout=timeout,  # Tavily blocks until done
        )
        if resp.status_code != 200:
            return DRResult(
                provider="tavily", status="failed",
                elapsed_s=time.monotonic() - t0,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            )

        init_data = resp.json()
        request_id = init_data.get("request_id", "")
        init_status = init_data.get("status", "")
        logger.info("Tavily DR: init response status=%s request_id=%s keys=%s",
                     init_status, request_id, list(init_data.keys()))

        if init_status == "completed":
            elapsed = time.monotonic() - t0
            report = _extract_tavily_report(init_data)
            sources = [s.get("url", "") for s in init_data.get("sources", []) if s.get("url")]
            return DRResult(
                provider="tavily", status="success",
                report_text=report, report_length=len(report),
                elapsed_s=elapsed, cost_estimate_usd=0.30,
                sources_found=sources, citations_found=len(sources),
            )

        if not request_id:
            return DRResult(
                provider="tavily", status="failed",
                elapsed_s=time.monotonic() - t0,
                error=f"No request_id in response. keys={list(init_data.keys())} body={str(init_data)[:500]}",
            )

        # Poll via GET /research/{request_id}
        deadline = time.monotonic() + timeout
        poll_interval = 10
        poll_count = 0
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            poll_count += 1
            poll_resp = _requests.get(
                f"https://api.tavily.com/research/{request_id}",
                headers=headers, timeout=30,
            )
            if poll_resp.status_code != 200:
                logger.debug("Tavily DR: poll #%d HTTP %d", poll_count, poll_resp.status_code)
                continue
            data = poll_resp.json()
            status = data.get("status", "unknown")
            logger.debug("Tavily DR: poll #%d status=%s keys=%s", poll_count, status, list(data.keys()))
            if status == "completed":
                elapsed = time.monotonic() - t0
                report = _extract_tavily_report(data)
                sources = [s.get("url", "") for s in data.get("sources", []) if s.get("url")]
                resp_time = data.get("response_time", 0)
                return DRResult(
                    provider="tavily", status="success",
                    report_text=report, report_length=len(report),
                    elapsed_s=elapsed, cost_estimate_usd=0.30,
                    sources_found=sources, citations_found=len(sources),
                    api_metadata={"request_id": request_id, "response_time": resp_time,
                                  "poll_count": poll_count},
                )
            elif status == "failed":
                return DRResult(
                    provider="tavily", status="failed",
                    elapsed_s=time.monotonic() - t0,
                    error=f"Tavily task failed: {data.get('error', str(data)[:300])}",
                )

        return DRResult(
            provider="tavily", status="timeout",
            elapsed_s=time.monotonic() - t0,
            error=f"Timed out after {timeout}s ({poll_count} polls)",
        )

    except Exception as e:
        return DRResult(
            provider="tavily", status="failed",
            elapsed_s=time.monotonic() - t0, error=str(e),
        )


def _extract_tavily_report(data: dict) -> str:
    """Extract report text from Tavily /research response.

    The API returns the report in the 'content' field (string or dict if schema used).
    """
    # 'content' is the canonical field per Tavily docs
    content = data.get("content")
    if content:
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Structured output — convert to JSON string for downstream processing
            return json.dumps(content, indent=2, ensure_ascii=False)

    # Fallback: try other possible field names
    for key in ("output", "report", "answer", "result"):
        val = data.get(key)
        if val and isinstance(val, str):
            return val
    return ""


def run_perplexity_dr(query: str) -> DRResult:
    """Run Perplexity sonar-deep-research."""
    try:
        import requests as _requests
    except ImportError:
        return DRResult(provider="perplexity", status="failed", error="requests not installed")

    api_key = os.environ.get("PPLX_API_KEY", "")
    if not api_key:
        return DRResult(provider="perplexity", status="skipped", error="PPLX_API_KEY not set")

    t0 = time.monotonic()

    try:
        # Build the extraction schema for structured output hint
        from .report_adapter import ReportAdapter
        extraction_schema = ReportAdapter._EXTRACTION_SCHEMA

        payload = {
            "model": "sonar-deep-research",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a materials science research agent. Provide specific "
                        "numeric values with units, measurement conditions, and citations. "
                        "Return your findings as a JSON object matching the requested schema."
                    ),
                },
                {"role": "user", "content": query},
            ],
            # Attempt structured output — sonar-deep-research may not support it,
            # in which case Perplexity will ignore the field and return prose.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "hapticnet_extraction",
                    "schema": extraction_schema,
                },
            },
        }
        resp = _requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload, timeout=300,
        )
        elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            return DRResult(
                provider="perplexity", status="failed",
                elapsed_s=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
            )

        data = resp.json()
        raw_report = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        usage = data.get("usage", {})

        # Strip <think>...</think> blocks
        report = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', raw_report, flags=re.DOTALL).strip()

        # Extract token-level usage from Perplexity response
        prompt_tok = usage.get("prompt_tokens", 0) or 0
        completion_tok = usage.get("completion_tokens", 0) or 0
        reasoning_tok = usage.get("reasoning_tokens", 0) or 0
        total_tok = prompt_tok + completion_tok + reasoning_tok
        # Perplexity sonar-deep-research pricing: ~$2/1M input, ~$8/1M output
        pplx_cost = (prompt_tok * 2.0 + completion_tok * 8.0) / 1_000_000

        return DRResult(
            provider="perplexity",
            status="success",
            report_text=report,
            report_length=len(report),
            elapsed_s=elapsed,
            cost_estimate_usd=round(pplx_cost, 4) if pplx_cost > 0 else 0.10,
            sources_found=citations if isinstance(citations, list) else [],
            citations_found=len(citations) if isinstance(citations, list) else 0,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            reasoning_tokens=reasoning_tok,
            total_tokens=total_tok,
            api_metadata={"usage": usage},
        )

    except Exception as e:
        return DRResult(
            provider="perplexity", status="failed",
            elapsed_s=time.monotonic() - t0, error=str(e),
        )


# ── Provider registry ──────────────────────────────────────────────────────

DR_PROVIDERS = {
    "gemini": run_gemini_dr,
    "openai": run_openai_dr,
    "tavily": run_tavily_dr,
    "perplexity": run_perplexity_dr,
}


# ── Unified DR Runner ──────────────────────────────────────────────────────

@dataclass
class DRRunner:
    """Unified Deep Research runner with full pipeline tracking.

    Attributes:
        provider: DR API provider name.
        guided: If True, include GT source URLs in the query as guidance.
        timeout: Max seconds to wait for async providers.
        adapter_model: LLM model for ReportAdapter structuring.
        fetch_sources: If True, fetch source documents via Tavily Extract and run grounding.
        source_cache_dir: Shared cache directory for fetched source documents.
        source_index_base: Production SourceIndex directory (for intersection tracking).
    """
    provider: str = "gemini"
    guided: bool = False
    timeout: int = 300
    adapter_model: str = "gemini-3.1-flash-lite-preview"
    fetch_sources: bool = False
    source_cache_dir: str = ""
    source_index_base: str = ""

    def run(
        self,
        query_material: str,
        query_property: str,
        gt_source_urls: Optional[List[str]] = None,
        output_dir: str = "",
        material_family: str = "",
    ) -> Dict[str, Any]:
        """Run the full DR pipeline: API call → Report → Adapter → db_entry.

        Args:
            query_material: Material name.
            query_property: Property name.
            gt_source_urls: GT source URLs (used for guided variant).
            output_dir: Directory to save outputs.
            material_family: Material family for metadata.

        Returns:
            Dict with: db_entry, dr_result, adapter_resources, paths.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_label = f"dr_{self.provider}_{'guided' if self.guided else 'unguided'}"

        # Build query
        property_display = query_property.replace("_", " ").title()
        if self.guided and gt_source_urls:
            query = GUIDED_QUERY.format(
                material=query_material,
                property=property_display,
                urls="\n".join(f"- {u}" for u in gt_source_urls),
            )
        else:
            query = UNGUIDED_QUERY.format(
                material=query_material,
                property=property_display,
            )

        logger.info(
            "DRRunner: Running %s on %s / %s (guided=%s)",
            self.provider, query_material, query_property, self.guided,
        )

        # 1. Call DR API
        runner_fn = DR_PROVIDERS.get(self.provider)
        if not runner_fn:
            raise ValueError(f"Unknown DR provider: {self.provider}")

        if self.provider in ("gemini", "openai", "tavily"):
            dr_result = runner_fn(query, timeout=self.timeout)
        else:
            dr_result = runner_fn(query)

        logger.info(
            "DRRunner: %s returned status=%s, %d chars in %.1fs",
            self.provider, dr_result.status,
            dr_result.report_length, dr_result.elapsed_s,
        )

        # 2. Adapt report to structured db_entry
        db_entry = None
        adapter_resources = {}
        if dr_result.status == "success" and dr_result.report_text:
            from .report_adapter import ReportAdapter
            adapter = ReportAdapter(model=self.adapter_model)
            db_entry = adapter.adapt(
                raw_report=dr_result.report_text,
                query_material=query_material,
                query_property=query_property,
                source_urls=dr_result.sources_found,
                material_family=material_family,
            )
            adapter_resources = adapter.resource_summary()
            logger.info(
                "DRRunner: Adapter extracted %d values from report.",
                len(db_entry.get("value_condition_mapping", [])),
            )

            # Validate adapted db_entry against GTFile pydantic schema
            try:
                from .schemas import GTFile
                GTFile(**db_entry)
                logger.info("DRRunner: db_entry passed GTFile validation.")
            except (ValidationError, Exception) as ve:
                logger.warning(
                    "DRRunner: db_entry failed GTFile validation (non-fatal): %s",
                    str(ve)[:500],
                )

        elif dr_result.status != "success":
            # Create empty db_entry for failed runs (matching production schema)
            empty_cat = {"values": [], "units": None, "count": 0}
            db_entry = {
                "material_family": material_family,
                "material_class": query_material,
                "haptic_property": query_property,
                "value_condition_mapping": [],
                "citations": [],
                "sources": [],
                "labeler_note": None,
                "property_stats": None,
                "value_list": {
                    "all_values": empty_cat.copy(),
                    "grounded_values": empty_cat.copy(),
                    "ungrounded_values": empty_cat.copy(),
                },
                "no_answer": True,
                "no_answer_reason": f"DR API returned status: {dr_result.status} — {dr_result.error}",
            }

        # 2b. Evidence fetching + grounding (for API DR methods)
        evidence_fetch_stats = {}
        source_intersection = {}
        grounding_stats = {}
        if self.fetch_sources and db_entry and not db_entry.get("no_answer"):
            source_urls = dr_result.sources_found or []
            if source_urls:
                from .evidence_fetcher import EvidenceFetcher, url_to_uid, resolve_content_path
                import os as _os

                # Resolve cache dir
                cache_dir = self.source_cache_dir or _os.path.expanduser("~/.hapticnet_source_cache")
                si_base = self.source_index_base or _os.environ.get(
                    "SOURCE_INDEX_BASE", "/mnt/cgm-atlas/ofri/HapticNet/SourceIndex"
                )

                # ── Source intersection tracking ──
                # Compare DR source URLs against existing SourceIndex
                dr_uids = {url_to_uid(url): url for url in source_urls}
                existing_in_si = 0
                existing_uids = []
                for uid in dr_uids:
                    if resolve_content_path(uid, si_base):
                        existing_in_si += 1
                        existing_uids.append(uid)
                source_intersection = {
                    "dr_source_count": len(source_urls),
                    "unique_uids": len(dr_uids),
                    "existing_in_source_index": existing_in_si,
                    "intersection_ratio": round(existing_in_si / max(len(dr_uids), 1), 4),
                    "existing_uids": existing_uids,
                }
                logger.info(
                    "DRRunner: Source intersection: %d/%d UIDs already in SourceIndex (%.1f%%)",
                    existing_in_si, len(dr_uids),
                    100 * existing_in_si / max(len(dr_uids), 1),
                )

                # ── Evidence fetching via Tavily Extract ──
                # Always fetch even if UID exists in SourceIndex (per user req)
                fetcher = EvidenceFetcher(cache_dir=cache_dir)
                url_uid_pairs = [(url, url_to_uid(url)) for url in source_urls]

                logger.info(
                    "DRRunner: Fetching %d source documents via Tavily Extract...",
                    len(url_uid_pairs),
                )
                t_fetch = time.time()
                fetcher._records = []
                if fetcher.tavily_api_key:
                    fetcher._fetch_batch(url_uid_pairs)
                else:
                    logger.warning("DRRunner: TAVILY_API_KEY not set — skipping evidence fetch")
                fetch_elapsed = time.time() - t_fetch
                evidence_fetch_stats = fetcher.summary()
                evidence_fetch_stats["fetch_elapsed_s"] = round(fetch_elapsed, 2)

                fetched_count = sum(1 for r in fetcher.records if r.status == "fetched")
                logger.info(
                    "DRRunner: Fetched %d/%d sources in %.1fs",
                    fetched_count, len(url_uid_pairs), fetch_elapsed,
                )

                # ── Grounding ──
                # Build source_texts from fetched + cached content
                source_texts = {}
                for uid in dr_uids:
                    # Check cache first (just fetched), then SourceIndex
                    content_path = resolve_content_path(uid, cache_dir, si_base)
                    if content_path:
                        try:
                            with open(content_path, "r", encoding="utf-8", errors="replace") as f:
                                text = f.read()
                            if len(text) > 100:
                                source_texts[uid] = text
                        except Exception as e:
                            logger.debug("DRRunner: Failed to read %s: %s", content_path, e)

                if source_texts:
                    logger.info(
                        "DRRunner: Running grounding on %d source documents...",
                        len(source_texts),
                    )
                    try:
                        import sys as _sys
                        _hapticnet_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                        _agents_dir = _os.path.join(_hapticnet_root, "research_agent_benchmark_suite", "agents")
                        if _agents_dir not in _sys.path:
                            _sys.path.insert(0, _agents_dir)

                        from shared.grounding_postprocess import run_grounding
                        t_ground = time.time()
                        db_entry = run_grounding(
                            db_entry,
                            source_texts=source_texts,
                            skip_phase2=True,  # Phase-1 only (fast, no LLM cost)
                        )
                        ground_elapsed = time.time() - t_ground

                        grounded = sum(
                            1 for v in db_entry.get("value_condition_mapping", [])
                            if v.get("is_grounded")
                        )
                        total_vals = len(db_entry.get("value_condition_mapping", []))
                        grounding_stats = {
                            "grounded_values": grounded,
                            "total_values": total_vals,
                            "grounding_ratio": round(grounded / max(total_vals, 1), 4),
                            "source_texts_loaded": len(source_texts),
                            "grounding_elapsed_s": round(ground_elapsed, 2),
                            "phase2_skipped": True,
                        }
                        logger.info(
                            "DRRunner: Grounding complete: %d/%d values grounded in %.1fs",
                            grounded, total_vals, ground_elapsed,
                        )
                    except Exception as e:
                        logger.error("DRRunner: Grounding failed (non-fatal): %s", e)
                        grounding_stats = {"error": str(e)}
                else:
                    logger.warning("DRRunner: No source texts available for grounding")
                    grounding_stats = {"error": "no_source_texts_available", "source_texts_loaded": 0}

        # 3. Save outputs
        paths = {}
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save raw report
            report_path = out_dir / f"{method_label}_report_{timestamp}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# {query_material} — {property_display}\n\n")
                f.write(f"**Provider**: {self.provider}\n")
                f.write(f"**Guided**: {self.guided}\n")
                f.write(f"**Timestamp**: {timestamp}\n\n")
                f.write("---\n\n")
                f.write(dr_result.report_text or "(no report generated)")
            paths["report"] = str(report_path)

            # Save db_entry
            if db_entry:
                db_path = out_dir / f"db_entry_{timestamp}_{method_label}.json"
                with open(db_path, "w", encoding="utf-8") as f:
                    json.dump(db_entry, f, indent=2, ensure_ascii=False, default=str)
                paths["db_entry"] = str(db_path)

            # Save run_metadata
            run_meta = {
                "method": method_label,
                "provider": self.provider,
                "guided": self.guided,
                "timestamp": timestamp,
                "query_material": query_material,
                "query_property": query_property,
                "query_text": query[:500],
                "dr_result": dr_result.to_dict(),
                "adapter_resources": adapter_resources,
                "evidence_fetch": evidence_fetch_stats,
                "source_intersection": source_intersection,
                "grounding": grounding_stats,
                "gt_source_urls": gt_source_urls or [],
                "output_dir": str(output_dir),
            }
            meta_path = out_dir / f"db_entry_{timestamp}_{method_label}_run_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)
            paths["run_metadata"] = str(meta_path)

        return {
            "db_entry": db_entry,
            "dr_result": dr_result.to_dict(),
            "adapter_resources": adapter_resources,
            "evidence_fetch": evidence_fetch_stats,
            "source_intersection": source_intersection,
            "grounding": grounding_stats,
            "paths": paths,
            "method": method_label,
            "timestamp": timestamp,
        }
