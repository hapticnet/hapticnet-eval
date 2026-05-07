"""
report_adapter.py — Converts unstructured Deep Research reports to structured db_entry format.

Transforms prose research reports from any DR API (Gemini, OpenAI, Tavily, Perplexity)
into the structured db_entry.json format that the HapticNetEval evaluator suite expects.

The adapter uses a cheap LLM (Gemini Flash-Lite by default) to extract:
  - Numeric values with units and measurement conditions
  - Source URLs and citations
  - Material/property metadata

Full resource tracking is built in: LLM calls, token usage, latency, and costs
are logged for observability.

Usage:
    from hapticnet_eval.report_adapter import ReportAdapter

    adapter = ReportAdapter()
    db_entry = adapter.adapt(
        raw_report="The thermal conductivity of Memory Foam is 0.03 W/(m·K)...",
        query_material="Memory Foam",
        query_property="thermal_conductivity",
        source_urls=["https://example.com/source1"],
    )
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Extraction prompt ──────────────────────────────────────────────────────
# Aligned with the production pipeline prompt in simple_llm_extraction.py.
# The adapter LLM receives this prompt + constrained decoding via _EXTRACTION_SCHEMA.

EXTRACTION_PROMPT = '''You are a materials science expert. Your task is to extract structured data about haptic/tactile material properties from the provided research report.

**Query:** What is the {property} of {material}?

**Source UIDs and URLs** (use these for the "sources" field and citation source_uid/source_url):
{source_info}

**Instructions:**
1. Read the report carefully and identify ALL reported values for the queried property.
2. For each value found, extract:
   - The numeric value (as a scalar {{value: N}}, range {{min: N, max: N}}, mean±std {{mean: N, std: N}}, or data_points list)
   - The units
   - Material subclass (specific variant/grade, or "unspecified")
   - Measurement conditions as label/val pairs (temperature, load, speed, method, etc.)
   - Material specifications as label/val pairs (treatment, coating, grade, etc.)
   - Object class (physical form: sheet, wire, bulk, film, etc.)
3. Extract verbatim citation snippets from the report that support each value.
4. For EACH citation, set source_uid and source_url to the UID and URL of the source document from which the snippet was extracted. Match the URL in the report text to the Source UIDs list above.
5. List all sources referenced in the report.
6. Set normalized_value = same as value, normalized_units = same as units.
7. If no relevant data is found, set no_answer: true with a reason.
8. Leave labeler_note empty — it is reserved for human labeler validation.

**CRITICAL:** Only extract values that are actually present in the text. Do NOT hallucinate or infer values.

**Report:**
{report}

Return a JSON object matching the provided schema exactly.'''


# ── Resource tracking ──────────────────────────────────────────────────────

@dataclass
class AdapterResourceRecord:
    """Tracks resource usage for a single adapter call."""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    success: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


# ── Report Adapter ──────────────────────────────────────────────────────────

@dataclass
class ReportAdapter:
    """Converts unstructured research reports to structured db_entry format.

    Attributes:
        model: LLM model for extraction (default: gemini-2.5-flash-lite).
        provider: LLM provider ('gemini' or 'openai').
        thinking_budget: Token budget for model thinking.
        material_family: Material family for output metadata.
    """
    model: str = "gemini-3.1-flash-lite-preview"
    provider: str = "gemini"
    thinking_budget: int = 1024
    material_family: str = ""
    _resource_records: List[AdapterResourceRecord] = field(default_factory=list, repr=False)

    @property
    def resource_records(self) -> List[AdapterResourceRecord]:
        return list(self._resource_records)

    def resource_summary(self) -> Dict[str, Any]:
        """Aggregate resource usage across all adapter calls."""
        total_tokens = sum(r.total_tokens for r in self._resource_records)
        total_cost = sum(r.cost_usd for r in self._resource_records)
        total_time = sum(r.elapsed_s for r in self._resource_records)
        return {
            "num_calls": len(self._resource_records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_time_s": round(total_time, 2),
            "records": [r.to_dict() for r in self._resource_records],
        }

    def adapt(
        self,
        raw_report: str,
        query_material: str,
        query_property: str,
        source_urls: Optional[List[str]] = None,
        material_family: str = "",
    ) -> Dict[str, Any]:
        """Convert a research report to db_entry-compatible dict.

        Args:
            raw_report: The prose research report text.
            query_material: Material name (e.g. "Memory Foam").
            query_property: Property name (e.g. "thermal_conductivity").
            source_urls: Optional list of source URLs found by the DR agent.
            material_family: Optional material family (e.g. "Foams").

        Returns:
            Dict in db_entry.json format compatible with GTFile schema.
        """
        if not raw_report or len(raw_report.strip()) < 20:
            logger.warning("ReportAdapter: Report too short (%d chars), returning empty db_entry.", len(raw_report))
            return self._empty_db_entry(query_material, query_property, material_family)

        # Build source UID/URL pairs for prompt
        from .evidence_fetcher import url_to_uid
        source_uids = [url_to_uid(url) for url in (source_urls or [])]
        source_info = "\n".join(
            f"  - UID: {uid}, URL: {url}"
            for uid, url in zip(source_uids, source_urls or [])
        ) or "  (no source URLs provided — extract URLs from the report text)"

        # Build extraction prompt
        prompt = EXTRACTION_PROMPT.format(
            material=query_material,
            property=query_property,
            source_info=source_info,
            report=raw_report[:50000],  # Cap at 50K chars to stay within context
        )

        # Call LLM
        extraction = self._call_llm(prompt)
        if extraction is None:
            return self._empty_db_entry(query_material, query_property, material_family)

        # Build db_entry from extraction
        return self._build_db_entry(
            extraction, query_material, query_property,
            material_family or self.material_family,
            source_urls or [],
            raw_report,
        )

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the extraction LLM and parse JSON response."""
        record = AdapterResourceRecord(model=self.model, provider=self.provider)
        t0 = time.monotonic()

        try:
            if self.provider == "gemini":
                result = self._call_gemini(prompt, record)
            elif self.provider == "openai":
                result = self._call_openai(prompt, record)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            record.elapsed_s = time.monotonic() - t0
            record.success = result is not None
            self._resource_records.append(record)
            return result

        except Exception as e:
            record.elapsed_s = time.monotonic() - t0
            record.error = str(e)
            record.success = False
            self._resource_records.append(record)
            logger.error("ReportAdapter: LLM call failed: %s", e)
            return None

    # ── Schema for constrained decoding ─────────────────────────────────────
    # Mirrors the production HapticMaterialProperty schema from grounding_eval.py
    # so that both the production pipeline and the DR adapter produce identical
    # output structures. Additional optional fields: no_answer, no_answer_reason.
    #
    # Post-hoc fields NOT included here (computed later by eval harness):
    #   property_stats, value_list, is_grounded, successful_groundings,
    #   normalized_value, normalized_units

    _VALUE_TYPE_SCHEMA = {
        "anyOf": [
            {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
            {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}, "required": ["min", "max"]},
            {"type": "object", "properties": {"mean": {"type": "number"}, "std": {"type": "number"}}, "required": ["mean"]},
            {"type": "object", "properties": {"data_points": {"type": "array", "items": {"type": "number"}}}, "required": ["data_points"]},
        ]
    }

    _CONDITION_PAIR_SCHEMA = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Condition name (temperature, grade, etc.)."},
            "val": {"type": "string", "description": "Verbatim value from the source text."},
        },
        "required": ["label", "val"],
    }

    _EXTRACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "material_family": {"type": "string", "description": "Top-level material category."},
            "material_class": {"type": "string", "description": "Intermediate material level."},
            "haptic_property": {"type": "string", "description": "Name of the haptic/tactile property measured."},
            "value_condition_mapping": {
                "type": "array",
                "description": "List of values, each bundled with its subclass & conditions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "material_subclass": {
                            "type": "string",
                            "description": "Most specific material name, or 'unspecified'.",
                        },
                        "measurement_conditions": {
                            "type": "array",
                            "description": "Environmental/procedural details as label/val pairs.",
                            "items": _CONDITION_PAIR_SCHEMA,
                        },
                        "material_specifications": {
                            "type": "array",
                            "description": "Treatments, gradings, coatings as label/val pairs.",
                            "items": _CONDITION_PAIR_SCHEMA,
                        },
                        "object_class": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Physical form-factor (sheet, wire, bulk, etc.).",
                        },
                        "units": {"type": "string", "description": "Units for the property (e.g. MPa)."},
                        "value": _VALUE_TYPE_SCHEMA,
                        "normalized_value": {
                            "type": "object",
                            "description": "Same as value — canonical normalized representation.",
                            "properties": {
                                "value": {"type": "number"},
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                                "mean": {"type": "number"},
                                "std": {"type": "number"},
                            },
                        },
                        "normalized_units": {"type": "string", "description": "Same as units — canonical normalized units."},
                    },
                    "required": ["units", "value"],
                },
            },
            "citations": {
                "type": "array",
                "description": "Minimal set of verbatim excerpts supporting the extracted values.",
                "items": {
                    "type": "object",
                    "properties": {
                        "verbatim_snippet": {
                            "type": "string",
                            "description": "Verbatim text exactly as it appears in the source.",
                        },
                        "supports_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact model-field paths justified by this snippet, e.g. value_condition_mapping[0].value",
                        },
                        "source_uid": {
                            "type": "string",
                            "description": "UID of the source document this snippet comes from (from the Source UIDs list).",
                        },
                        "source_url": {
                            "type": "string",
                            "description": "URL of the source document this snippet comes from.",
                        },
                    },
                    "required": ["verbatim_snippet", "supports_fields"],
                },
            },
            "sources": {
                "type": "array",
                "description": "List of structured source references.",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Direct link to the source."},
                        "title": {"type": "string", "description": "Title of the source."},
                        "note": {"type": "string", "description": "How the source reported the value."},
                        "answer_exists": {"type": "boolean", "description": "Does the source contain an answer."},
                    },
                    "required": ["url"],
                },
            },
            "labeler_note": {
                "type": "string",
                "description": "Leave empty — reserved for human labeler validation.",
                "nullable": True,
            },
            "no_answer": {
                "type": "boolean",
                "description": "Set to true if no relevant data was found for the query.",
            },
            "no_answer_reason": {
                "type": "string",
                "description": "If no_answer is true, explain why no data was found.",
            },
        },
        "required": ["value_condition_mapping", "citations", "sources"],
    }

    def _call_gemini(self, prompt: str, record: AdapterResourceRecord) -> Optional[Dict[str, Any]]:
        """Call Gemini API for extraction with constrained JSON schema decoding."""
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            ),
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=self._EXTRACTION_SCHEMA,
            max_output_tokens=65536,
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        # Extract usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            record.prompt_tokens = getattr(um, 'prompt_token_count', 0) or 0
            record.completion_tokens = getattr(um, 'candidates_token_count', 0) or 0
            record.total_tokens = getattr(um, 'total_token_count', 0) or 0
            record.cost_usd = self._estimate_cost_gemini(record)

        # Parse response text as JSON
        text = response.text or ""
        return self._parse_json_response(text)

    def _call_openai(self, prompt: str, record: AdapterResourceRecord) -> Optional[Dict[str, Any]]:
        """Call OpenAI API for extraction with constrained JSON schema decoding."""
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model if "gpt" in self.model else "gpt-5.4-nano",
            messages=[
                {"role": "system", "content": "You are a precise scientific data extractor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "HapticNetDRExtraction",
                    "schema": self._EXTRACTION_SCHEMA,
                    "strict": False,
                },
            },
        )

        # Extract usage
        if response.usage:
            record.prompt_tokens = response.usage.prompt_tokens or 0
            record.completion_tokens = response.usage.completion_tokens or 0
            record.total_tokens = response.usage.total_tokens or 0
            ctd = getattr(response.usage, "completion_tokens_details", None)
            if ctd:
                record.cost_usd = self._estimate_cost_openai(record, getattr(ctd, "reasoning_tokens", 0) or 0)

        text = response.choices[0].message.content or ""
        return self._parse_json_response(text)

    @staticmethod
    def _estimate_cost_gemini(record: AdapterResourceRecord) -> float:
        """Estimate USD cost for a Gemini adapter call."""
        # gemini-2.5-flash-lite: $0.10/1M in, $0.40/1M out
        return (record.prompt_tokens * 0.10 + record.completion_tokens * 0.40) / 1_000_000

    @staticmethod
    def _estimate_cost_openai(record: AdapterResourceRecord, reasoning_tokens: int = 0) -> float:
        """Estimate USD cost for an OpenAI adapter call."""
        # gpt-5.4-nano: $0.20/1M in, $1.25/1M out
        return (record.prompt_tokens * 0.20 + (record.completion_tokens + reasoning_tokens) * 1.25) / 1_000_000

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("ReportAdapter: Failed to parse JSON: %s\nText: %s", e, text[:300])
            # Try to find JSON object in the text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            # Truncated JSON recovery: close unclosed brackets/strings
            recovered = self._recover_truncated_json(text)
            if recovered:
                logger.info("ReportAdapter: Recovered %d values from truncated JSON.", 
                           len(recovered.get("values", [])))
                return recovered

            return None

    def _recover_truncated_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to recover a truncated JSON response by closing brackets.
        
        When the LLM output is cut off mid-JSON, we iteratively remove the
        last incomplete object from the values array and try to close the
        JSON structure.
        """
        import re as _re
        
        # Find the start of the values array
        val_match = _re.search(r'"values"\s*:\s*\[', text)
        if not val_match:
            return None
        
        # Strategy: find complete value objects by matching balanced braces
        # We'll collect text up to the last complete "}" that ends a value object
        values_start = val_match.end()
        
        # Find all complete value objects (they end with "}")
        depth = 0
        last_complete_pos = values_start
        i = values_start
        while i < len(text):
            ch = text[i]
            if ch == '"':
                # Skip string content
                i += 1
                while i < len(text) and text[i] != '"':
                    if text[i] == '\\':
                        i += 1  # skip escaped char
                    i += 1
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_complete_pos = i + 1
            elif ch == ']' and depth == 0:
                # End of values array
                last_complete_pos = i
                break
            i += 1
        
        if last_complete_pos <= values_start:
            return None
        
        # Build a valid JSON by closing the structure
        truncated_values = text[values_start:last_complete_pos].rstrip().rstrip(',')
        reconstructed = '{"value_condition_mapping": [' + truncated_values + '], "citations": [], "sources": [], "no_answer": false}'
        
        try:
            result = json.loads(reconstructed)
            if result.get("values"):
                return result
        except json.JSONDecodeError:
            pass
        
        return None

    def _build_db_entry(
        self,
        extraction: Dict[str, Any],
        material: str,
        property_name: str,
        family: str,
        source_urls: List[str],
        raw_report: str,
    ) -> Dict[str, Any]:
        """Build db_entry-compatible dict from extraction result.

        The extraction dict now matches the production HapticMaterialProperty schema
        (value_condition_mapping, citations, sources). This method:
        1. Normalizes VCM entries (ensures value payloads are valid dicts)
        2. Enriches sources with UIDs
        3. Computes post-hoc fields (property_stats, value_list)
        """
        from .evidence_fetcher import url_to_uid

        # ── Sources: merge extraction sources + DR-discovered URLs ────────
        source_records = {}
        for src in extraction.get("sources", []):
            url = src.get("url", "")
            if url:
                uid = url_to_uid(url)
                source_records[uid] = {
                    "url": url, "uid": uid,
                    "title": src.get("title", ""),
                    "note": src.get("note", ""),
                    "answer_exists": src.get("answer_exists", True),
                }
        # Also from legacy "sources_found" key (backwards compat with old schema)
        for sf in extraction.get("sources_found", []):
            url = sf.get("url", "")
            if url:
                uid = url_to_uid(url)
                if uid not in source_records:
                    source_records[uid] = {
                        "url": url, "uid": uid,
                        "title": sf.get("title", ""),
                        "note": sf.get("note", ""),
                        "answer_exists": True,
                    }
        for url in source_urls:
            uid = url_to_uid(url)
            if uid not in source_records:
                source_records[uid] = {
                    "url": url, "uid": uid, "title": "", "note": "",
                    "answer_exists": True,
                }

        # ── VCM entries: normalize value payloads ─────────────────────────
        vcm_entries = []
        # Production format: extraction has "value_condition_mapping"
        raw_vcm = extraction.get("value_condition_mapping", [])
        # Backwards compat: old format had "values"
        if not raw_vcm:
            raw_vcm = extraction.get("values", [])

        for v in raw_vcm:
            raw_value = v.get("value")
            value_payload = self._to_value_payload(raw_value)
            if value_payload is None:
                continue  # Skip unparseable values

            # Normalize conditions — may be list of {label,val} or dict
            conditions = v.get("measurement_conditions", [])
            if isinstance(conditions, dict):
                conditions = [
                    {"label": k, "val": str(val)}
                    for k, val in conditions.items() if val
                ]

            # Normalize material_specifications — may be list or missing
            specs = v.get("material_specifications", [])
            if isinstance(specs, dict):
                specs = [
                    {"label": k, "val": str(val)}
                    for k, val in specs.items() if val
                ]

            # Normalize object_class — may be list or string
            obj_class = v.get("object_class", [])
            if isinstance(obj_class, str):
                obj_class = [obj_class] if obj_class else []

            entry = {
                "value": value_payload,
                "units": v.get("units", ""),
                "normalized_value": v.get("normalized_value") or value_payload,
                "normalized_units": v.get("normalized_units") or v.get("units", ""),
                "measurement_conditions": conditions,
                "material_subclass": v.get("material_subclass", "unspecified"),
                "material_specifications": specs,
                "object_class": obj_class,
                "is_grounded": False,
                "successful_groundings": [],
            }
            vcm_entries.append(entry)

        # ── Citations: use directly if production format, else build from values ──
        citations = extraction.get("citations", [])
        if not citations:
            # Backwards compat: build citations from old values[].citation_snippet
            for i, v in enumerate(extraction.get("values", [])):
                snippet = v.get("citation_snippet", "")
                if snippet:
                    citations.append({
                        "verbatim_snippet": snippet,
                        "supports_fields": [f"value_condition_mapping[{i}].value"],
                    })

        # Link citations to sources via fuzzy URL matching
        source_uid_list = [s["uid"] for s in source_records.values()]
        source_url_list = [s["url"] for s in source_records.values()]
        self._link_citations_to_sources(citations, source_uid_list, source_url_list)

        # Handle no_answer
        no_answer = extraction.get("no_answer", False)
        no_answer_reason = extraction.get("no_answer_reason", "")

        # Compute post-hoc fields: property_stats and value_list
        property_stats, value_list = self._compute_property_stats_and_value_list(
            vcm_entries
        )

        db_entry = {
            "material_family": extraction.get("material_family", family) or family,
            "material_class": extraction.get("material_class", material) or material,
            "haptic_property": extraction.get("haptic_property", property_name) or property_name,
            "value_condition_mapping": vcm_entries,
            "citations": citations,
            "sources": list(source_records.values()),
            "labeler_note": None,
            "property_stats": property_stats if property_stats else None,
            "value_list": value_list if value_list else None,
        }

        if no_answer:
            db_entry["no_answer"] = True
            db_entry["no_answer_reason"] = no_answer_reason

        return db_entry

    @staticmethod
    def _link_citations_to_sources(
        citations: List[Dict[str, Any]],
        source_uids: List[str],
        source_urls: List[str],
    ) -> None:
        """Resolve citation source_url fields to source UIDs via fuzzy URL matching.

        Ported from shared/gtfile_adapter.py — handles URL variations (trailing slash,
        www prefix, partial matches) that the LLM may produce.
        """
        from urllib.parse import urlparse

        url_to_uid = {}
        for uid, url in zip(source_uids, source_urls):
            url_to_uid[url] = uid
            # Also index by domain+path for fuzzy matching
            parsed = urlparse(url)
            key = (parsed.netloc.lower().replace('www.', ''), parsed.path.rstrip('/'))
            url_to_uid[key] = uid

        for cit in citations:
            cit_url = cit.get("source_url", "") or ""
            cit_uid = cit.get("source_uid", "") or ""

            # If already has a valid UID, skip
            if cit_uid and cit_uid in source_uids:
                continue

            # Try exact URL match
            if cit_url in url_to_uid:
                cit["source_uid"] = url_to_uid[cit_url]
                continue

            # Try fuzzy domain+path match
            if cit_url:
                parsed = urlparse(cit_url)
                key = (parsed.netloc.lower().replace('www.', ''), parsed.path.rstrip('/'))
                if key in url_to_uid:
                    cit["source_uid"] = url_to_uid[key]
                    cit["source_url"] = next(
                        (u for u, uid in zip(source_urls, source_uids) if uid == url_to_uid[key]), cit_url
                    )
                    continue

            # Try substring match
            for url, uid in zip(source_urls, source_uids):
                if cit_url and (cit_url in url or url in cit_url):
                    cit["source_uid"] = uid
                    cit["source_url"] = url
                    break
            else:
                # Last resort: if only one source, assign it
                if len(source_uids) == 1:
                    cit["source_uid"] = source_uids[0]
                    cit["source_url"] = source_urls[0]

    @staticmethod
    def _to_value_payload(raw_value) -> Optional[Dict[str, Any]]:
        """Convert a raw value to ValuePayload dict format.

        Handles: floats, ints, strings like "6.48", "6.48-7.65", "{value: 0.081}".
        """
        if raw_value is None:
            return None

        # Already a dict (ValuePayload format)
        if isinstance(raw_value, dict):
            if "value" in raw_value or "min" in raw_value or "mean" in raw_value:
                return raw_value
            return None

        # Numeric
        if isinstance(raw_value, (int, float)):
            return {"value": float(raw_value)}

        # String: try parsing
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not raw_value:
                return None

            # Range: "6.48-7.65" or "6.48 - 7.65"
            import re
            range_match = re.match(
                r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[-–—to]+\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)$',
                raw_value
            )
            if range_match:
                return {"min": float(range_match.group(1)), "max": float(range_match.group(2))}

            # Single number
            try:
                return {"value": float(raw_value)}
            except ValueError:
                pass

            # Last resort: treat as scalar with None
            return None

        return None

    @staticmethod
    def _compute_property_stats_and_value_list(
        vcm_entries: List[Dict[str, Any]],
    ) -> tuple:
        """Compute property_stats and value_list from VCM entries.

        Mirrors the production pipeline's format:
            property_stats = {mean, median, standard_deviation}
            value_list = {
                all_values:       {values: [...], units, count},
                grounded_values:  {values: [...], units, count},
                ungrounded_values:{values: [...], units, count},
            }
        """
        import statistics

        all_values = []
        grounded_values = []
        ungrounded_values = []
        units_set = set()
        all_nums = []

        for entry in vcm_entries:
            val_payload = entry.get("value", {})
            units = entry.get("units", "")
            if units:
                units_set.add(units)

            # Extract numeric values from payload
            nums = []
            if isinstance(val_payload, dict):
                if val_payload.get("value") is not None:
                    nums.append(float(val_payload["value"]))
                if val_payload.get("min") is not None:
                    nums.append(float(val_payload["min"]))
                if val_payload.get("max") is not None:
                    nums.append(float(val_payload["max"]))
                if val_payload.get("mean") is not None:
                    nums.append(float(val_payload["mean"]))
                for dp in val_payload.get("data_points", []) or []:
                    if dp is not None:
                        nums.append(float(dp))

            # Build value dict (exclude None values)
            vd = {k: v for k, v in val_payload.items() if v is not None} if isinstance(val_payload, dict) else {}

            if vd:
                all_values.append(vd)
                is_grounded = entry.get("is_grounded", False) or bool(entry.get("successful_groundings"))
                if is_grounded:
                    grounded_values.append(vd)
                else:
                    ungrounded_values.append(vd)

            all_nums.extend(nums)

        # Compute property_stats
        property_stats = {}
        if all_nums:
            property_stats["mean"] = statistics.mean(all_nums)
            property_stats["median"] = statistics.median(all_nums)
            if len(all_nums) >= 2:
                property_stats["standard_deviation"] = statistics.stdev(all_nums)
            else:
                property_stats["standard_deviation"] = 0.0

        # Build value_list
        canonical_units = (
            list(units_set)[0] if len(units_set) == 1
            else ", ".join(sorted(units_set)) if units_set
            else None
        )
        value_list = {
            "all_values": {
                "values": all_values,
                "units": canonical_units,
                "count": len(all_values),
            },
            "grounded_values": {
                "values": grounded_values,
                "units": canonical_units,
                "count": len(grounded_values),
            },
            "ungrounded_values": {
                "values": ungrounded_values,
                "units": canonical_units,
                "count": len(ungrounded_values),
            },
        }

        return property_stats, value_list

    def _empty_db_entry(
        self, material: str, property_name: str, family: str,
    ) -> Dict[str, Any]:
        """Return an empty db_entry for failed extractions."""
        empty_cat = {"values": [], "units": None, "count": 0}
        return {
            "material_family": family,
            "material_class": material,
            "haptic_property": property_name,
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
            "no_answer_reason": "Report adapter failed to extract values",
        }
