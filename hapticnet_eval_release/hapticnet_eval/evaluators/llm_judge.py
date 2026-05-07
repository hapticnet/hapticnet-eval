import os
import json
import time
import logging
from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMResourceTracker:
    """Tracks token usage (input, output, thinking) and cost across LLM judge calls."""
    # Per-model pricing (per 1M tokens)
    MODEL_PRICING = {
        # Gemini 2.5 Flash Lite
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "thinking": 0.40},
        # Gemini 2.5 Flash
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "thinking": 0.60},
        # Gemini 3.1 Pro
        "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00, "thinking": 12.00},
        "gemini-3.1-pro": {"input": 2.00, "output": 12.00, "thinking": 12.00},
        # Gemini 3.1 Flash Lite
        "gemini-3.1-flash-lite-preview": {"input": 0.10, "output": 0.40, "thinking": 0.40},
    }
    # Fallback pricing (Flash Lite)
    INPUT_COST_PER_M = 0.10    # $0.10/1M input tokens
    OUTPUT_COST_PER_M = 0.40   # $0.40/1M output tokens
    THINKING_COST_PER_M = 0.40  # $0.40/1M thinking tokens

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.total_calls = 0
        self.total_latency_ms = 0
        self.per_call_records = []

    def _get_pricing(self, model: str = "") -> tuple:
        """Get (input, output, thinking) pricing per 1M tokens for a model."""
        p = self.MODEL_PRICING.get(model, None)
        if p:
            return p["input"], p["output"], p["thinking"]
        # Try partial match
        for key, pricing in self.MODEL_PRICING.items():
            if key in model or model in key:
                return pricing["input"], pricing["output"], pricing["thinking"]
        return self.INPUT_COST_PER_M, self.OUTPUT_COST_PER_M, self.THINKING_COST_PER_M

    def record(self, input_tokens: int, output_tokens: int, thinking_tokens: int, latency_ms: float, evaluator: str = "", model: str = ""):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_thinking_tokens += thinking_tokens
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        ip, op, tp = self._get_pricing(model)
        cost = (input_tokens * ip + output_tokens * op + thinking_tokens * tp) / 1_000_000
        self.per_call_records.append({
            "evaluator": evaluator,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
        })

    @property
    def total_cost_usd(self) -> float:
        return (
            self.total_input_tokens * self.INPUT_COST_PER_M / 1_000_000 +
            self.total_output_tokens * self.OUTPUT_COST_PER_M / 1_000_000 +
            self.total_thinking_tokens * self.THINKING_COST_PER_M / 1_000_000
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.total_latency_ms / max(self.total_calls, 1), 1),
        }

    def evaluator_summary(self, evaluator_name: str) -> Dict[str, Any]:
        """Get resource summary for a specific evaluator."""
        records = [r for r in self.per_call_records if r["evaluator"] == evaluator_name]
        inp = sum(r["input_tokens"] for r in records)
        out = sum(r["output_tokens"] for r in records)
        think = sum(r["thinking_tokens"] for r in records)
        lat = sum(r["latency_ms"] for r in records)
        cost = (
            inp * self.INPUT_COST_PER_M / 1_000_000 +
            out * self.OUTPUT_COST_PER_M / 1_000_000 +
            think * self.THINKING_COST_PER_M / 1_000_000
        )
        return {
            "calls": len(records),
            "input_tokens": inp,
            "output_tokens": out,
            "thinking_tokens": think,
            "total_tokens": inp + out + think,
            "cost_usd": round(cost, 6),
            "latency_ms": round(lat, 1),
        }

    def evaluator_call_records(self, evaluator_name: str) -> list:
        """Get individual call records for a specific evaluator (for per-call display)."""
        records = [r for r in self.per_call_records if r["evaluator"] == evaluator_name]
        # Add cost per call
        for r in records:
            r["cost_usd"] = round(
                r["input_tokens"] * self.INPUT_COST_PER_M / 1_000_000 +
                r["output_tokens"] * self.OUTPUT_COST_PER_M / 1_000_000 +
                r["thinking_tokens"] * self.THINKING_COST_PER_M / 1_000_000,
                6
            )
        return records


# Global tracker instance — shared across all evaluators in a single benchmark run
_global_tracker = None

def get_resource_tracker() -> LLMResourceTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LLMResourceTracker()
    return _global_tracker

def reset_resource_tracker():
    global _global_tracker
    _global_tracker = LLMResourceTracker()


class LLMJudge:
    """
    A robust LLM judge using Google GenAI SDK.
    Supports structured output via Pydantic schemas and incorporates rate-limiting retries.
    Tracks token usage and cost via the global LLMResourceTracker.

    For Gemini 2.5 models: use thinking_budget (int, token count).
    For Gemini 3.x models: use thinking_level ("low", "medium", "high").
    """
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        thinking_budget: int = 1024,
        thinking_level: str | None = None,
    ):
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level  # "low", "medium", "high" for 3.x models
        
        # We import locally to keep dependency optional for users who don't run LLM evaluators
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-genai is required for LLM evaluators. "
                "Please run: pip install google-genai"
            )

        # Load .env file if GOOGLE_API_KEY is not already in the environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                # Walk upward from this file to find the project root .env
                _dir = os.path.dirname(os.path.abspath(__file__))
                for _ in range(10):
                    env_path = os.path.join(_dir, ".env")
                    if os.path.isfile(env_path):
                        load_dotenv(env_path, override=False)
                        logger.info("Loaded .env from %s", env_path)
                        break
                    _dir = os.path.dirname(_dir)
                api_key = os.getenv("GOOGLE_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your environment or in a .env file "
                "in the project root."
            )
            
        self.client = self.genai.Client(api_key=api_key)

    def _is_3x_model(self) -> bool:
        """Check if the model is a Gemini 3.x series model."""
        return "3." in self.model_name or "gemini-3" in self.model_name

    # Map thinking_level names to budget token counts.
    # Matches the mapping in simple_llm_extraction.py for consistency.
    THINKING_LEVEL_BUDGETS = {
        "minimal": 128,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }

    def generate_structured(self, prompt: str, schema_class: Type[T], max_retries: int = 8, evaluator_name: str = "", timeout_seconds: int = 120) -> Optional[T]:
        """
        Generate a structured response adhering to the provided Pydantic schema class.
        Includes exponential backoff for rate limits, per-call timeout, and token tracking.
        """
        result = self.generate_structured_with_usage(prompt, schema_class, max_retries, evaluator_name, timeout_seconds)
        if result is None:
            return None
        return result[0]

    def generate_structured_with_usage(
        self, prompt: str, schema_class: Type[T],
        max_retries: int = 8, evaluator_name: str = "",
        timeout_seconds: int = 120,
    ) -> Optional[tuple]:
        """
        Like generate_structured but returns (parsed_result, usage_dict) or None.
        usage_dict has: input_tokens, output_tokens, thinking_tokens, latency_ms.
        """
        from google.genai import types, errors
        import concurrent.futures
        
        config_kwargs = dict(
            response_mime_type="application/json",
            response_schema=schema_class.model_json_schema(),
            temperature=0.0,
            max_output_tokens=8192,
        )
        
        # Determine thinking budget
        budget = self.thinking_budget
        if self.thinking_level:
            budget = self.THINKING_LEVEL_BUDGETS.get(
                self.thinking_level.lower(), self.thinking_budget
            )

        if budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget
            )
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                
                # Use a thread pool to enforce a timeout on the API call
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    try:
                        response = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        logger.error(
                            "LLM Judge call timed out after %ds (model=%s, evaluator=%s, attempt %d/%d)",
                            timeout_seconds, self.model_name, evaluator_name, attempt + 1, max_retries
                        )
                        continue  # retry on timeout
                
                latency_ms = (time.time() - t0) * 1000
                
                # Extract token usage from response metadata
                input_tokens = 0
                output_tokens = 0
                thinking_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(um, 'candidates_token_count', 0) or 0
                    # Thinking tokens reported separately in some models
                    thinking_tokens = getattr(um, 'thoughts_token_count', 0) or 0
                    if thinking_tokens == 0:
                        thinking_tokens = getattr(um, 'cached_content_token_count', 0) or 0
                
                # Thread-safe tracker update
                import threading
                tracker = get_resource_tracker()
                with threading.Lock():
                    tracker.record(input_tokens, output_tokens, thinking_tokens, latency_ms, evaluator_name, model=self.model_name)
                
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "latency_ms": round(latency_ms, 1),
                }
                
                # Parse JSON string into Pydantic model
                data = json.loads(response.text)
                parsed = schema_class.model_validate(data)
                return parsed, usage
                
            except errors.ClientError as e:
                if e.code == 429:
                    # Aggressive backoff: 4s, 8s, 16s, 32s, 64s... for free-tier rate limits
                    sleep_time = min(4 * (2 ** attempt), 120)  # cap at 2 minutes
                    logger.warning("LLM Judge rate limited (429). Retrying in %ds... (attempt %d/%d)", 
                                   sleep_time, attempt + 1, max_retries)
                    time.sleep(sleep_time)
                    continue
                logger.error("LLM Judge ClientError (model=%s, evaluator=%s): %s", 
                            self.model_name, evaluator_name, e)
                break
            except Exception as e:
                logger.error("LLM Judge generation failed (model=%s, evaluator=%s, attempt %d/%d): %s: %s", 
                            self.model_name, evaluator_name, attempt + 1, max_retries,
                            type(e).__name__, e)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                break
                
        logger.error("LLM Judge failed after %d retries (model=%s, evaluator=%s).", 
                     max_retries, self.model_name, evaluator_name)
        return None


