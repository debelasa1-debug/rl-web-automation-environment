"""
agents/llm_agent.py
LLM-based agent with multi-provider support.

Provider auto-detection priority (first key found wins):
  1. Explicit api_key argument
  2. CI env vars (FREE_LLM_BASE_URL / FREE_LLM_MODEL / FREE_LLM_KEY_VAR)
  3. GEMINI_API_KEY    → Google Gemini 2.5 Flash (free, no credit card)
  4. GROQ_API_KEY      → Groq Llama 3.3 70B    (free, 1 000 req/day)
  5. OPENROUTER_API_KEY → OpenRouter free models (free, 50 req/day)
  6. ANTHROPIC_API_KEY → Claude (paid)

Providers 3-5 use the OpenAI-compatible endpoint so only the `openai`
package is needed for them.  Provider 6 uses the `anthropic` package.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from loguru import logger

from agents.rule_based_agent import RuleBasedAgent
from env.action_space import ActionSpace, ActionType, WebAction
from env.observation import WebObservation
from env.task_manager import Task


SYSTEM_PROMPT = """\
You are an expert web automation AI agent operating inside a Gymnasium RL environment.
Your goal is to complete the given web task as efficiently as possible.

You receive a structured observation of the current browser state and must output
a single action in JSON format.

AVAILABLE ACTION TYPES
──────────────────────
- navigate   : Go to a URL.         {"action_type": "navigate", "target": "https://..."}
- click      : Click an element.    {"action_type": "click",    "target": "css_selector"}
- type       : Type into a field.   {"action_type": "type",     "target": "css_selector", "value": "text"}
- scroll     : Scroll the page.     {"action_type": "scroll",   "value": "down|up"}
- press_key  : Press a key.         {"action_type": "press_key","value": "Enter|Tab|..."}
- submit     : Submit a form.       {"action_type": "submit",   "target": "form_selector"}
- extract    : Read element text.   {"action_type": "extract",  "target": "css_selector"}
- wait       : Wait N seconds.      {"action_type": "wait",     "value": "1"}
- no_op      : Do nothing.          {"action_type": "no_op"}

RULES
─────
1. Respond ONLY with a single JSON object — no markdown, no explanation.
2. Prefer clicking links and buttons visible on the page over navigating directly.
3. After typing in a search box, use press_key Enter to submit.
4. If the task appears complete, use no_op.
5. Do NOT repeat an action that already failed.
6. Always pick the action most likely to make progress toward the task goal.

OUTPUT FORMAT (strict JSON, nothing else)
─────────────────────────────────────────
{"action_type": "...", "target": "...", "value": "..."}
"""


class LLMAgent:
    """
    LLM-powered agent supporting Anthropic Claude, Google Gemini (free),
    Groq (free), and OpenRouter (free) via auto-detection of env vars.
    Falls back to rule-based decisions on API failure or missing key.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        fallback_to_rules: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_to_rules = fallback_to_rules
        self.verbose = verbose
        self._action_space = ActionSpace()
        self._fallback = RuleBasedAgent(verbose=verbose)
        self._client = None
        self._is_openai_compat = False
        self._history: list[dict] = []

        self._init_client(api_key)

    def _init_client(self, explicit_key: Optional[str]) -> None:
        """
        Detect the best available LLM provider and initialise the client.
        """
        # ── 1. CI-patched free provider (set by GitHub Actions workflow) ───
        ci_base_url  = os.getenv("FREE_LLM_BASE_URL")
        ci_model     = os.getenv("FREE_LLM_MODEL")
        ci_key_var   = os.getenv("FREE_LLM_KEY_VAR")
        if ci_base_url and ci_key_var:
            key = os.getenv(ci_key_var)
            if key:
                self._setup_openai_compat(key, ci_base_url, ci_model or self.model)
                return

        # ── 2. Explicit key argument (assumes Anthropic) ──────────────────
        if explicit_key:
            self._setup_anthropic(explicit_key)
            return

        # ── 3. Free providers (auto-detected from env) ────────────────────
        FREE_PROVIDERS = [
            ("GEMINI_API_KEY",
             "https://generativelanguage.googleapis.com/v1beta/openai/",
             "gemini-2.5-flash"),
            ("GROQ_API_KEY",
             "https://api.groq.com/openai/v1",
             "llama-3.3-70b-versatile"),
            ("OPENROUTER_API_KEY",
             "https://openrouter.ai/api/v1",
             "meta-llama/llama-3.3-70b-instruct:free"),
        ]
        for env_var, base_url, default_model in FREE_PROVIDERS:
            key = os.getenv(env_var)
            if key:
                # If caller didn't override the model, use provider default
                mdl = self.model if self.model != "claude-sonnet-4-6" else default_model
                self._setup_openai_compat(key, base_url, mdl)
                logger.info("LLMAgent: using {} → model={}", env_var, self.model)
                return

        # ── 4. Anthropic (paid) ───────────────────────────────────────────
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            self._setup_anthropic(key)
            return

        logger.warning(
            "LLMAgent: no API key found. Set GEMINI_API_KEY (free), GROQ_API_KEY (free), "
            "OPENROUTER_API_KEY (free), or ANTHROPIC_API_KEY (paid). "
            "Falling back to rule-based agent."
        )

    def _setup_openai_compat(self, key: str, base_url: str, model: str) -> None:
        try:
            import openai
            self._client = openai.OpenAI(api_key=key, base_url=base_url)
            self._is_openai_compat = True
            self.model = model
            logger.success("LLMAgent ready (OpenAI-compat) base_url={} model={}", base_url, model)
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    def _setup_anthropic(self, key: str) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
            self._is_openai_compat = False
            logger.success("LLMAgent ready (Anthropic) model={}", self.model)
        except ImportError:
            logger.warning("anthropic package not installed. Run: pip install anthropic")

    def reset(self) -> None:
        """Reset conversation history for a new episode."""
        self._history = []
        self._fallback.reset()

    def act(self, observation: dict, task: Task) -> WebAction:
        """
        Select an action given the current observation and task.
        Returns a WebAction (falls back to rule-based on any error).
        """
        if self._client is None:
            return self._fallback.act(observation, task)

        obs_obj = WebObservation(**{k: v for k, v in observation.items()
                                    if k in WebObservation.__dataclass_fields__})
        prompt = obs_obj.to_llm_prompt()
        self._history.append({"role": "user", "content": prompt})
        if len(self._history) > 6:
            self._history = self._history[-6:]

        try:
            if self._is_openai_compat:
                raw = self._call_openai_compat()
            else:
                raw = self._call_anthropic()

            if self.verbose:
                logger.debug("LLM response: {}", raw[:200])

            self._history.append({"role": "assistant", "content": raw})

            action_dict = self._parse_json(raw)
            if action_dict is None:
                raise ValueError(f"Could not parse JSON from: {raw}")

            web_action, err = self._action_space.validate(action_dict)
            if web_action is None:
                raise ValueError(f"Invalid action: {err}")

            logger.info("[LLMAgent] → {}", web_action)
            return web_action

        except Exception as exc:
            logger.warning("LLMAgent error: {} – falling back to rules", exc)
            if self.fallback_to_rules:
                return self._fallback.act(observation, task)
            return WebAction(action_type=ActionType.NO_OP)

    def _call_openai_compat(self) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._history,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=self._history,
        )
        return response.content[0].text.strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Extract and parse the first JSON object from text. Must return a dict."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return None
