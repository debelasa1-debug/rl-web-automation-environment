"""
agents/rule_based_agent.py
Deterministic rule-based agent that uses heuristics to complete web tasks.

Decision logic (priority order)
────────────────────────────────
1. If page has an unvisited required URL → navigate there
2. If task requires text search and a search box is visible → type + submit
3. If a relevant link text matches task keywords → click it
4. If the page has a submit/login button and form is filled → submit
5. If no progress → scroll down to reveal more elements
6. Fallback → no_op
"""

from __future__ import annotations

import re
from typing import Optional
from loguru import logger

from env.action_space import ActionType, WebAction
from env.observation import WebObservation
from env.task_manager import Task, TaskCategory


class RuleBasedAgent:
    """
    Lightweight rule-based agent.
    Stateless between calls; relies entirely on the observation dict.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._submitted_forms: set[str] = set()
        self._clicked: set[str] = set()
        self._typed: dict[str, str] = {}
        self._scroll_count: int = 0

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._submitted_forms.clear()
        self._clicked.clear()
        self._typed.clear()
        self._scroll_count = 0

    def act(self, observation: dict, task: Task) -> WebAction:
        """
        Choose an action given the current observation and task.

        Returns
        -------
        WebAction
        """
        url = observation.get("current_url", "")
        text = observation.get("text_content", "").lower()
        inputs = observation.get("inputs", [])
        links = observation.get("links", [])
        buttons = observation.get("buttons", [])
        has_search = observation.get("has_search_box", False)
        has_login = observation.get("has_login_form", False)
        task_instr = task.instruction.lower()
        task_words = self._keywords(task_instr)

        # ── Rule 1: Direct navigation if we know the target URL ─────────
        if task.success_criteria.url_contains:
            target_url_part = task.success_criteria.url_contains
            if target_url_part.lower() not in url.lower():
                # Try to find a link pointing there
                nav_url = self._find_nav_link(links, target_url_part)
                if nav_url:
                    return self._action(ActionType.CLICK, target=nav_url,
                                        reason=f"nav toward {target_url_part}")

        # ── Rule 2: Search box available + task needs search ────────────
        if has_search and not self._typed.get("search"):
            search_query = self._build_search_query(task)
            if search_query:
                # Find the best search input selector
                selector = self._find_search_selector(inputs)
                if selector:
                    self._typed["search"] = search_query
                    return self._action(ActionType.TYPE, target=selector,
                                        value=search_query, reason="fill search box")

        # ── Rule 2b: Submit search after typing ─────────────────────────
        if self._typed.get("search") and "search" not in self._submitted_forms:
            self._submitted_forms.add("search")
            return self._action(ActionType.PRESS_KEY, value="Enter",
                                reason="submit search")

        # ── Rule 3: Login form handling ─────────────────────────────────
        if has_login and "login" not in self._submitted_forms:
            # Look for username / password fields
            user_sel = self._find_input_by_type(inputs, "text", ["user", "name", "email", "login"])
            pass_sel = self._find_input_by_type(inputs, "password", [])

            if user_sel and "user_typed" not in self._typed:
                self._typed["user_typed"] = "user"
                return self._action(ActionType.TYPE, target=user_sel,
                                    value="user", reason="fill username")

            if pass_sel and "pass_typed" not in self._typed:
                self._typed["pass_typed"] = "password"
                return self._action(ActionType.TYPE, target=pass_sel,
                                    value="password", reason="fill password")

            if "user_typed" in self._typed and "pass_typed" in self._typed:
                self._submitted_forms.add("login")
                return self._action(ActionType.PRESS_KEY, value="Enter",
                                    reason="submit login form")

        # ── Rule 4: Form fill (generic) ─────────────────────────────────
        if task.category == TaskCategory.FORM_FILLING:
            action = self._handle_form_fill(inputs, buttons, task)
            if action:
                return action

        # ── Rule 5: Click a relevant link ───────────────────────────────
        link_action = self._find_best_link(links, task_words, url)
        if link_action:
            return link_action

        # ── Rule 6: Click a relevant button ─────────────────────────────
        btn_action = self._find_best_button(buttons, task_words)
        if btn_action:
            return btn_action

        # ── Rule 7: Scroll down to reveal more content ──────────────────
        if self._scroll_count < 4:
            self._scroll_count += 1
            return self._action(ActionType.SCROLL, value="down",
                                reason=f"scroll down #{self._scroll_count}")

        # ── Fallback ─────────────────────────────────────────────────────
        logger.warning("[RuleAgent] No rule matched – no_op")
        return WebAction(action_type=ActionType.NO_OP)

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------

    def _build_search_query(self, task: Task) -> str:
        """
        Extract a search query from the task instruction.
        Strategy: look for quoted strings, then fall back to keyword extraction.
        """
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", task.instruction)
        if quoted:
            return quoted[0][0] or quoted[0][1]

        # Keywords approach: remove common stop words
        stop = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "by",
                "for", "of", "with", "navigate", "go", "search", "find", "click",
                "use", "type", "enter", "fill", "visit", "open", "page", "site"}
        words = [w for w in self._keywords(task.instruction) if w not in stop]
        return " ".join(words[:5]) if words else ""

    def _find_search_selector(self, inputs: list[dict]) -> Optional[str]:
        """Return CSS selector for the most likely search input."""
        for el in inputs:
            t = el.get("type", "").lower()
            p = el.get("placeholder", "").lower()
            n = el.get("name", "").lower()
            el_id = el.get("id", "").lower()
            if t == "search":
                return f"input[type='search']"
            if "search" in p or "search" in n or "query" in n or "q" == n:
                return f"input[name='{el.get('name','q')}']" if el.get("name") else "input[type='text']"
            if "search" in el_id:
                return f"#{el.get('id')}"
        # Fallback: first text input
        for el in inputs:
            if el.get("type", "").lower() in ("text", ""):
                return "input[type='text']"
        return "input"

    def _find_input_by_type(
        self, inputs: list[dict], input_type: str, name_hints: list[str]
    ) -> Optional[str]:
        for el in inputs:
            if el.get("type", "").lower() == input_type:
                n = el.get("name", "").lower()
                p = el.get("placeholder", "").lower()
                el_id = el.get("id", "").lower()
                if not name_hints or any(h in n or h in p or h in el_id for h in name_hints):
                    name = el.get("name") or el.get("id")
                    if name:
                        return f"input[name='{el.get('name')}']" if el.get("name") else f"#{el.get('id')}"
                    return f"input[type='{input_type}']"
        return None

    def _find_nav_link(self, links: list[dict], target_part: str) -> Optional[str]:
        """Find a link whose href contains the target URL fragment."""
        for el in links:
            href = el.get("href", "")
            if target_part.lower() in href.lower():
                if href.startswith("http"):
                    return href
        return None

    def _find_best_link(
        self, links: list[dict], task_words: list[str], current_url: str
    ) -> Optional[WebAction]:
        """Find and click the most relevant unvisited link."""
        scored: list[tuple[int, dict]] = []
        for el in links:
            text = el.get("text", "").lower()
            href = el.get("href", "")
            if not href or href == current_url:
                continue
            score = sum(1 for w in task_words if w in text or w in href.lower())
            if score > 0:
                scored.append((score, el))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        best_el = scored[0][1]
        href = best_el.get("href", "")
        text = best_el.get("text", "")[:30]

        click_key = href or text
        if click_key in self._clicked:
            # Try second best
            if len(scored) > 1:
                best_el = scored[1][1]
                href = best_el.get("href", "")
                text = best_el.get("text", "")[:30]
                click_key = href or text

        if click_key in self._clicked:
            return None

        self._clicked.add(click_key)

        if href.startswith("http"):
            return self._action(ActionType.NAVIGATE, target=href,
                                reason=f"navigate to relevant link: {text}")

        # Use partial link text selector
        safe_text = re.sub(r'["\']', '', text).strip()
        if safe_text:
            return self._action(ActionType.CLICK, target=f"a:has-text('{safe_text[:30]}')",
                                reason=f"click link: {safe_text}")
        return None

    def _find_best_button(
        self, buttons: list[dict], task_words: list[str]
    ) -> Optional[WebAction]:
        for el in buttons:
            text = el.get("text", "").lower()
            if any(w in text for w in task_words):
                btn_id = el.get("id", "")
                safe = re.sub(r'["\']', '', text[:30]).strip()
                sel = f"button:has-text('{safe}')" if safe else "button[type='submit']"
                if sel not in self._clicked:
                    self._clicked.add(sel)
                    return self._action(ActionType.CLICK, target=sel,
                                        reason=f"click button: {text}")
        return None

    def _handle_form_fill(
        self, inputs: list[dict], buttons: list[dict], task: Task
    ) -> Optional[WebAction]:
        """Handle generic form filling tasks."""
        # If there's an unfilled text input, fill it
        for el in inputs:
            if el.get("type", "").lower() in ("text", ""):
                name = el.get("name") or el.get("id") or ""
                if name and name not in self._typed:
                    query = self._build_search_query(task)
                    sel = f"input[name='{name}']" if el.get("name") else "input[type='text']"
                    self._typed[name] = query
                    return self._action(ActionType.TYPE, target=sel,
                                        value=query or "RL Agent",
                                        reason=f"fill field: {name}")
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _keywords(text: str) -> list[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())

    def _action(
        self,
        action_type: ActionType,
        target: str = "",
        value: str = "",
        reason: str = "",
    ) -> WebAction:
        if self.verbose:
            logger.info("[RuleAgent] {} → {} | {}", action_type.value, target[:50], reason)
        return WebAction(action_type=action_type.value, target=target, value=value)
