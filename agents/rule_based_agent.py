"""
agents/rule_based_agent.py
Deterministic rule-based agent — patched v3.

Fixes applied
─────────────
• Rule 0 : direct_url metadata shortcut (nav_001 ✓)
• Rule 1 : NAVIGATE not CLICK for href URLs
• noise filter: wikidata / wikimedia / apple.com / theverge / apps.apple etc.
• domain-lock: once a task's start_url domain is known, don't navigate away
  to completely off-topic domains (search tasks stay on the search results)
• Wikipedia search: detect the Wikipedia search box specifically and use it
• _find_best_link: only follow same-domain OR task-target links when on a
  results page; avoids chasing random ads/sidebars
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from env.action_space import ActionType, WebAction
from env.task_manager import Task, TaskCategory


# Domains that are never a useful navigation target for our tasks
_NOISE_DOMAINS = (
    "wikidata.org", "wikimedia.org", "creativecommons.org",
    "mediawiki.org", "w3.org", "javascript:", "apple.com",
    "apps.apple.com", "theverge.com", "twitter.com", "facebook.com",
    "instagram.com", "youtube.com", "reddit.com", "amazon.com",
)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


class RuleBasedAgent:
    """
    Lightweight rule-based agent.
    Completely stateless between episodes (reset() clears everything).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._submitted_forms: set[str] = set()
        self._clicked: set[str] = set()
        self._navigated: set[str] = set()
        self._typed: dict[str, str] = {}
        self._scroll_count: int = 0

    def reset(self) -> None:
        self._submitted_forms.clear()
        self._clicked.clear()
        self._navigated.clear()
        self._typed.clear()
        self._scroll_count = 0

    def act(self, observation: dict, task: Task) -> WebAction:
        url      = observation.get("current_url", "")
        inputs   = observation.get("inputs", [])
        links    = observation.get("links", [])
        buttons  = observation.get("buttons", [])
        has_search = observation.get("has_search_box", False)
        has_login  = observation.get("has_login_form", False)
        task_words = self._keywords(task.instruction)

        current_domain = _domain(url)
        start_domain   = _domain(task.start_url)

        # ── Rule 0: direct_url / fallback_url metadata shortcuts ────────
        for meta_key in ("direct_url", "fallback_url"):
            shortcut = task.metadata.get(meta_key, "")
            if shortcut and shortcut not in self._navigated:
                crit = task.success_criteria
                already = crit.url_contains and crit.url_contains.lower() in url.lower()
                if not already:
                    self._navigated.add(shortcut)
                    return self._nav(shortcut, f"{meta_key} shortcut")

        # ── Rule 1: navigate toward success URL via matching link ────────
        if task.success_criteria.url_contains:
            target_frag = task.success_criteria.url_contains
            if target_frag.lower() not in url.lower():
                nav_url = self._find_nav_link(links, target_frag)
                if nav_url and nav_url not in self._navigated:
                    self._navigated.add(nav_url)
                    return self._nav(nav_url, f"nav toward {target_frag}")

        # ── Rule 2: Wikipedia search box (special case) ─────────────────
        # Wikipedia renders its search as input[name='search'] or #searchInput
        if "wikipedia.org" in url and not self._typed.get("wiki_search"):
            query = self._build_search_query(task)
            if query:
                for sel in ("input[name='search']", "#searchInput",
                            "input[type='search']", ".cdx-text-input__input"):
                    self._typed["wiki_search"] = query
                    return self._action(ActionType.TYPE, sel, query,
                                        "wikipedia search box")

        # ── Rule 2b: submit wiki search ─────────────────────────────────
        if self._typed.get("wiki_search") and "wiki_search_submitted" not in self._submitted_forms:
            self._submitted_forms.add("wiki_search_submitted")
            return self._action(ActionType.PRESS_KEY, "", "Enter", "submit wiki search")

        # ── Rule 3: generic search box ───────────────────────────────────
        if has_search and not self._typed.get("search"):
            query = self._build_search_query(task)
            if query:
                sel = self._find_search_selector(inputs)
                if sel:
                    self._typed["search"] = query
                    return self._action(ActionType.TYPE, sel, query, "fill search box")

        # ── Rule 3b: submit generic search ──────────────────────────────
        if self._typed.get("search") and "search" not in self._submitted_forms:
            self._submitted_forms.add("search")
            return self._action(ActionType.PRESS_KEY, "", "Enter", "submit search")

        # ── Rule 4: login form ───────────────────────────────────────────
        if has_login and "login" not in self._submitted_forms:
            user_sel = self._find_input_by_type(inputs, "text",
                                                ["user", "name", "email", "login"])
            pass_sel = self._find_input_by_type(inputs, "password", [])

            if user_sel and "user_typed" not in self._typed:
                self._typed["user_typed"] = "user"
                return self._action(ActionType.TYPE, user_sel, "user", "fill username")
            if pass_sel and "pass_typed" not in self._typed:
                self._typed["pass_typed"] = "password"
                return self._action(ActionType.TYPE, pass_sel, "password", "fill password")
            if "user_typed" in self._typed and "pass_typed" in self._typed:
                self._submitted_forms.add("login")
                return self._action(ActionType.PRESS_KEY, "", "Enter", "submit login")

        # ── Rule 5: generic form fill ────────────────────────────────────
        if task.category == TaskCategory.FORM_FILLING:
            a = self._handle_form_fill(inputs, task)
            if a:
                return a

        # ── Rule 6: click a relevant link (same-domain preferred) ────────
        link_action = self._find_best_link(links, task_words, url,
                                           start_domain, current_domain)
        if link_action:
            return link_action

        # ── Rule 7: click a relevant button ─────────────────────────────
        btn_action = self._find_best_button(buttons, task_words)
        if btn_action:
            return btn_action

        # ── Rule 8: scroll ───────────────────────────────────────────────
        if self._scroll_count < 4:
            self._scroll_count += 1
            return self._action(ActionType.SCROLL, "", "down",
                                f"scroll #{self._scroll_count}")

        # ── Fallback ─────────────────────────────────────────────────────
        logger.warning("[RuleAgent] No rule matched – no_op")
        return WebAction(action_type="no_op")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_search_query(self, task: Task) -> str:
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", task.instruction)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        stop = {"the","a","an","to","and","or","in","on","at","by","for","of",
                "with","navigate","go","search","find","click","use","type",
                "enter","fill","visit","open","page","site","confirm","verify",
                "article","successfully","loads","appear","results"}
        words = [w for w in self._keywords(task.instruction) if w not in stop]
        return " ".join(words[:5]) if words else ""

    def _find_search_selector(self, inputs: list[dict]) -> Optional[str]:
        for el in inputs:
            t  = el.get("type", "").lower()
            p  = el.get("placeholder", "").lower()
            n  = el.get("name", "").lower()
            eid = el.get("id", "").lower()
            if t == "search":
                return "input[type='search']"
            if "search" in p or "search" in n or n == "q" or "query" in n:
                name = el.get("name")
                return f"input[name='{name}']" if name else "input[type='text']"
            if "search" in eid:
                return f"#{el.get('id')}"
        for el in inputs:
            if el.get("type", "").lower() in ("text", ""):
                return "input[type='text']"
        return "input"

    def _find_input_by_type(self, inputs, input_type, name_hints):
        for el in inputs:
            if el.get("type", "").lower() != input_type:
                continue
            n   = el.get("name", "").lower()
            p   = el.get("placeholder", "").lower()
            eid = el.get("id", "").lower()
            if not name_hints or any(h in n or h in p or h in eid for h in name_hints):
                nm = el.get("name")
                eid2 = el.get("id")
                if nm:
                    return f"input[name='{nm}']"
                if eid2:
                    return f"#{eid2}"
                return f"input[type='{input_type}']"
        return None

    def _find_nav_link(self, links: list[dict], target_part: str) -> Optional[str]:
        for el in links:
            href = el.get("href", "")
            if target_part.lower() in href.lower() and href.startswith("http"):
                if not any(nd in href for nd in _NOISE_DOMAINS):
                    return href
        return None

    def _find_best_link(
        self,
        links: list[dict],
        task_words: list[str],
        current_url: str,
        start_domain: str,
        current_domain: str,
    ) -> Optional[WebAction]:
        scored: list[tuple[int, dict]] = []
        for el in links:
            text = el.get("text", "").lower()
            href = el.get("href", "")
            if not href or href == current_url:
                continue
            if href.startswith("#"):
                continue
            if any(nd in href for nd in _NOISE_DOMAINS):
                continue

            link_domain = _domain(href)

            # Domain guard: if we're on a search results page (URL has 'q=' or
            # we already typed a search), only follow same-domain or
            # task-start-domain links to avoid chasing ads
            on_results = ("q=" in current_url or self._typed.get("search"))
            if on_results and href.startswith("http"):
                if link_domain not in (start_domain, current_domain, ""):
                    continue

            score = sum(1 for w in task_words if w in text or w in href.lower())

            # Bonus: same domain as task start URL is safer
            if link_domain == start_domain:
                score += 1

            if score > 0:
                scored.append((score, el))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        for _, best_el in scored:
            href = best_el.get("href", "")
            text = best_el.get("text", "")[:30]
            key  = href or text
            if key in self._clicked:
                continue
            self._clicked.add(key)
            if href.startswith("http"):
                return self._nav(href, f"relevant link: {text}")
            safe = re.sub(r'["\']', '', text).strip()
            if safe:
                return self._action(ActionType.CLICK,
                                    f"a:has-text('{safe[:30]}')",
                                    "", f"click link: {safe}")
        return None

    def _find_best_button(self, buttons: list[dict], task_words: list[str]):
        for el in buttons:
            text = el.get("text", "").lower()
            if any(w in text for w in task_words):
                safe = re.sub(r'["\']', '', text[:30]).strip()
                sel = f"button:has-text('{safe}')" if safe else "button[type='submit']"
                if sel not in self._clicked:
                    self._clicked.add(sel)
                    return self._action(ActionType.CLICK, sel, "", f"button: {text}")
        return None

    def _handle_form_fill(self, inputs: list[dict], task: Task):
        for el in inputs:
            if el.get("type", "").lower() in ("text", ""):
                name = el.get("name") or el.get("id") or ""
                if name and name not in self._typed:
                    query = self._build_search_query(task)
                    sel = f"input[name='{el.get('name')}']" if el.get("name") else "input[type='text']"
                    self._typed[name] = query
                    return self._action(ActionType.TYPE, sel,
                                        query or "RL Agent", f"fill: {name}")
        return None

    @staticmethod
    def _keywords(text: str) -> list[str]:
        return re.findall(r'\b\w{3,}\b', text.lower())

    def _nav(self, url: str, reason: str = "") -> WebAction:
        if self.verbose:
            logger.info("[RuleAgent] NAVIGATE → {} | {}", url[:60], reason)
        return WebAction(action_type="navigate", target=url)

    def _action(self, action_type: ActionType, target: str,
                value: str, reason: str = "") -> WebAction:
        if self.verbose:
            logger.info("[RuleAgent] {} → {} | {}", action_type.value, target[:50], reason)
        return WebAction(action_type=action_type.value, target=target, value=value)
