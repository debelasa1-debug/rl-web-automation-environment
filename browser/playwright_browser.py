"""
browser/playwright_browser.py
Playwright-based browser automation layer for the RL Web Environment.
Provides a clean async API for all browser interactions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from loguru import logger
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)
from bs4 import BeautifulSoup


@dataclass
class BrowserConfig:
    headless: bool = True
    slow_mo: int = 50          # ms between actions (for visibility)
    timeout: int = 10_000      # default action timeout in ms
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class PageState:
    """Snapshot of the current page state."""
    url: str = ""
    title: str = ""
    text_content: str = ""
    interactive_elements: list[dict] = field(default_factory=list)
    html_snapshot: str = ""
    screenshot_path: Optional[str] = None
    load_time_ms: float = 0.0
    error: Optional[str] = None


class PlaywrightBrowser:
    """
    Async Playwright wrapper providing a structured action API
    consumed by the RL environment.
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._action_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the browser and create an initial page."""
        logger.info("Starting Playwright browser (headless={})", self.config.headless)
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        await self._new_context()
        logger.success("Browser started successfully")

    async def _new_context(self) -> None:
        self._context = await self._browser.new_context(
            viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
            user_agent=self.config.user_agent,
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.config.timeout)

    async def stop(self) -> None:
        """Close the browser and release resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("Browser stopped")
        except Exception as exc:
            logger.warning("Error during browser shutdown: {}", exc)

    async def reset(self) -> None:
        """Hard reset: close context and open a fresh page."""
        logger.debug("Resetting browser context")
        if self._context:
            await self._context.close()
        await self._new_context()
        self._action_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> dict:
        """Navigate to a URL and return result metadata."""
        t0 = time.time()
        try:
            response = await self._page.goto(url, wait_until="domcontentloaded")
            elapsed = (time.time() - t0) * 1000
            status = response.status if response else 0
            success = status < 400
            self._action_count += 1
            logger.debug("navigate({}) → status={} ({:.0f}ms)", url, status, elapsed)
            return {"success": success, "status": status, "elapsed_ms": elapsed, "url": self._page.url}
        except PlaywrightTimeoutError:
            self._error_count += 1
            logger.warning("navigate({}) timed out", url)
            return {"success": False, "error": "timeout", "url": url}
        except Exception as exc:
            self._error_count += 1
            logger.error("navigate({}) error: {}", url, exc)
            return {"success": False, "error": str(exc), "url": url}

    async def click(self, selector: str) -> dict:
        """Click an element identified by a CSS selector."""
        try:
            await self._page.wait_for_selector(selector, state="visible", timeout=5000)
            await self._page.click(selector)
            self._action_count += 1
            logger.debug("click({})", selector)
            return {"success": True, "selector": selector}
        except PlaywrightTimeoutError:
            self._error_count += 1
            logger.warning("click({}) – element not found/visible", selector)
            return {"success": False, "error": "element_not_found", "selector": selector}
        except Exception as exc:
            self._error_count += 1
            logger.warning("click({}) error: {}", selector, exc)
            return {"success": False, "error": str(exc), "selector": selector}

    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> dict:
        """Type text into an input field."""
        try:
            await self._page.wait_for_selector(selector, state="visible", timeout=5000)
            if clear_first:
                await self._page.fill(selector, "")
            await self._page.type(selector, text, delay=30)
            self._action_count += 1
            logger.debug("type({}, '{}')", selector, text[:40])
            return {"success": True, "selector": selector, "text": text}
        except PlaywrightTimeoutError:
            self._error_count += 1
            return {"success": False, "error": "element_not_found", "selector": selector}
        except Exception as exc:
            self._error_count += 1
            return {"success": False, "error": str(exc), "selector": selector}

    async def scroll(self, direction: str = "down", amount: int = 300) -> dict:
        """Scroll the page."""
        try:
            delta = amount if direction == "down" else -amount
            await self._page.evaluate(f"window.scrollBy(0, {delta})")
            self._action_count += 1
            logger.debug("scroll({}, {}px)", direction, amount)
            return {"success": True, "direction": direction, "amount": amount}
        except Exception as exc:
            self._error_count += 1
            return {"success": False, "error": str(exc)}

    async def press_key(self, key: str) -> dict:
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        try:
            await self._page.keyboard.press(key)
            self._action_count += 1
            logger.debug("press_key({})", key)
            return {"success": True, "key": key}
        except Exception as exc:
            self._error_count += 1
            return {"success": False, "error": str(exc)}

    async def submit_form(self, selector: str) -> dict:
        """Submit a form by pressing Enter or clicking submit."""
        try:
            await self._page.press(selector, "Enter")
            await self._page.wait_for_load_state("domcontentloaded", timeout=8000)
            self._action_count += 1
            logger.debug("submit_form({})", selector)
            return {"success": True, "selector": selector}
        except Exception as exc:
            self._error_count += 1
            return {"success": False, "error": str(exc)}

    async def extract_text(self, selector: str = "body") -> dict:
        """Extract visible text from a selector."""
        try:
            text = await self._page.inner_text(selector)
            return {"success": True, "text": text.strip(), "selector": selector}
        except Exception as exc:
            return {"success": False, "error": str(exc), "text": ""}

    async def get_attribute(self, selector: str, attribute: str) -> dict:
        """Get an attribute value from an element."""
        try:
            value = await self._page.get_attribute(selector, attribute)
            return {"success": True, "value": value}
        except Exception as exc:
            return {"success": False, "error": str(exc), "value": None}

    # ------------------------------------------------------------------
    # State Observation
    # ------------------------------------------------------------------

    async def get_page_state(self, screenshot_path: Optional[str] = None) -> PageState:
        """Capture a full snapshot of the current page state."""
        t0 = time.time()
        try:
            url = self._page.url
            title = await self._page.title()

            # Full visible text via inner_text on body
            try:
                text_content = await self._page.inner_text("body")
            except Exception:
                text_content = ""

            # HTML for element extraction
            try:
                html = await self._page.content()
            except Exception:
                html = ""

            interactive = await self._extract_interactive_elements()

            shot_path = None
            if screenshot_path:
                try:
                    await self._page.screenshot(path=screenshot_path, full_page=False)
                    shot_path = screenshot_path
                except Exception as exc:
                    logger.debug("Screenshot failed: {}", exc)

            elapsed = (time.time() - t0) * 1000
            return PageState(
                url=url,
                title=title,
                text_content=text_content[:4000],   # cap for observation space
                interactive_elements=interactive[:50],
                html_snapshot=html[:8000],
                screenshot_path=shot_path,
                load_time_ms=elapsed,
            )
        except Exception as exc:
            logger.error("get_page_state error: {}", exc)
            return PageState(error=str(exc))

    async def _extract_interactive_elements(self) -> list[dict]:
        """
        Return a structured list of interactive elements visible on the page.
        Each element has: tag, selector, text, type, placeholder, href.
        """
        try:
            elements = await self._page.evaluate("""
                () => {
                    const els = [];
                    const seen = new Set();
                    const selectors = [
                        'a[href]', 'button', 'input', 'textarea',
                        'select', '[role="button"]', '[onclick]',
                        'label', 'h1', 'h2', 'h3'
                    ];
                    document.querySelectorAll(selectors.join(',')).forEach((el, idx) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return;
                        const key = el.tagName + el.textContent.trim().slice(0,30);
                        if (seen.has(key)) return;
                        seen.add(key);
                        els.push({
                            tag: el.tagName.toLowerCase(),
                            text: el.textContent.trim().slice(0, 80),
                            type: el.getAttribute('type') || '',
                            placeholder: el.getAttribute('placeholder') || '',
                            href: el.getAttribute('href') || '',
                            id: el.id || '',
                            name: el.getAttribute('name') || '',
                            index: els.length
                        });
                    });
                    return els;
                }
            """)
            return elements or []
        except Exception as exc:
            logger.debug("Element extraction error: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def wait_for_navigation(self, timeout_ms: int = 8000) -> bool:
        try:
            await self._page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return True
        except Exception:
            return False

    async def current_url(self) -> str:
        return self._page.url if self._page else ""

    async def page_contains_text(self, text: str) -> bool:
        """Check whether the page contains a given substring (case-insensitive)."""
        try:
            content = await self._page.inner_text("body")
            return text.lower() in content.lower()
        except Exception:
            return False

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def error_count(self) -> int:
        return self._error_count
