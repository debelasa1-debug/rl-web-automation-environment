"""
tests/test_integration.py
End-to-end integration tests for the complete RL loop.
Uses a mock browser so no live internet connection is needed.
Validates that reset → step → reward → done flow works correctly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser.playwright_browser import PageState
from env.action_space import ActionType, WebAction
from env.observation import ObservationBuilder
from env.reward import RewardEngine
from env.task_manager import SuccessCriteria, Task, TaskCategory, TaskDifficulty, TaskManager
from env.web_env import EnvConfig, WebAutomationEnv
from agents.rule_based_agent import RuleBasedAgent
from eval.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_task(
    task_id: str = "test_nav",
    url_contains: str = "docs.python.org",
    text: str | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        category=TaskCategory.NAVIGATION,
        difficulty=TaskDifficulty.EASY,
        instruction="Navigate to the Python documentation homepage.",
        start_url="https://www.python.org",
        success_criteria=SuccessCriteria(
            url_contains=url_contains,
            page_contains_text=[text] if text else None,
        ),
        max_steps=10,
    )


def make_page_state(url: str = "https://www.python.org", text: str = "Welcome") -> PageState:
    return PageState(
        url=url,
        title="Python",
        text_content=text,
        interactive_elements=[
            {"tag": "a", "text": "Documentation", "href": "https://docs.python.org/3/", "type": "", "placeholder": "", "id": "", "name": "", "index": 0},
            {"tag": "input", "text": "", "href": "", "type": "text", "placeholder": "Search", "id": "id-search-field", "name": "q", "index": 1},
        ],
    )


def make_mock_browser(page_state: PageState | None = None) -> MagicMock:
    """Return a mock PlaywrightBrowser pre-wired with async methods."""
    state = page_state or make_page_state()
    browser = MagicMock()
    browser.start = AsyncMock()
    browser.stop = AsyncMock()
    browser.reset = AsyncMock()
    browser.navigate = AsyncMock(return_value={"success": True, "status": 200, "url": state.url})
    browser.click = AsyncMock(return_value={"success": True})
    browser.type_text = AsyncMock(return_value={"success": True})
    browser.scroll = AsyncMock(return_value={"success": True})
    browser.press_key = AsyncMock(return_value={"success": True})
    browser.submit_form = AsyncMock(return_value={"success": True})
    browser.extract_text = AsyncMock(return_value={"success": True, "text": state.text_content})
    browser.page_contains_text = AsyncMock(return_value=True)
    browser.current_url = AsyncMock(return_value=state.url)
    browser.get_page_state = AsyncMock(return_value=state)
    browser.action_count = 0
    browser.error_count = 0
    return browser


# ---------------------------------------------------------------------------
# Test: env reset initialises correctly
# ---------------------------------------------------------------------------

class TestEnvReset:

    @pytest.mark.asyncio
    async def test_reset_returns_obs_and_info(self):
        """reset() must return (observation_dict, info_dict) with expected keys."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr, config=EnvConfig(headless=True))

        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        obs, info = await env.reset(options={"task_id": "nav_001"})

        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "current_url" in obs
        assert "task_instruction" in obs
        assert "task" in info
        assert "episode" in info

    @pytest.mark.asyncio
    async def test_reset_navigates_to_start_url(self):
        """reset() must call navigate with the task's start_url."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr, config=EnvConfig(headless=True))

        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})

        mock_browser.navigate.assert_called_once()
        call_url = mock_browser.navigate.call_args[0][0]
        assert "python.org" in call_url

    @pytest.mark.asyncio
    async def test_reset_increments_episode_count(self):
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        assert env._episode_count == 0
        await env.reset(options={"task_id": "nav_001"})
        assert env._episode_count == 1
        await env.reset(options={"task_id": "nav_002"})
        assert env._episode_count == 2


# ---------------------------------------------------------------------------
# Test: env step dispatches actions
# ---------------------------------------------------------------------------

class TestEnvStep:

    @pytest.mark.asyncio
    async def test_step_click_returns_5_tuple(self):
        """step() must return (obs, reward, terminated, truncated, info)."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})
        result = await env.step({"action_type": "click", "target": "a"})

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_invalid_action_returns_negative_reward(self):
        """A structurally invalid action dict must yield a negative reward."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        env._browser = make_mock_browser()
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})
        _, reward, _, _, info = await env.step({"action_type": "navigate", "target": ""})

        assert reward < 0
        assert "invalid" in info.get("reward_reason", "")

    @pytest.mark.asyncio
    async def test_step_navigate_action_calls_browser(self):
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})
        mock_browser.navigate.reset_mock()

        await env.step({"action_type": "navigate", "target": "https://docs.python.org"})
        mock_browser.navigate.assert_called_once_with("https://docs.python.org")

    @pytest.mark.asyncio
    async def test_step_type_action_calls_browser(self):
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "form_001"})
        await env.step({"action_type": "type", "target": "input[name='q']", "value": "test"})
        mock_browser.type_text.assert_called_once_with("input[name='q']", "test")

    @pytest.mark.asyncio
    async def test_step_scroll_action_calls_browser(self):
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        mock_browser = make_mock_browser()
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})
        await env.step({"action_type": "scroll", "value": "down"})
        mock_browser.scroll.assert_called_once_with("down")

    @pytest.mark.asyncio
    async def test_step_truncates_at_max_steps(self):
        """Episode must truncate after max_steps, even without task completion."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)

        # Use a no-op state that never matches success criteria
        mock_browser = make_mock_browser(make_page_state("https://www.python.org", "nothing useful"))
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})
        task = env.current_task
        max_steps = task.max_steps

        terminated = truncated = False
        for _ in range(max_steps + 2):
            if terminated or truncated:
                break
            _, _, terminated, truncated, _ = await env.step({"action_type": "no_op"})

        assert truncated or terminated  # must end by max_steps


# ---------------------------------------------------------------------------
# Test: task completion triggers +10 reward and terminated=True
# ---------------------------------------------------------------------------

class TestTaskCompletion:

    @pytest.mark.asyncio
    async def test_navigate_to_success_url_terminates_episode(self):
        """
        When the browser lands on a URL matching success_criteria.url_contains,
        the reward must be +10 and terminated must be True.
        """
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)

        # Simulate landing on the docs URL after navigate
        success_state = make_page_state("https://docs.python.org/3/", "Python Documentation")
        mock_browser = make_mock_browser(success_state)
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})

        obs, reward, terminated, truncated, info = await env.step({
            "action_type": "navigate",
            "target": "https://docs.python.org/3/",
        })

        assert reward == 10.0
        assert terminated is True
        assert "task_complete" in info["reward_reason"]


# ---------------------------------------------------------------------------
# Test: full episode loop with rule-based agent
# ---------------------------------------------------------------------------

class TestFullLoop:

    @pytest.mark.asyncio
    async def test_rule_agent_completes_search_task(self):
        """
        Simulate a full search episode:
        step 1: agent types in search box
        step 2: agent presses Enter
        step 3: page now matches success criteria → +10
        """
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)
        agent = RuleBasedAgent(verbose=False)

        # Phase 1: search page (has search box)
        search_page = make_page_state("https://duckduckgo.com", "Search the web")
        # Phase 2 & 3: results page (matches success criteria)
        results_page = make_page_state(
            "https://duckduckgo.com/?q=OpenAI+GPT",
            "GPT results on the page"
        )

        call_count = {"n": 0}

        async def dynamic_page_state(*args, **kwargs):
            call_count["n"] += 1
            return search_page if call_count["n"] <= 2 else results_page

        mock_browser = make_mock_browser(search_page)
        mock_browser.get_page_state = dynamic_page_state
        env._browser = mock_browser
        env._initialised = True

        obs, _ = await env.reset(options={"task_id": "search_001"})
        agent.reset()

        done = truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated and steps < 15:
            action = agent.act(obs, env.current_task)
            obs, reward, done, truncated, info = await env.step(action)
            total_reward += reward
            steps += 1

        # The episode must have run at least 1 step and accumulated some reward
        assert steps >= 1
        assert total_reward != 0.0 or done  # made progress or completed

    @pytest.mark.asyncio
    async def test_episode_stats_recorded_correctly(self):
        """EpisodeStats must accurately track reward, steps, and success."""
        task_mgr = TaskManager()
        env = WebAutomationEnv(task_manager=task_mgr)

        success_state = make_page_state("https://docs.python.org/3/", "Python docs")
        mock_browser = make_mock_browser(success_state)
        env._browser = mock_browser
        env._initialised = True

        await env.reset(options={"task_id": "nav_001"})

        # One action that directly hits success criteria
        _, reward, terminated, _, _ = await env.step({
            "action_type": "navigate",
            "target": "https://docs.python.org/3/",
        })

        stats = env.episode_stats
        assert stats is not None
        assert stats.steps == 1
        assert stats.total_reward == 10.0
        assert stats.success is True

    @pytest.mark.asyncio
    async def test_evaluator_records_episode_after_loop(self):
        """Evaluator must capture episode data and compute correct success rate."""
        task_mgr = TaskManager()
        evaluator = Evaluator(output_dir=Path("/tmp/test_eval2"), agent_type="test")
        env = WebAutomationEnv(task_manager=task_mgr)

        success_state = make_page_state("https://docs.python.org/3/", "docs")
        mock_browser = make_mock_browser(success_state)
        env._browser = mock_browser
        env._initialised = True

        for _ in range(3):
            evaluator.start_episode()
            await env.reset(options={"task_id": "nav_001"})
            await env.step({"action_type": "navigate", "target": "https://docs.python.org/3/"})
            stats = env.episode_stats
            evaluator.record_episode(
                task=env.current_task,
                success=stats.success,
                total_reward=stats.total_reward,
                steps=stats.steps,
            )

        metrics = evaluator.compute_metrics()
        assert metrics.n_episodes == 3
        assert metrics.success_rate == 1.0
        assert metrics.avg_reward == 10.0


# ---------------------------------------------------------------------------
# Test: reward edge cases
# ---------------------------------------------------------------------------

class TestRewardEdgeCases:

    @pytest.mark.asyncio
    async def test_milestone_rewards_fire_once(self):
        """A milestone reward of +2 must only fire once per milestone index."""
        from env.reward import RewardEngine
        from env.observation import WebObservation

        engine = RewardEngine()
        engine.reset()

        task = Task(
            task_id="multi_test",
            category=TaskCategory.MULTI_STEP,
            difficulty=TaskDifficulty.HARD,
            instruction="Multi-step test",
            start_url="https://en.wikipedia.org",
            success_criteria=SuccessCriteria(
                url_contains="Reinforcement_learning",
                intermediate_milestones=[{"url_contains": "Artificial_intelligence"}],
            ),
            max_steps=20,
        )

        obs = WebObservation(
            task_id="multi_test",
            task_instruction="test",
            current_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            text_content="Artificial intelligence article",
        )

        mock_browser = make_mock_browser(
            make_page_state("https://en.wikipedia.org/wiki/Artificial_intelligence", "AI")
        )
        action = {"action_type": "click", "target": "a", "value": ""}
        result = {"success": True}

        sig1 = await engine.compute(mock_browser, task, action, result, obs)
        # First time at milestone URL → should award milestone reward
        assert sig1.value == 2.0 or sig1.task_complete  # milestone or if it also matches success

        # Re-check with same observation: milestone already credited, no double reward
        sig2 = await engine.compute(mock_browser, task, action, result, obs)
        assert sig2.value != 2.0   # milestone not awarded again

    @pytest.mark.asyncio
    async def test_failed_browser_action_gives_minus_one(self):
        """Browser failures (element not found) yield -1 reward."""
        from env.reward import RewardEngine
        from env.observation import WebObservation

        engine = RewardEngine()
        engine.reset()
        task = make_task()
        obs = WebObservation(task_id="test_nav", task_instruction="test",
                             current_url="https://www.python.org")
        mock_browser = make_mock_browser()

        action = {"action_type": "click", "target": "#ghost-button", "value": ""}
        result = {"success": False, "error": "element_not_found"}

        sig = await engine.compute(mock_browser, task, action, result, obs)
        assert sig.value == -1.0
        assert sig.is_invalid is True

    @pytest.mark.asyncio
    async def test_loop_penalty_after_repeated_actions(self):
        """Repeating the same action 3+ times triggers the -3 loop penalty."""
        from env.reward import RewardEngine
        from env.observation import WebObservation

        engine = RewardEngine()
        engine.reset()
        task = make_task()
        obs = WebObservation(task_id="test_nav", task_instruction="test",
                             current_url="https://www.python.org",
                             text_content="no match here at all")
        mock_browser = make_mock_browser()

        action = {"action_type": "scroll", "target": "", "value": "down"}
        result = {"success": True}

        rewards = []
        for _ in range(5):
            sig = await engine.compute(mock_browser, task, action, result, obs)
            rewards.append(sig.value)

        # By the 4th+ repetition, loop penalty must fire
        assert -3.0 in rewards


# ---------------------------------------------------------------------------
# Test: LLM agent JSON parsing robustness
# ---------------------------------------------------------------------------

class TestLLMAgentRobustness:

    def test_parse_returns_none_for_garbage(self):
        from agents.llm_agent import LLMAgent
        assert LLMAgent._parse_json("") is None
        assert LLMAgent._parse_json("   ") is None
        assert LLMAgent._parse_json("not json at all!") is None
        assert LLMAgent._parse_json("[1, 2, 3]") is None  # list, not object

    def test_parse_handles_whitespace_around_json(self):
        from agents.llm_agent import LLMAgent
        raw = '\n\n  {"action_type": "click", "target": "a"}  \n'
        result = LLMAgent._parse_json(raw)
        assert result is not None
        assert result["action_type"] == "click"

    def test_llm_agent_falls_back_on_no_api_key(self):
        """LLMAgent with no API key must silently fall back to rule-based agent."""
        from agents.llm_agent import LLMAgent
        from env.task_manager import TaskManager

        agent = LLMAgent(api_key=None, fallback_to_rules=True, verbose=False)
        assert agent._client is None

        task = TaskManager().get_task("search_001")
        obs = {
            "current_url": "https://duckduckgo.com",
            "page_title": "DuckDuckGo",
            "text_content": "Search the web",
            "interactive_elements": [],
            "has_search_box": True,
            "has_login_form": False,
            "has_submit_button": True,
            "inputs": [{"tag": "input", "type": "text", "name": "q", "placeholder": "Search",
                        "id": "", "text": "", "index": 0}],
            "links": [], "buttons": [], "form_fields": [],
            "step_number": 1, "max_steps": 10,
            "milestone_reached": 0, "error_message": None,
        }
        agent.reset()
        action = agent.act(obs, task)
        assert action is not None
        assert action.action_type in list(ActionType)
