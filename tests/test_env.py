"""
tests/test_env.py
Lightweight test suite for the RL Web Automation Environment.
Runs without a live browser for most tests using mocks.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Action space tests (no browser needed)
# ---------------------------------------------------------------------------

class TestActionSpace:
    def setup_method(self):
        from env.action_space import ActionSpace, ActionType, WebAction
        self.space = ActionSpace()
        self.ActionType = ActionType
        self.WebAction = WebAction

    def test_valid_navigate(self):
        action, err = self.space.validate({
            "action_type": "navigate",
            "target": "https://example.com"
        })
        assert action is not None
        assert err is None
        assert action.action_type == self.ActionType.NAVIGATE

    def test_valid_click(self):
        action, err = self.space.validate({
            "action_type": "click",
            "target": "button.submit"
        })
        assert action is not None
        assert action.action_type == self.ActionType.CLICK

    def test_valid_type(self):
        action, err = self.space.validate({
            "action_type": "type",
            "target": "input[name='q']",
            "value": "test query"
        })
        assert action is not None
        assert action.value == "test query"

    def test_invalid_navigate_no_target(self):
        action, err = self.space.validate({
            "action_type": "navigate",
            "target": ""
        })
        assert action is None
        assert err is not None

    def test_valid_scroll(self):
        action, err = self.space.validate({
            "action_type": "scroll",
            "value": "down"
        })
        assert action is not None

    def test_no_op(self):
        action, err = self.space.validate({"action_type": "no_op"})
        assert action is not None
        assert err is None

    def test_invalid_action_type(self):
        action, err = self.space.validate({"action_type": "teleport"})
        assert action is None

    def test_to_dict_roundtrip(self):
        orig = {"action_type": "click", "target": "a.link", "value": ""}
        action, _ = self.space.validate(orig)
        d = action.to_dict()
        assert d["action_type"] == "click"
        assert d["target"] == "a.link"

    def test_sample_random(self):
        action = self.space.sample_random()
        assert action is not None
        assert action.action_type in list(self.ActionType)


# ---------------------------------------------------------------------------
# Task manager tests
# ---------------------------------------------------------------------------

class TestTaskManager:
    def setup_method(self):
        from env.task_manager import TaskManager, TaskCategory, TaskDifficulty
        self.mgr = TaskManager(seed=42)
        self.TaskCategory = TaskCategory
        self.TaskDifficulty = TaskDifficulty

    def test_builtin_tasks_loaded(self):
        tasks = self.mgr.all_tasks()
        assert len(tasks) > 0

    def test_get_existing_task(self):
        task = self.mgr.get_task("nav_001")
        assert task is not None
        assert task.task_id == "nav_001"

    def test_get_nonexistent_task(self):
        task = self.mgr.get_task("does_not_exist")
        assert task is None

    def test_sample_task(self):
        task = self.mgr.sample_task()
        assert task is not None
        assert task.task_id

    def test_sample_by_category(self):
        task = self.mgr.sample_task(category=self.TaskCategory.SEARCH)
        assert task.category == self.TaskCategory.SEARCH

    def test_sample_by_difficulty(self):
        task = self.mgr.sample_task(difficulty=self.TaskDifficulty.EASY)
        assert task.difficulty == self.TaskDifficulty.EASY

    def test_record_and_success_rate(self):
        self.mgr.record_episode("nav_001", True, 8.0, 5)
        self.mgr.record_episode("nav_001", False, -2.0, 10)
        rate = self.mgr.success_rate("nav_001")
        assert rate == pytest.approx(0.5)

    def test_task_to_dict(self):
        task = self.mgr.get_task("nav_001")
        d = task.to_dict()
        assert "task_id" in d
        assert "instruction" in d
        assert "start_url" in d


# ---------------------------------------------------------------------------
# Observation builder tests
# ---------------------------------------------------------------------------

class TestObservationBuilder:
    def setup_method(self):
        from env.observation import ObservationBuilder
        from browser.playwright_browser import PageState
        from env.task_manager import TaskManager
        self.builder = ObservationBuilder()
        self.PageState = PageState
        self.task = TaskManager().get_task("nav_001")

    def test_build_basic_observation(self):
        state = self.PageState(
            url="https://python.org",
            title="Python",
            text_content="Welcome to Python documentation",
            interactive_elements=[
                {"tag": "a", "text": "Documentation", "href": "/docs/", "type": "", "placeholder": "", "id": "", "name": "", "index": 0},
                {"tag": "input", "text": "", "href": "", "type": "text", "placeholder": "Search", "id": "search", "name": "q", "index": 1},
            ]
        )
        obs = self.builder.build(state, self.task, step_number=1)
        assert obs.current_url == "https://python.org"
        assert obs.has_search_box is True
        assert obs.step_number == 1
        assert len(obs.links) >= 1

    def test_to_dict_contains_required_keys(self):
        state = self.PageState(url="https://example.com", title="Example", text_content="Hello")
        obs = self.builder.build(state, self.task, step_number=0)
        d = obs.to_dict()
        for key in ["current_url", "task_instruction", "step_number", "text_content"]:
            assert key in d

    def test_to_llm_prompt(self):
        state = self.PageState(url="https://example.com", title="Example", text_content="Hello world")
        obs = self.builder.build(state, self.task, step_number=2)
        prompt = obs.to_llm_prompt()
        assert "TASK" in prompt
        assert "https://example.com" in prompt


# ---------------------------------------------------------------------------
# Reward engine tests (mocked browser)
# ---------------------------------------------------------------------------

class TestRewardEngine:
    def setup_method(self):
        from env.reward import RewardEngine
        from env.task_manager import TaskManager
        self.engine = RewardEngine()
        self.engine.reset()
        self.task = TaskManager().get_task("search_001")

    def _make_obs(self, url="https://duckduckgo.com/?q=test", text="GPT results"):
        from env.observation import WebObservation
        return WebObservation(
            task_id="search_001",
            task_instruction="Search for OpenAI GPT",
            step_number=1,
            max_steps=10,
            current_url=url,
            text_content=text,
        )

    def _mock_browser(self):
        browser = MagicMock()
        browser.page_contains_text = AsyncMock(return_value=True)
        browser.current_url = AsyncMock(return_value="https://duckduckgo.com/?q=OpenAI+GPT")
        return browser

    @pytest.mark.asyncio
    async def test_failed_action_gives_negative_reward(self):
        browser = self._mock_browser()
        obs = self._make_obs()
        action = {"action_type": "click", "target": "#missing", "value": ""}
        result = {"success": False, "error": "element_not_found"}
        sig = await self.engine.compute(browser, self.task, action, result, obs)
        assert sig.value < 0
        assert sig.is_invalid

    @pytest.mark.asyncio
    async def test_success_gives_high_reward(self):
        browser = self._mock_browser()
        # Observation matches success criteria: url has q= and text has GPT
        obs = self._make_obs(url="https://duckduckgo.com/?q=OpenAI+GPT", text="GPT OpenAI results")
        action = {"action_type": "press_key", "target": "", "value": "Enter"}
        result = {"success": True}
        sig = await self.engine.compute(browser, self.task, action, result, obs)
        assert sig.value == 10.0
        assert sig.task_complete

    @pytest.mark.asyncio
    async def test_loop_detection(self):
        browser = self._mock_browser()
        obs = self._make_obs(url="https://duckduckgo.com", text="search page")
        action = {"action_type": "click", "target": "button", "value": ""}
        result = {"success": True}
        # Repeat the same action 4 times
        for _ in range(3):
            await self.engine.compute(browser, self.task, action, result, obs)
        sig = await self.engine.compute(browser, self.task, action, result, obs)
        assert sig.value == -3.0
        assert "loop" in sig.reason


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def setup_method(self):
        from eval.evaluator import Evaluator
        from env.task_manager import TaskManager
        self.eval = Evaluator(output_dir=Path("/tmp/test_eval"), agent_type="test")
        self.task = TaskManager().get_task("nav_001")

    def test_record_episode(self):
        self.eval.start_episode()
        record = self.eval.record_episode(
            task=self.task,
            success=True,
            total_reward=8.0,
            steps=5,
        )
        assert record.success is True
        assert record.total_reward == 8.0

    def test_compute_metrics(self):
        for success, reward in [(True, 10.0), (False, -2.0), (True, 7.0)]:
            self.eval.start_episode()
            self.eval.record_episode(self.task, success, reward, 5)
        m = self.eval.compute_metrics()
        assert m.n_episodes == 3
        assert m.success_rate == pytest.approx(2 / 3)

    def test_save_json(self):
        self.eval.start_episode()
        self.eval.record_episode(self.task, True, 10.0, 3)
        path = self.eval.save_json("test_results.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "episodes" in data


# ---------------------------------------------------------------------------
# LLM agent JSON parsing tests
# ---------------------------------------------------------------------------

class TestLLMAgentParsing:
    def setup_method(self):
        from agents.llm_agent import LLMAgent
        self.agent = LLMAgent.__new__(LLMAgent)

    def test_parse_clean_json(self):
        from agents.llm_agent import LLMAgent
        result = LLMAgent._parse_json('{"action_type": "click", "target": "a.link"}')
        assert result is not None
        assert result["action_type"] == "click"

    def test_parse_json_with_code_block(self):
        from agents.llm_agent import LLMAgent
        text = '```json\n{"action_type": "navigate", "target": "https://example.com"}\n```'
        result = LLMAgent._parse_json(text)
        assert result is not None
        assert result["action_type"] == "navigate"

    def test_parse_json_embedded_in_text(self):
        from agents.llm_agent import LLMAgent
        text = 'I will click the button. {"action_type": "click", "target": "button"}'
        result = LLMAgent._parse_json(text)
        assert result is not None

    def test_parse_invalid_returns_none(self):
        from agents.llm_agent import LLMAgent
        result = LLMAgent._parse_json("This is not JSON at all")
        assert result is None


# ---------------------------------------------------------------------------
# Rule-based agent tests
# ---------------------------------------------------------------------------

class TestRuleBasedAgent:
    def setup_method(self):
        from agents.rule_based_agent import RuleBasedAgent
        from env.task_manager import TaskManager
        self.agent = RuleBasedAgent(verbose=False)
        self.agent.reset()
        self.task = TaskManager().get_task("search_001")

    def _obs(self, url="https://duckduckgo.com", has_search=True):
        return {
            "current_url": url,
            "page_title": "DuckDuckGo",
            "text_content": "Search the web without being tracked",
            "interactive_elements": [],
            "has_search_box": has_search,
            "has_login_form": False,
            "has_submit_button": True,
            "inputs": [{"tag": "input", "type": "text", "name": "q", "placeholder": "Search", "id": "search_input", "text": "", "index": 0}],
            "links": [],
            "buttons": [{"tag": "button", "type": "submit", "text": "Search", "id": "", "name": "", "placeholder": "", "href": "", "index": 1}],
            "form_fields": [],
            "step_number": 1,
            "max_steps": 10,
            "milestone_reached": 0,
            "error_message": None,
        }

    def test_types_in_search_box_when_available(self):
        from env.action_space import ActionType
        obs = self._obs(has_search=True)
        action = self.agent.act(obs, self.task)
        assert action.action_type == ActionType.TYPE
        assert len(action.value) > 0

    def test_submits_after_typing(self):
        from env.action_space import ActionType
        obs = self._obs(has_search=True)
        # First action: type
        self.agent.act(obs, self.task)
        # Second action: submit
        action = self.agent.act(obs, self.task)
        assert action.action_type == ActionType.PRESS_KEY
        assert action.value == "Enter"

    def test_scrolls_when_no_useful_elements(self):
        from env.action_space import ActionType
        obs = self._obs(has_search=False)
        obs["links"] = []
        obs["buttons"] = []
        action = self.agent.act(obs, self.task)
        assert action.action_type in (ActionType.SCROLL, ActionType.NO_OP)
