# RL Web Automation Environment

A production-grade **Reinforcement Learning environment** for web automation tasks, built on [Gymnasium](https://gymnasium.farama.org/) and [Playwright](https://playwright.dev/python/).

An AI agent interacts with a live browser to complete structured web tasks, receiving structured observations, deterministic rewards, and rich feedback at every step.

---

## Architecture

```
rl_web_env/
├── browser/
│   ├── __init__.py
│   └── playwright_browser.py    # Async Playwright wrapper (navigate, click, type, scroll, …)
├── env/
│   ├── __init__.py
│   ├── web_env.py               # Gymnasium-compatible RL environment (reset / step)
│   ├── task_manager.py          # Task library + sampling (11 built-in tasks)
│   ├── observation.py           # Page state → structured WebObservation
│   ├── reward.py                # Deterministic reward engine + EpisodeStats
│   └── action_space.py          # Typed action schema + validation (Pydantic)
├── agents/
│   ├── __init__.py
│   ├── rule_based_agent.py      # Heuristic rule-based baseline agent
│   └── llm_agent.py             # Claude-powered LLM agent with rule fallback
├── eval/
│   ├── __init__.py
│   └── evaluator.py             # Episode recorder + JSON/CSV export + summary
├── tests/
│   └── test_env.py              # 33 unit tests (no browser required)
├── main.py                      # CLI entry point / training loop
├── Dockerfile                   # Containerised execution
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Core Design

### RL Loop

```
Agent observes state
     ↓
Agent selects action (JSON)
     ↓
Environment validates action
     ↓
Browser executes action (Playwright)
     ↓
Page state captured → WebObservation built
     ↓
RewardEngine verifies success criteria → reward signal
     ↓
(observation, reward, terminated, truncated, info) returned
     ↓
Loop until done or max_steps
```

### Gymnasium API

```python
env = WebAutomationEnv(task_manager=task_mgr, config=EnvConfig(headless=True))
await env.async_init()

obs, info = await env.reset()               # starts episode, navigates to task URL
obs, reward, done, truncated, info = await env.step(action_dict)
await env.close()
```

---

## Observation Space

Each step returns a `WebObservation` dictionary:

| Field | Type | Description |
|---|---|---|
| `task_instruction` | str | Natural-language task description |
| `current_url` | str | Current browser URL |
| `page_title` | str | Page `<title>` |
| `text_content` | str | Visible page text (max 3 000 chars) |
| `interactive_elements` | list[dict] | Up to 50 buttons, inputs, links |
| `has_search_box` | bool | Derived feature |
| `has_login_form` | bool | Derived feature |
| `links` / `buttons` / `inputs` | list[dict] | Partitioned element lists |
| `step_number` | int | Current step |
| `milestone_reached` | int | Highest intermediate milestone satisfied |

---

## Action Space

Actions are structured JSON dictionaries validated by Pydantic:

```json
{ "action_type": "navigate", "target": "https://example.com" }
{ "action_type": "click",    "target": "button[type='submit']" }
{ "action_type": "type",     "target": "input[name='q']", "value": "search query" }
{ "action_type": "scroll",   "value": "down" }
{ "action_type": "press_key","value": "Enter" }
{ "action_type": "submit",   "target": "form" }
{ "action_type": "extract",  "target": "h1" }
{ "action_type": "wait",     "value": "2" }
{ "action_type": "no_op" }
```

---

## Reward Schedule

| Event | Reward |
|---|---|
| Task successfully completed | **+10** |
| New intermediate milestone | **+2** |
| Meaningful progress detected | **+1** |
| Neutral action | **0** |
| Invalid / failed action | **−1** |
| Repeated same action (loop) | **−3** |
| Hard failure / timeout | **−5** |

Rewards are **deterministic** — based on actual page verification (URL matching, text presence, custom verifiers) — not heuristics alone.

---

## Task Library

11 built-in tasks across 5 categories and 3 difficulty levels:

| ID | Category | Difficulty | Description |
|---|---|---|---|
| `nav_001` | navigation | easy | Navigate to Python docs |
| `nav_002` | navigation | easy | Wikipedia random article |
| `nav_003` | navigation | medium | Navigate to RL article on Wikipedia |
| `search_001` | search | easy | DuckDuckGo search for OpenAI GPT |
| `search_002` | search | medium | Wikipedia search for Deep Learning |
| `extract_001` | extraction | easy | Verify httpbin.org JSON response |
| `extract_002` | extraction | medium | Find Einstein quote on quotes.toscrape.com |
| `form_001` | form_filling | easy | DuckDuckGo search form |
| `form_002` | form_filling | medium | httpbin.org pizza order form |
| `multi_001` | multi_step | hard | Wikipedia AI article → See also |
| `multi_002` | multi_step | hard | Login to quotes.toscrape.com |

Each task defines:
- `instruction` — natural-language goal
- `start_url` — where the episode begins
- `success_criteria` — URL/text conditions for completion
- `intermediate_milestones` — optional step-wise checkpoints
- `max_steps` — episode step limit

---

## Agents

### Rule-Based Agent (baseline)

Deterministic heuristics, no external dependencies:
1. Navigate toward goal URL if known
2. Fill search box with task keywords
3. Submit search with Enter
4. Handle login forms
5. Click highest-scoring relevant link
6. Scroll to reveal more content
7. No-op fallback

### LLM Agent (Claude-powered)

Uses the Anthropic Claude API to select actions from the formatted observation:
- Sends `WebObservation.to_llm_prompt()` to Claude
- Parses JSON action from response
- Falls back to rule-based agent on parse failure or API error
- Maintains short conversation history for context

---

## Setup

### Local (Python 3.11+)

```bash
git clone <repo>
cd rl_web_env
pip install -r requirements.txt
playwright install chromium

# For LLM agent
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=...
```

### Docker

```bash
docker build -t rl-web-env .
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  rl-web-env \
  python main.py --agent llm --episodes 5
```

---

## Usage

### Run rule-based agent (5 episodes, random tasks)
```bash
python main.py --agent rule --episodes 5
```

### Run LLM agent on a specific task
```bash
python main.py --agent llm --task-id search_001 --episodes 3
```

### Benchmark all tasks (1 episode each)
```bash
python main.py --agent rule --benchmark
```

### Filter by category and difficulty
```bash
python main.py --agent rule --category search --difficulty easy --episodes 4
```

### Visible browser (for debugging)
```bash
python main.py --agent rule --headless false --episodes 2 --slow-mo 200
```

### Full CLI options
```
--agent        rule | llm               Agent type
--model        claude-sonnet-4-6        LLM model name
--episodes     N                        Number of episodes
--task-id      nav_001                  Run specific task
--category     navigation|search|...   Filter by category
--difficulty   easy|medium|hard         Filter by difficulty
--benchmark                            One episode per task
--headless     true|false              Browser visibility
--slow-mo      N                        ms between actions
--timeout      N                        Action timeout (ms)
--screenshots                          Save step screenshots
--seed         N                        Reproducibility seed
--log-level    DEBUG|INFO|WARNING       Logging verbosity
```

---

## Evaluation Output

Results are saved to `logs/`:

**`results.json`**
```json
{
  "summary": {
    "n_episodes": 10,
    "success_rate": 0.7,
    "avg_reward": 5.4,
    "avg_steps": 8.2,
    "by_category": { "search": { "success_rate": 0.9, ... } },
    "by_difficulty": { "easy": { "success_rate": 0.95 }, ... }
  },
  "episodes": [ { "episode_id": 1, "task_id": "nav_001", ... } ]
}
```

**`episodes.csv`** — one row per episode for spreadsheet analysis.

**Console summary** — printed after the run:
```
╔══════════════════════════════════════════════════════╗
║         RL WEB AUTOMATION — EVALUATION SUMMARY       ║
╚══════════════════════════════════════════════════════╝
Agent        : rule_based
  Episodes     : 5
  Success Rate : 80.0%
  Avg Reward   : +8.00  (σ=7.45)
  Avg Steps    : 4.0     (σ=4.64)
  Avg Duration : 18.2s

  ── By Category ──
  multi_step          : success=100%  avg_reward=+14.0  n=1
  navigation          : success=100%  avg_reward=+10.5  n=2
  search              : success=50%  avg_reward=+2.5  n=2

  ── By Difficulty ──
  easy        : success=67%  n=3
  hard        : success=100%  n=1
  medium      : success=100%  n=1

  ── Per Task ──
  multi_002      : success=100%  avg_reward=+14.0  avg_steps=4  n=1
  nav_001        : success=100%  avg_reward=+10.0  avg_steps=1  n=1
  nav_002        : success=100%  avg_reward=+11.0  avg_steps=2  n=1
  search_001     : success=0%  avg_reward=-5.0  avg_steps=12  n=1
  search_002     : success=100%  avg_reward=+10.0  avg_steps=1  n=1

11:12:32 | INFO     | eval.evaluator:237 – CSV saved → logs\episodes.csv
11:12:32 | INFO     | __main__:287 – Results saved:
11:12:32 | INFO     | __main__:288 –   JSON: logs\results.json
11:12:32 | INFO     | __main__:289 –   CSV : logs\episodes.csv
11:12:36 | INFO     | browser.playwright_browser:97 – Browser stopped
11:12:36 | INFO     | env.web_env:137 – WebAutomationEnv closed

Process finished with exit code 0


## Running Tests

```bash
pytest tests/test_env.py -v
```

33 unit tests covering action space, task manager, observations, reward engine, evaluator, LLM JSON parsing, and rule-based agent — all run without a live browser.

---

## Extending the System

### Add a custom task

```python
from env.task_manager import Task, TaskCategory, TaskDifficulty, SuccessCriteria

task = Task(
    task_id="my_task_001",
    category=TaskCategory.EXTRACTION,
    difficulty=TaskDifficulty.MEDIUM,
    instruction="Navigate to example.com and confirm the page title is 'Example Domain'.",
    start_url="https://example.com",
    success_criteria=SuccessCriteria(
        url_contains="example.com",
        page_contains_text=["Example Domain"],
    ),
    max_steps=5,
)
task_manager.add_task(task)
```

### Implement a custom agent

```python
class MyAgent:
    def reset(self): ...

    def act(self, observation: dict, task: Task) -> WebAction:
        # observation is the full WebObservation.to_dict()
        # return a WebAction
        return WebAction(action_type="click", target="a.my-link")
```

### Add a custom verifier

```python
async def verify_logged_in(browser) -> bool:
    return await browser.page_contains_text("Welcome back")

task = Task(
    ...
    success_criteria=SuccessCriteria(custom_verifier=verify_logged_in),
)
```

---

## Requirements

- Python ≥ 3.11
- Playwright Chromium (installed via `playwright install chromium`)
- gymnasium, pydantic, loguru, beautifulsoup4, python-dotenv
- anthropic (optional, for LLM agent)

See `requirements.txt` for full list.
"# rl-web-automation-environment" 
