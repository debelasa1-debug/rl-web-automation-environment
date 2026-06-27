"""
main.py
Entry point for the RL Web Automation Environment.

Usage examples
──────────────
# Rule-based agent, 5 random episodes:
    python main.py --agent rule --episodes 5

# Gemini LLM agent (free, no credit card):
    python main.py --agent gemini --episodes 5

# Specific task, visible browser:
    python main.py --agent rule --task-id search_001 --headless false --slow-mo 400

# Benchmark all 11 tasks:
    python main.py --agent rule --benchmark

# Gemini on search tasks only:
    python main.py --agent gemini --category search --episodes 4
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from agents.llm_agent import LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from env.task_manager import TaskCategory, TaskDifficulty, TaskManager
from env.web_env import EnvConfig, WebAutomationEnv
from eval.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}:{line}</cyan> – {message}"
        ),
        colorize=True,
    )
    log_file = Path("logs") / "run.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="10 MB", retention="7 days")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(args: argparse.Namespace):
    """
    Build the agent from CLI args.

    --agent rule        → RuleBasedAgent (no API key needed)
    --agent gemini      → LLMAgent wired to Gemini 2.5 Flash (free)
    --agent groq        → LLMAgent wired to Groq Llama-3.3-70B (free)
    --agent openrouter  → LLMAgent wired to OpenRouter free tier
    --agent llm         → LLMAgent using ANTHROPIC_API_KEY (paid)
    """
    agent_type = args.agent

    if agent_type == "rule":
        logger.info("Agent: RuleBasedAgent (no API key needed)")
        return RuleBasedAgent(verbose=args.verbose), "rule_based"

    # ── Gemini (free, no credit card) ────────────────────────────────────
    if agent_type == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            logger.error(
                "GEMINI_API_KEY not set!\n"
                "  1. Get a free key at: https://aistudio.google.com\n"
                "  2. Add to your .env file:  GEMINI_API_KEY=AIza...\n"
                "  3. Or set as environment variable and rerun."
            )
            sys.exit(1)
        try:
            import openai  # noqa: F401
        except ImportError:
            logger.error("openai package required for Gemini agent. Run:  pip install openai")
            sys.exit(1)
        model = args.model if args.model != "claude-sonnet-4-6" else "gemini-2.5-flash"
        agent = LLMAgent(
            model=model,
            api_key=key,
            fallback_to_rules=True,
            verbose=args.verbose,
        )
        # Override to Gemini endpoint (LLMAgent auto-detects GEMINI_API_KEY,
        # but we also force it here so --model works correctly)
        try:
            import openai as _oa
            agent._client = _oa.OpenAI(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            agent._is_openai_compat = True
            agent.model = model
        except Exception as e:
            logger.warning("Gemini client setup warning: {}", e)
        logger.info("Agent: Gemini LLM (model={})", model)
        return agent, f"gemini:{model}"

    # ── Groq (free, 1 000 req/day) ───────────────────────────────────────
    if agent_type == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            logger.error(
                "GROQ_API_KEY not set!\n"
                "  1. Get a free key at: https://console.groq.com\n"
                "  2. Add to your .env file:  GROQ_API_KEY=gsk_...\n"
            )
            sys.exit(1)
        model = args.model if args.model != "claude-sonnet-4-6" else "llama-3.3-70b-versatile"
        import openai as _oa
        agent = LLMAgent(model=model, api_key=key, fallback_to_rules=True,
                         verbose=args.verbose)
        agent._client = _oa.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        agent._is_openai_compat = True
        agent.model = model
        logger.info("Agent: Groq LLM (model={})", model)
        return agent, f"groq:{model}"

    # ── OpenRouter (free tier) ───────────────────────────────────────────
    if agent_type == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            logger.error(
                "OPENROUTER_API_KEY not set!\n"
                "  1. Get a free key at: https://openrouter.ai\n"
                "  2. Add to your .env file:  OPENROUTER_API_KEY=sk-or-...\n"
            )
            sys.exit(1)
        model = args.model if args.model != "claude-sonnet-4-6" \
            else "meta-llama/llama-3.3-70b-instruct:free"
        import openai as _oa
        agent = LLMAgent(model=model, api_key=key, fallback_to_rules=True,
                         verbose=args.verbose)
        agent._client = _oa.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        agent._is_openai_compat = True
        agent.model = model
        logger.info("Agent: OpenRouter LLM (model={})", model)
        return agent, f"openrouter:{model}"

    # ── Anthropic Claude (paid) ──────────────────────────────────────────
    if agent_type == "llm":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — LLMAgent will fall back to rule-based.\n"
                "  For free alternatives use:  --agent gemini  or  --agent groq"
            )
        model = args.model
        agent = LLMAgent(model=model, api_key=key, fallback_to_rules=True,
                         verbose=args.verbose)
        logger.info("Agent: Claude LLM (model={})", model)
        return agent, f"llm:{model}"

    logger.error("Unknown agent type: {}", agent_type)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core RL loop
# ---------------------------------------------------------------------------

async def run_episode(
    env: WebAutomationEnv,
    agent,
    evaluator: Evaluator,
    task_options: dict,
) -> dict:
    evaluator.start_episode()

    obs, info = await env.reset(options=task_options)
    task = env.current_task
    agent.reset()

    logger.info("━" * 60)
    logger.info("Task: {}", task.instruction)

    done = truncated = False
    total_reward = 0.0
    step = 0
    final_url = obs.get("current_url", "")

    while not done and not truncated:
        action = agent.act(obs, task)
        obs, reward, done, truncated, info = await env.step(action)
        total_reward += reward
        step += 1
        final_url = obs.get("current_url", "")
        logger.debug(
            "  step={:3d} reward={:+.1f} total={:+.1f} | {}",
            step, reward, total_reward, info.get("reward_reason", "")
        )

    stats = env.episode_stats
    record = evaluator.record_episode(
        task=task,
        success=stats.success,
        total_reward=stats.total_reward,
        steps=stats.steps,
        reward_breakdown=dict(zip(stats.reasons, stats.rewards)),
        milestones_hit=stats.milestones_hit,
        final_url=final_url,
    )
    return record.to_dict()


async def run_evaluation(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    log_dir = Path("logs")
    screenshot_dir = Path("logs/screenshots") if args.screenshots else None

    task_mgr = TaskManager(seed=args.seed)

    env_config = EnvConfig(
        headless=args.headless,
        slow_mo=args.slow_mo,
        timeout_ms=args.timeout,
        screenshot_dir=screenshot_dir,
        seed=args.seed,
    )
    env = WebAutomationEnv(task_manager=task_mgr, config=env_config)
    await env.async_init()

    agent, agent_label = build_agent(args)
    evaluator = Evaluator(output_dir=log_dir, agent_type=agent_label)

    # Build episode plan
    if args.benchmark:
        all_tasks = task_mgr.all_tasks()
        episodes_plan = [{"task_id": t.task_id} for t in all_tasks]
        while len(episodes_plan) < args.episodes:
            episodes_plan += [{"task_id": t.task_id} for t in all_tasks]
        episodes_plan = episodes_plan[:args.episodes]
    else:
        task_options: dict = {}
        if args.task_id:
            task_options["task_id"] = args.task_id
        if args.category:
            task_options["category"] = TaskCategory(args.category)
        if args.difficulty:
            task_options["difficulty"] = TaskDifficulty(args.difficulty)
        episodes_plan = [task_options] * args.episodes

    logger.info("Starting {} episodes with agent={}", len(episodes_plan), agent_label)

    results = []
    for i, task_opts in enumerate(episodes_plan):
        logger.info("\n[Episode {}/{}]", i + 1, len(episodes_plan))
        try:
            result = await run_episode(env, agent, evaluator, task_opts)
            results.append(result)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            break
        except Exception as exc:
            logger.error("Episode {} failed with error: {}", i + 1, exc)
            if args.debug:
                raise

    json_path = evaluator.save_json("results.json")
    csv_path  = evaluator.save_csv("episodes.csv")
    evaluator.print_summary()

    logger.info("Results saved:")
    logger.info("  JSON: {}", json_path)
    logger.info("  CSV : {}", csv_path)

    await env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL Web Automation Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--agent",
        choices=["rule", "gemini", "groq", "openrouter", "llm"],
        default="rule",
        help=(
            "Agent type:\n"
            "  rule       — deterministic heuristics, no API key\n"
            "  gemini     — Google Gemini 2.5 Flash (free, needs GEMINI_API_KEY)\n"
            "  groq       — Groq Llama-3.3-70B (free, needs GROQ_API_KEY)\n"
            "  openrouter — OpenRouter free tier (needs OPENROUTER_API_KEY)\n"
            "  llm        — Anthropic Claude (paid, needs ANTHROPIC_API_KEY)"
        ),
    )
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Override LLM model name (default per provider otherwise)")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to run (default: 5)")
    p.add_argument("--task-id", default=None,
                   help="Run a specific task by ID")
    p.add_argument("--category", default=None,
                   choices=["navigation", "search", "form_filling",
                            "extraction", "multi_step"],
                   help="Filter tasks by category")
    p.add_argument("--difficulty", default=None,
                   choices=["easy", "medium", "hard"],
                   help="Filter tasks by difficulty")
    p.add_argument("--benchmark", action="store_true",
                   help="Run one episode per built-in task")
    p.add_argument("--headless",
                   type=lambda x: x.lower() != "false", default=True,
                   help="Headless browser (default: true). Use --headless false to watch")
    p.add_argument("--slow-mo", type=int, default=50,
                   help="Milliseconds between browser actions (default: 50)")
    p.add_argument("--timeout", type=int, default=15000,
                   help="Browser action timeout ms (default: 15000)")
    p.add_argument("--screenshots", action="store_true",
                   help="Save a screenshot at each step to logs/screenshots/")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for task sampling (default: 42)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: INFO)")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose agent action logging")
    p.add_argument("--debug", action="store_true",
                   help="Re-raise exceptions instead of logging them")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(run_evaluation(args))
