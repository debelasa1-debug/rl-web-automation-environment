"""
main.py
Entry point for the RL Web Automation Environment.

Usage examples
──────────────
# Run rule-based agent on all built-in tasks (1 episode each):
    python main.py --agent rule --episodes 10

# Run LLM agent on a specific task:
    python main.py --agent llm --task-id nav_001 --episodes 3

# Run on specific category and difficulty:
    python main.py --agent rule --category search --difficulty easy --episodes 5

# Visible browser (for debugging):
    python main.py --agent rule --headless false --episodes 2

# Full benchmark run:
    python main.py --agent rule --benchmark --episodes 22
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agents.llm_agent import LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from env.task_manager import TaskCategory, TaskDifficulty, TaskManager
from env.web_env import EnvConfig, WebAutomationEnv
from eval.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Logging setup
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
# Core RL loop
# ---------------------------------------------------------------------------

async def run_episode(
    env: WebAutomationEnv,
    agent,
    evaluator: Evaluator,
    task_options: dict,
) -> dict:
    """Run a single episode and return result metadata."""
    evaluator.start_episode()

    # Reset environment
    obs, info = await env.reset(options=task_options)
    task = env.current_task
    agent.reset()

    logger.info("━" * 60)
    logger.info("Task: {}", task.instruction)

    done = False
    truncated = False
    total_reward = 0.0
    step = 0
    final_url = obs.get("current_url", "")

    while not done and not truncated:
        # Agent selects action
        action = agent.act(obs, task)

        # Environment executes action
        obs, reward, done, truncated, info = await env.step(action)

        total_reward += reward
        step += 1
        final_url = obs.get("current_url", "")

        logger.debug(
            "  step={:3d} reward={:+.1f} total={:+.1f} | {}",
            step, reward, total_reward, info.get("reward_reason", "")
        )

    # Record episode
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
    """Main evaluation loop."""
    configure_logging(args.log_level)

    # Directories
    log_dir = Path("logs")
    screenshot_dir = Path("logs/screenshots") if args.screenshots else None

    # Task manager
    task_mgr = TaskManager(seed=args.seed)

    # Environment
    env_config = EnvConfig(
        headless=args.headless,
        slow_mo=args.slow_mo,
        timeout_ms=args.timeout,
        screenshot_dir=screenshot_dir,
        seed=args.seed,
    )
    env = WebAutomationEnv(task_manager=task_mgr, config=env_config)
    await env.async_init()

    # Agent
    if args.agent == "llm":
        agent = LLMAgent(
            model=args.model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            fallback_to_rules=True,
        )
        agent_label = f"llm:{args.model}"
    else:
        agent = RuleBasedAgent(verbose=args.verbose)
        agent_label = "rule_based"

    # Evaluator
    evaluator = Evaluator(output_dir=log_dir, agent_type=agent_label)

    # Build task options for each episode
    if args.benchmark:
        # Run every built-in task once
        all_tasks = task_mgr.all_tasks()
        episodes_plan = [{"task_id": t.task_id} for t in all_tasks]
        # If more episodes than tasks, cycle
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

    # Save results
    json_path = evaluator.save_json("results.json")
    csv_path = evaluator.save_csv("episodes.csv")
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
    p.add_argument("--agent", choices=["rule", "llm"], default="rule",
                   help="Agent type (default: rule)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="LLM model name (default: claude-sonnet-4-6)")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to run (default: 5)")
    p.add_argument("--task-id", default=None,
                   help="Run a specific task by ID")
    p.add_argument("--category", default=None,
                   choices=["navigation", "search", "form_filling", "extraction", "multi_step"],
                   help="Filter tasks by category")
    p.add_argument("--difficulty", default=None,
                   choices=["easy", "medium", "hard"],
                   help="Filter tasks by difficulty")
    p.add_argument("--benchmark", action="store_true",
                   help="Run one episode per built-in task")
    p.add_argument("--headless", type=lambda x: x.lower() != "false", default=True,
                   help="Run browser headless (default: true)")
    p.add_argument("--slow-mo", type=int, default=50,
                   help="Milliseconds between browser actions (default: 50)")
    p.add_argument("--timeout", type=int, default=10000,
                   help="Browser action timeout in ms (default: 10000)")
    p.add_argument("--screenshots", action="store_true",
                   help="Save screenshots at each step")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: INFO)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose agent output")
    p.add_argument("--debug", action="store_true",
                   help="Re-raise exceptions for debugging")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(run_evaluation(args))
