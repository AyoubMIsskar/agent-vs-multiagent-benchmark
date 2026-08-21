import argparse
from pathlib import Path

from dotenv import load_dotenv

from agentbench import config
from agentbench.report import print_results_table, print_summary, write_csv, write_markdown_summary
from agentbench.runner import run_benchmark
from agentbench.tasks import get_tasks
from agentbench.metrics import summarize


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compare single-agent vs multi-agent on a task set.")
    parser.add_argument("--approaches", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    parser.add_argument("--tasks", nargs="+", default=None, help="Task ids to run (default: all)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--max-iters", type=int, default=config.DEFAULT_MAX_ITERS)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each task x approach N times")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    all_tasks = get_tasks()
    tasks = [t for t in all_tasks if args.tasks is None or t.id in args.tasks]
    if not tasks:
        parser.error(f"No matching tasks. Available: {[t.id for t in all_tasks]}")

    results, judge_call_log = run_benchmark(
        tasks=tasks,
        approaches=args.approaches,
        model=args.model,
        max_iters=args.max_iters,
        repeats=args.repeats,
    )

    print_results_table(results)
    summaries = summarize(results)
    print_summary(summaries, judge_call_log)

    output_dir = Path(args.output_dir)
    write_csv(results, output_dir / "results.csv")
    write_markdown_summary(summaries, judge_call_log, output_dir / "summary.md")
    print(f"\nWrote {output_dir / 'results.csv'} and {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
