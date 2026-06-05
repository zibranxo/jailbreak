"""CLI tool for interactive text classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.classifiers.safety_classifier import SafetyClassifier
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Safety Classifier CLI")
    parser.add_argument("--text", "-t", help="Text to classify")
    parser.add_argument("--file", "-f", type=Path, help="File with text (one example per line)")
    parser.add_argument("--config", "-c", type=Path, help="Path to config.yaml")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = load_config(str(args.config) if args.config else None)
    classifier = SafetyClassifier(config)

    def run(text: str) -> None:
        result = classifier.classify(text)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            label = result["label"]
            conf = result["confidence"]
            stage = result["stage"]

            # Color coding
            colors = {
                "safe": "\033[32m",       # green
                "toxic": "\033[33m",       # yellow
                "jailbreak": "\033[31m",   # red
            }
            reset = "\033[0m"
            color = colors.get(label, reset)

            print(f"\n{'=' * 60}")
            print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"{'=' * 60}")
            print(f"Label:    {color}{label.upper()}{reset}")
            print(f"Confidence: {conf:.2%}")
            print(f"Stage:    {stage}")

            for expl in result.get("explanations", []):
                print(f"  [{expl['source'].upper()}] {expl['message']}")
            print()

    if args.interactive:
        print("LLM Safety Classifier (interactive mode)")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                run(text)
    elif args.file:
        with open(args.file) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    run(line)
    elif args.text:
        run(args.text)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
