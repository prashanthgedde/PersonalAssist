"""
Test runner for both simple and multi-turn agents.

Run individual tests:
    python test_simple_agent.py     # Single-round agent
    python test_multi_turn_agent.py # Iterative agent

Or run both:
    python test_agent.py
"""

import sys


def main():
    print("=" * 70)
    print("Running Agent Tests")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1. SIMPLE AGENT (Single Round)")
    print("=" * 70)
    print("Run: python test_simple_agent.py")
    print("-" * 70)
    exec(open("test_simple_agent.py").read())

    print("\n" + "=" * 70)
    print("2. MULTI-TURN AGENT (Iterative)")
    print("=" * 70)
    print("Run: python test_multi_turn_agent.py")
    print("-" * 70)
    exec(open("test_multi_turn_agent.py").read())


if __name__ == "__main__":
    main()
