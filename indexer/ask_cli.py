"""One question from the command line -> answer JSON on stdout.

    python ask_cli.py --db ../app.db --drawing NAME --question "..."

The server calls this first: a known question template costs nothing and
needs no model.
"""
import argparse
import json
import sqlite3
import sys

import answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--drawing", required=True)
    ap.add_argument("--question", required=True)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    json.dump(answer.ask(con, a.drawing, a.question), sys.stdout,
              ensure_ascii=False, default=str)


if __name__ == "__main__":
    sys.exit(main())
