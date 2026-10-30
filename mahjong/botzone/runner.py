"""Simple-IO Mahjong with cold-history replay and persistent subsequent turns."""

import json
import sys
from .inference import load
from .session import Session


def main():
    choose, config = load()
    session = Session(choose)
    first = True
    for line in sys.stdin:
        if not line.strip():
            continue
        line = line.strip()
        if first:
            first = False
            turn = int(line)
            for _ in range(turn - 1):
                request = sys.stdin.readline().strip()
                response = sys.stdin.readline().strip()
                session.handle(request, forced=response)
            request = sys.stdin.readline().strip()
        else:
            request = line
        response = session.handle(request)
        print(response)
        print("")  # debug
        print("")  # data
        print("")  # globaldata
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
