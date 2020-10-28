import argparse

from rls.distribute.apex.learner import server
from rls.distribute.apex.worker import client


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a', type=str, nargs='*')
    parser.add_argument('--environment', '-e', type=str, default='')
    parser.add_argument('--learner', '-l', action='store_true')
    parser.add_argument('--worker', '-w', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cmd_args()
    print(args)
    if args.learner:
        server()
    if args.worker:
        client()
