import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Profile function line by line',
        prog='Do line Profiling'
    )
    parser.add_argument(
        'script',
        type=str,
        help='Path to script e.g. /Python/foo.py'
    )
    results = parser.parse_args()
    cmd = 'kernprof -l -v {}'.format(results.script)
    subprocess.call(cmd)

if __name__ == '__main__':
    main()