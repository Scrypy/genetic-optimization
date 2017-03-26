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
    parser.add_argument(
        'image',
        type=str,
        help='Path to image e.g. image.jpg'
    )
    results = parser.parse_args()
    cmd = 'kernprof -l -v {} {}'.format(results.script, results.image)
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()