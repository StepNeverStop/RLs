# autopep8

import os
import glob
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file-path', type=str, default=None,
                        help='.py file path that need to be formatted.')
    parser.add_argument('-d', '--file-dir', type=str, default=None,
                        help='.py dictionary that need to be formatted.')
    return parser.parse_args()


def autopep8(file_path):
    os.system(f"autopep8 -i {file_path}")
    print(f'autopep8 file: {file_path} SUCCESSFULLY.')


if __name__ == '__main__':

    args = get_args()

    if args.file_path:
        autopep8(args.file_path)

    if args.file_dir:
        py_files = []
        for root, dirnames, filenames in os.walk(args.file_dir):
            py_files.extend(glob.glob(root + "/*.py"))

        for path in py_files:
            autopep8(path)

    print('autopep8 finished.')