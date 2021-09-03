# autopep8

import argparse
import glob
import os

import isort


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--file-path', type=str, default=None,
                        help='.py file path that need to be formatted.')
    parser.add_argument('-d', '--file-dir', type=str, default=None,
                        help='.py dictionary that need to be formatted.')
    parser.add_argument('--ignore-pep', default=False, action='store_true',
                        help='whether format the file')
    return parser.parse_args()


def autopep8(file_path, ignore_pep):
    isort.file(file_path)
    print(f'isort file: {file_path} SUCCESSFULLY.')
    if not ignore_pep:
        os.system(f"autopep8 -j 0 -i {file_path} --max-line-length 200")
        print(f'autopep8 file: {file_path} SUCCESSFULLY.')


if __name__ == '__main__':

    args = get_args()

    if args.file_path:
        autopep8(args.file_path, args.ignore_pep)

    if args.file_dir:
        py_files = []
        for root, dirnames, filenames in os.walk(args.file_dir):
            py_files.extend(glob.glob(root + "/*.py"))

        for path in py_files:
            autopep8(path, args.ignore_pep)

    print('auto-format finished.')
