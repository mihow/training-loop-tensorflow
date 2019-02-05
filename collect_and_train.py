#! /usr/bin/env python3

import os
import sys
import time
import string
import argparse


PROJECT = 'project'
ARCHITECTURE = os.environ.get('ARCHITECTURE', 'mobilenet_1.0_224')


def run(cmd):
    """Run shell command."""
    print("\n\033[0;32m", cmd.strip(), "\033[0m\n")
    result = os.system(cmd)
    if result != 0:
        sys.exit(1)


def collect():
    # Capture training samples
    keep_collecting = True
    while keep_collecting:
        run(
            f"""
            python -m scripts.capture \
                    --project={PROJECT}
            """
        )
        answer = input("\nContinue collecting samples? Y/n: ").strip().lower()
        keep_collecting = True if 'y' in answer else False


def train():
    # Train on new data
    run(
        f"""
        python -m scripts.retrain \
                --bottleneck_dir=tf_files/bottlenecks/{PROJECT} \
                --how_many_training_steps=500 \
                --model_dir=tf_files/models/ \
                --summaries_dir=tf_files/training_summaries/"{ARCHITECTURE}"/{PROJECT}-{time.time()}\
                --output_graph=tf_files/{PROJECT}-retrained_graph.pb \
                --output_labels=tf_files/{PROJECT}-retrained_labels.txt \
                --architecture="{ARCHITECTURE}" \
                --image_dir=tf_files/{PROJECT} \
                --learning-rate=0.01
        """
    )


def test():
    # Test model with webcam
    run(
        f"""
        python -m scripts.label_image \
                --graph=tf_files/{PROJECT}-retrained_graph.pb \
                --labels=tf_files/{PROJECT}-retrained_labels.txt \
                --webcam
        """
    )


def main(do_collect=True, do_train=True, do_test=True):
    # Clear screen
    os.system('clear')

    if do_collect:
        collect()

    if do_train:
        train()

    if do_test:
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='project',
                        help="Project name used for directory names, etc.")
    parser.add_argument('--no-collect', dest='collect', action='store_false',
                        help="Skip collecting samples")
    parser.add_argument('--no-train', dest='train', action='store_false',
                        help="Skip training")
    parser.add_argument('--no-test', dest='test', action='store_false',
                        help="Skip live testing of trained model")
    args = parser.parse_args()

    PROJECT = ''.join([c for c in args.project
                       if c in string.ascii_letters + string.digits])

    main(args.collect, args.train, args.test)
