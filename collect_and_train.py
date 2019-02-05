import os
import sys
import time
import string
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('project', help="Project name used for directory names, etc.")
args = parser.parse_args()

PROJECT = ''.join([c for c in args.project
                   if c in string.ascii_letters + string.digits])
PROJECT_DIR = f'tf_files/{PROJECT}'
ARCHITECTURE = os.environ.get('ARCHITECTURE', 'mobilenet_1.0_224')


def run(cmd):
    """Run shell command."""
    print("\n\033[0;32m", cmd, "\033[0m\n")
    result = os.system(cmd)
    if result != 0:
        sys.exit(1)


# Clear screen
os.system('cls')


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


# Train on new data
os.system(
    f"""
    python -m scripts.retrain \
            --bottleneck_dir=tf_files/bottlenecks/{PROJECT} \
            --how_many_training_steps=500 \
            --model_dir=tf_files/models/ \
            --summaries_dir=tf_files/training_summaries/"{ARCHITECTURE}"/{PROJECT}-{time.time()}\
            --output_graph=tf_files/retrained_graph.pb \
            --output_labels=tf_files/retrained_labels.txt \
            --architecture="{ARCHITECTURE}" \
            --image_dir={PROJECT_DIR} \
            --learning-rate=0.01
    """
)


# Test model with webcam
os.system(
    f"""
    python -m scripts.label_image \
            --graph=tf_files/retrained_graph.pb \
            --webcam
    """
)
