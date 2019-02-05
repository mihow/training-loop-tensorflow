'''
...display the contents of the webcam using OpenCV 
'''

import time
import os
import argparse

import cv2


rand_str = lambda: str(int(time.time()*100))[4:]


def choose_dir(base_dir='tf_files', preselected=None):
    if preselected:
        sub_dir = preselected
    else:
        dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) 
                and not d.startswith('.')]
        for i, d in enumerate(dirs):
            print(f"\033[0;32m{i}) {d}\033[0m")
        if not dirs:
            print(f"\033[0;32m(nothing found)\033[0m")

        choice = input("Select from the list above or provide a new name: ").strip()
        try:
            dir_index = int(choice)
            sub_dir = dirs[dir_index] 
        except ValueError:
            sub_dir = choice
            print("Created:", sub_dir)

    chosen_path = os.path.join(base_dir, sub_dir)
    os.makedirs(chosen_path, exist_ok=True)

    print(f'Using dir "{chosen_path}"')
    return chosen_path


def capture(image_path):
    cam = cv2.VideoCapture(0)

    print('get ready...')
    time.sleep(3)

    num_samples = 0
    last_time = time.time()

    while num_samples < 30:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        cv2.imshow('webcam', img)
        if (time.time() - last_time) > 0.5:
            print("sample...", end="")
            print(num_samples+1, "...", end="")
            #time.sleep(0.5)
            fname = os.path.join(image_path, rand_str()+'.jpg')
            cv2.imwrite(fname, img)
            print(fname)
            last_time = time.time()
            num_samples += 1
        if cv2.waitKey(1) == 27: 
            import ipdb; ipdb.set_trace()
            break  # esc to quit
    cv2.destroyAllWindows()


def main(base_dir='tf_files', project=None, label=None, mirror=False):

    if not base_dir:
        print(f"\nChoose or create a base directory for all projects:")
        base_dir = choose_dir(base_dir='.')

    if not project:
        print(f"\nChoose or create a new project:")
    project_dir = choose_dir(base_dir='tf_files', preselected=project)

    if not label:
        print(f"\nChoose or create a new label:")
    label_dir = choose_dir(base_dir=project_dir, preselected=label)

    capture(image_path=label_dir)

    return project_dir, label_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='tf_files', help="Root directory of data for all projects")
    parser.add_argument('--project', help="Name of project to use for base image dir")
    parser.add_argument('--label', help="Name of label or category to collect")
    args = parser.parse_args()

    main(**args.__dict__)
