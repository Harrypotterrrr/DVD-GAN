import os
import sys


def classify_video(dir_path, dest_path, file_name):
    class_name = file_name.split('_')[1]
    class_path = os.path.join(dest_path, class_name)
    if os.path.exists(class_path) is False:
        os.mkdir(class_path)
    origin_path = os.path.join(dir_path, file_name)
    file_path = os.path.join(class_path, file_name)
    cmd = "mv {} {}".format(origin_path, file_path)
    print(cmd)
    os.system(cmd)
    return

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dest_path = sys.argv[2]
    if os.path.exists(dest_path) is False:
        os.makedirs(dest_path)
    for file in os.listdir(dir_path):
        classify_video(dir_path, dest_path, file)
