import os
import sys
import xml.etree.ElementTree as ET
import collections
from collections import defaultdict
import globalvar as gl

gl._init()
gl.set_value('total_person_num_gt',0)
gl.set_value('total_person_num_detect',0)

def get_person_num(filepath):
    try:
        tree = ET.parse(filepath)
        # 获得根节点
        root = tree.getroot()
    except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
        print("parse .xml fail!")
        sys.exit()

    person_num_gt = 0
    for r in root.iter('Rect'):
        person_num_gt += 1

    return person_num_gt


def get_person_num_info(dirName):
    results = defaultdict(dict)
    for file in os.listdir(dirName):
        file_path = os.path.join(dirName, file)
        if os.path.isfile(file_path):
            filename = file.split(".")[0]
            results[filename] = get_person_num(file_path)
            gl.update_value('total_person_num_gt', results[filename])

    return results

def get_precision(imageName,class_ids,person_num_gt,file):
    if len(class_ids) == 0:
        print("\n*** class_ids length is 0 *** \n")
    person_dict = collections.Counter(class_ids)
    person__num_detect = person_dict[1]

    gl.update_value('total_person_num_detect', person__num_detect)

    print("检测图片名称：{:<23}，真实行人数量：{:<4}，检测到的行人数量：{:<4},相差：{:<4}".format(
        imageName,person_num_gt,person__num_detect,person__num_detect - person_num_gt),file=file)

