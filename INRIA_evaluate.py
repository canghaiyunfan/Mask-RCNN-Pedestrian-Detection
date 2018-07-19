import os
import collections
from collections import defaultdict
import globalvar as gl

gl._init()
gl.set_value('total_person_num_detect',0)
gl.set_value('total_person_num_gt',0)

def get_person_num(filepath):
    with open(filepath, 'rb') as x:
        line = x.readlines()
        content = line[5]
        content_split = content.decode('utf-8').split("{")[0].split(":")
        person_num_gt = int(content_split[1])

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

def get_precision(imageName,class_ids,person_num_gt):
    if len(class_ids) == 0:
        print("\n*** No instances *** \n")
    person_dict = collections.Counter(class_ids)
    person__num_detect = person_dict[1]

    gl.update_value('total_person_num_detect', person__num_detect)

    print("检测图片名称：{}，真实行人数量：{}，检测到的行人数量：{}".format(
        imageName,person_num_gt,person__num_detect))



