import os
import shutil

# 当前项目根目录
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))).replace("\\", "/")
# 存储待处理点云数据的路径
CLASS_POINT_PATH = ROOT_DIR + "/myLabel/class/"
# 生成的点云信息文件保存根路径
INFO_ROOT_PATH = ROOT_DIR + "/data/modelnet40_normal_resampled/"
# filelist.txt 文件保存路径
FILE_LIST_SAVE_PATH = ROOT_DIR + "/data/modelnet40_normal_resampled/filelist.txt"
# 形状分类名文件 modelnet40_shape_names.txt 保存路径
SHAPE_NAME_SAVE_PATH = ROOT_DIR + "/data/modelnet40_normal_resampled/modelnet40_shape_names.txt"
# 训练集文件 modelnet40_train.txt 保存路径
TRAIN_SAVE_PATH = ROOT_DIR + "/data/modelnet40_normal_resampled/modelnet40_train.txt"
# 测试集文件 modelnet40_test.txt 保存路径
TEST_SAVE_PATH = ROOT_DIR + "/data/modelnet40_normal_resampled/modelnet40_test.txt"
# 训练集：测试集的比例，默认为 9：1
TRAIN_STEP, TEST_STEP = 9, 1


# 生成分类所需的点云文件信息
def getClassInfo():
    files, train, test = [], [], []
    if not os.path.exists(INFO_ROOT_PATH):
        os.makedirs(INFO_ROOT_PATH)

    # 创建 filelist.txt 文件
    for item in os.listdir(CLASS_POINT_PATH):
        for ff in os.listdir(CLASS_POINT_PATH + item):
            files.append(item + "/" + ff)
    with open(FILE_LIST_SAVE_PATH, "w+", encoding="utf-8") as f:
        for fil in files: f.write(fil + "\n")
    print("filelist.txt 生成成功...")

    # 创建 modelnet40_shape_names.txt 文件
    with open(SHAPE_NAME_SAVE_PATH, "w+", encoding="utf-8") as f:
        for item in os.listdir(CLASS_POINT_PATH):
            f.write(item + "\n")
    print("modelnet40_shape_names.txt 生成成功...")
    # 创建 modelnet40_train.txt / modelnet40_test.txt文件
    # 训练集：测试集默认比例为9：1，可在上方常量处修改
    for item in os.listdir(CLASS_POINT_PATH):
        temp_point_list = os.listdir(CLASS_POINT_PATH + item)
        point_nums = len(temp_point_list)
        step = int(point_nums / 10)
        for ff in temp_point_list[: step * 9]:
            train.append(ff.split(".")[0])
        for ff in temp_point_list[step * 9:]:
            test.append(ff.split(".")[0])
    with open(TRAIN_SAVE_PATH, "w+", encoding="utf-8") as f:
        for fi in train: f.write(fi + "\n")
    print("modelnet40_train.txt 生成成功...")
    with open(TEST_SAVE_PATH, "w+", encoding="utf-8") as f:
        for fi in test: f.write(fi + "\n")
    print("modelnet40_test.txt 生成成功...")


# 将数据集复制到正确位置
def moveDataSet():
    print("开始复制点云文件到正确路径...")
    for item in os.listdir(CLASS_POINT_PATH):
        if not os.path.exists(INFO_ROOT_PATH + item + "/"):
            os.makedirs(INFO_ROOT_PATH + item + "/")
        for ff in os.listdir(CLASS_POINT_PATH + item):
            old_path = CLASS_POINT_PATH + item + "/" + ff
            new_path = INFO_ROOT_PATH + item + "/" + ff
            shutil.copy(old_path, new_path)
            print("文件{}移动成功...".format(ff))


if __name__ == '__main__':
    getClassInfo()
    moveDataSet()
