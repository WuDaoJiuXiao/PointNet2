import os
import shutil

# 当前项目根目录
ROOT_DIR = os.path.abspath(os.path.dirname(os.getcwd())).replace("\\", "/")
# 未处理的标注数据所在的文件夹
CC_FILE_PATH = ROOT_DIR + "/myLabel/partSeg/"
# 处理好的标准化数据保存路径
SAVE_PATH = ROOT_DIR + "/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/"
# 生成的 json 文件保存路径
JSON_PATH = ROOT_DIR + "/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/"
# 读取 synsetoffset2category.txt 的路径
READ_TXT_PATH = ROOT_DIR + "/synsetoffset2category.txt"
# 生成的 synsetoffset2category.txt 保存路径
SET2CATEGORY_SAVE_PATH = ROOT_DIR + "/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt"


# 将 CC 导出的标注数据转换为 PointNet++ 要求的标准形式
def standardizationData():
    print("====================== 开始标准化CC文件 ============================\n")
    for temp_dir in os.listdir(CC_FILE_PATH):
        print("开始处理 {} 文件夹下文件...\n".format(temp_dir))
        temp_path = CC_FILE_PATH + "/" + temp_dir + "/"
        for fil in os.listdir(temp_path):
            res = []
            with open(CC_FILE_PATH + "/" + temp_dir + "/" + fil, "r", encoding='utf-8') as f:
                data = f.readlines()
                new_data = data[2:]
                point_num = len(new_data)
                print("正在处理文件：{}，该文件点云数据共{}个...".format(fil, point_num))
                for item in new_data:
                    spl_list = item.split(',')
                    new_str = ""
                    length = len(spl_list)
                    # 添加三维坐标、三维法向量信息
                    for i in [0, 1, 2, length - 3, length - 2, length - 1]:
                        # 去除末尾的换行符
                        if i == length - 1:
                            new_str += spl_list[-1][:-1] + " "
                        else:
                            new_str += spl_list[i] + " "
                    # 去除中间的 nan 数据，只保留标签数据
                    # CC 导出的数据如有 filed 标签选项，下方为 [4:-3]，没有则为 [3: -3]
                    for value in spl_list[3: -3]:
                        if value != 'nan':
                            new_str += value + " "
                    res.append(new_str[:-1] + "\n")
            # 保存处理之后的数据
            save_dirs = SAVE_PATH + "/" + temp_dir + "/"
            if not os.path.exists(save_dirs):
                os.makedirs(save_dirs)
            with open(save_dirs + fil, "w+", encoding='utf-8') as f:
                for cont in res:
                    f.writelines(cont)
        print("{}文件夹下文件全部处理完成...\n".format(temp_dir))
    print("====================== 所有CC文件均已标准化完成 ============================\n")


# 生成 json 文件，训练：测试：验证 = 7：2：1
def getJson():
    train, test, val = [], [], []
    # 首先将所有的文件按照7：2：1比例划分为三部分，全部加入到列表中，最后再写入 json 文件中
    print("===================== 开始处理 json 文件 ===============================\n")
    for temp_dir in os.listdir(SAVE_PATH):
        print("正在处理 {} 文件夹下数据...".format(temp_dir))
        d = SAVE_PATH + temp_dir
        txt_list = os.listdir(d)
        length = len(txt_list)
        step = int(length / 10)
        for item in txt_list[: step * 7]: train.append("shape_data/" + temp_dir + "/" + item.split('.')[0])
        for item in txt_list[step * 7: -step]: test.append("shape_data/" + temp_dir + "/" + item.split('.')[0])
        for item in txt_list[-step:]: val.append("shape_data/" + temp_dir + "/" + item.split('.')[0])
        print("文件夹 {} 下数据处理完成...".format(temp_dir))
    # 写入 json 数据
    print("正在写入 json 文件...")
    writeJson(train, JSON_PATH, "shuffled_train_file_list.json")
    writeJson(test, JSON_PATH, "shuffled_test_file_list.json")
    writeJson(val, JSON_PATH, "shuffled_val_file_list.json")
    print("===================== json 文件已全部生成 ===============================\n")


# 写入 json 数据
def writeJson(data_list: list, json_dir: str, file_name: str):
    # 路径不存在则创建文件夹
    idx = 0
    length = len(data_list)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(JSON_PATH + file_name, "w+", encoding="utf-8") as f:
        f.write("[\n")
        for data in data_list:
            idx += 1
            if idx == length:
                f.write("\t\"" + data + "\"" + "\n")
            else:
                f.write("\t\"" + data + "\"," + "\n")
        f.write("]")
    print("文件{}生成完毕".format(file_name))


# 将原本的 synsetoffset2category.txt 移动到正确位置
def moveCategoryTxt():
    new_path, _ = os.path.split(SET2CATEGORY_SAVE_PATH)
    # 不存在则创建文件夹
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    shutil.copy(READ_TXT_PATH, SET2CATEGORY_SAVE_PATH)
    print("synsetoffset2category.txt 复制成功...\n所有数据文件均创建完成!")


if __name__ == '__main__':
    standardizationData()
    getJson()
    moveCategoryTxt()
