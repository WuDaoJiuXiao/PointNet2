import torch
import re
import time
import matplotlib.pyplot as plt
from pylab import mpl


# 加载 pth 模型文件
def loadPth(path: str) -> torch.nn.Module:
    return torch.load(path, map_location=torch.device('cpu'))


# 清洗日志，提取出每次epoch的概率
def washLogs(log_file: str) -> list:
    res = []
    with open(log_file, "r", encoding='utf-8') as f:
        epoch_index, train_accuracy, test_accuracy = None, None, None
        test_class_accuracy, best_accuracy, best_class_accuracy = None, None, None
        for log in f.readlines():
            epoch_info = re.findall("Epoch.*?:", log)
            train_info = re.findall("Train .*", log)
            test_info = re.findall("Test .*", log)
            best_info = re.findall("Best .*", log)
            if len(epoch_info) > 0:
                epoch_index = epoch_info[0].split(" ")[-1].split("/")[0][1:]
            if len(train_info) > 0:
                train_accuracy = train_info[0].split(":")[-1].strip()
            if len(test_info) > 0:
                test_accuracy = test_info[0].split(",")[0].split(":")[-1].split(",")[0].strip()
                test_class_accuracy = test_info[-1].split(",")[-1].split(":")[-1].strip()
            if len(best_info) > 0:
                best_accuracy = best_info[0].split(",")[0].split(":")[-1].split(",")[0].strip()
                best_class_accuracy = best_info[-1].split(",")[-1].split(":")[-1].strip()
            if epoch_index is not None and train_accuracy is not None and test_accuracy is not None \
                    and test_class_accuracy is not None and best_accuracy is not None \
                    and best_class_accuracy is not None:
                temp = {
                    'epoch_index': epoch_index,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_class_accuracy': test_class_accuracy,
                    'best_accuracy': best_accuracy,
                    'best_class_accuracy': best_class_accuracy
                }
                res.append(temp)
                epoch_index, train_accuracy, test_accuracy = None, None, None
                test_class_accuracy, best_accuracy, best_class_accuracy = None, None, None
    return res


# 绘制训练信息
def drawNetInfo(logs: list, isSave: bool) -> None:
    # 准备数据
    epoch, train_acc, test_acc, test_cls_acc, best_acc, best_cls_scc = [], [], [], [], [], []
    for item in logs:
        epoch.append(float(item['epoch_index']))
        train_acc.append(float(item['train_accuracy']))
        test_acc.append(float(item['test_accuracy']))
        test_cls_acc.append(float(item['test_class_accuracy']))
        best_acc.append(float(item['best_accuracy']))
        best_cls_scc.append(float(item['best_class_accuracy']))
    # 设置图像的属性
    plt.title("模型精度变化趋势图")
    # 自己设置了图像尺寸后，会出现图像没有标题的bug
    # plt.figure(figsize=(12, 8))
    # 解决不能显示中文的问题
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # 设置点的属性
    plt.plot(epoch, train_acc)
    plt.plot(epoch, test_acc)
    plt.plot(epoch, test_cls_acc)
    plt.plot(epoch, best_acc)
    plt.plot(epoch, best_cls_scc)
    # 设置每个坐标数据标签位置及大小，一般不显示，显示了看不清
    # for a, b in zip(epoch, train_acc):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=5)
    # 设置折线名称
    plt.legend(['Train Accuracy', 'Test Accuracy', 'Test Class Accuracy', 'Best Accuracy', 'Best Class Accuracy'])
    # 如果用户要保存图片，那就保存当当前路径，图片命名格式为日期加时间
    if isSave:
        img_name = str(time.time()).replace(".", "")
        plt.savefig(img_name, dpi=1000)
        print("图片保存在当前路径下，图片名为: {}.png".format(img_name))
    plt.show()
