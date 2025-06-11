import copy
import math
import sys
import time
import traceback
from copy import deepcopy

import cv2
import numpy as np
import os
import json

import torch

print(torch.cuda.is_available())

from PIL import Image, ImageDraw, ImageFont
# import matplotlib.font_manager as fm  # to create font
from matplotlib import pyplot as plt
from torchvision import transforms

sys.path.append("..")
from GAN2_hd_split_xl_cd7_RTM_attention_adain_X2_6b_infer import Generator

from RTM_drawtool import draw_pose_split_after_instance, draw_pose_after_instance
from threading import Thread

# GAN模型加载:
# 1.权重位置
checkpoint_dir = './training_checkpoints_split_raw_cd'
checkpoint_pt = '/model_g_20250428_144210_2360000.pt'
checkpoint_path = checkpoint_dir + checkpoint_pt
# 2.初始化模型以及权重加载
generator = Generator()
generator.load_state_dict(torch.load(checkpoint_path, weights_only=True))
# 3.将模型推入GPU，并启动推理模式
generator = generator.cuda()
generator = generator.eval()

# 输入和输出的图像的分辨率
IMG_WIDTH = 512
IMG_HEIGHT = 512


# 欧几里得距离（二维）
def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))

# 模型推理生成函数
def generate_images(model, example_input_image, example_style_image, npy_list):
    """
            输入模型实例、骨架图、STYLE图，坐标数组，输出推理的图像
            :param model: 输入的模型实例
            :param example_input_image: 输入的骨架信息图，形状：[batch_size,3,512,512]（torch tensor）
            :param example_style_image: 输入的STYLE信息图，形状：[batch_size,3,512,512]（torch tensor）
            :param npy_list: 输入的骨架坐标数组，形状：[batch_size,133,2]（torch tensor）
            :return: 返回的模型推理结果的图像，形状：[batch_size,3,512,512]（torch tensor）
    """
    npy_list = torch.nan_to_num(npy_list)
    bodies = npy_list[:, 0:23, :]
    faces = npy_list[:, 23:91, :]
    left_hands = npy_list[:, 91:112, :]
    right_hands = npy_list[:, 112:, :]
    content = example_input_image
    style = example_style_image
    prediction = model(content, style, bodies, faces, left_hands, right_hands)
    # 测试代码
    # tset_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer("add_7").output)
    # prd_test = tset_layer.predict(test_input)
    # print(prd_test)
    # y, idx = tf.unique(tf.math.is_nan(tf.reshape(prd_test, [-1])))
    # print(y)
    return prediction

# Key points 的文件路径
file_names_list = os.listdir(f'./RTM_keypoint_frames_npy_all_det')
file_names_list.sort()
file_names_list_length = len(file_names_list)

# 视频输出的大小
size = (512, 512)
# 骨架画面视频输出的大小
size_input = (720, 720)
batch_size = 1
# 读取STYLE图像，并转为值域（-1，1）的 torch tensor，形状最终为[batch_size,3,512,512]
style_image = Image.open("./style_img_raw (1).jpg")
style_image = np.array(style_image, dtype=np.uint8)  # H*W*C
transform_s = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(p=p),
    transforms.Resize(size),
    transforms.ToTensor(),
])
style_image = transform_s(style_image)
style_image = (style_image * 2) - 1
style_images = [style_image for i in range(batch_size)]
style_images = torch.stack(style_images, dim=0)
style_images = style_images.cuda()

# 定义承接模型输出的数据数组，以及承接骨架输入的数据数组（这两个将通过CV2输出视频）
outputs_list = []
input_images_list = []
videoWrite = None
videoWrite_input = None
# 是否需要将输入的内容也输出为视频，默认False
need_output_keypoints_data = False
# 推理结束标志
finish = False



def show_output():
    """
        -将模型输出的内容写入视频文件
        -关于全局变量的说明在全局变量定义处
    """
    global outputs_list, input_images_list, videoWrite, finish, videoWrite_input, need_output_keypoints_data
    while True:
        if len(input_images_list) > 0:
            input_show = input_images_list.pop(0)
            input_show = cv2.cvtColor(input_show, cv2.COLOR_BGR2RGB)
            cv2.imshow("infer_input", input_show)
            if need_output_keypoints_data:
                if videoWrite_input is not None:
                    videoWrite_input.write(input_show)

        if len(outputs_list) > 0:
            prediction_show = outputs_list.pop(0)
            prediction_show = np.array(prediction_show, dtype=np.uint8)
            # print(prediction_show.shape)
            # print(np.max(prediction_show))
            # print(np.min(prediction_show))
            #
            # print(input_show.shape)
            prediction_show = cv2.cvtColor(prediction_show, cv2.COLOR_BGR2RGB)

            cv2.imshow("infer_prediction", prediction_show)
            if videoWrite is not None:
                videoWrite.write(prediction_show)
        if finish and len(outputs_list) == 0 and len(input_images_list) == 0:
            videoWrite.release()
            videoWrite_input.release()
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue
        time.sleep(0.1)
    cv2.destroyAllWindows()

# 启动输出函数线程，使用异步的方式输出到生成的视频文件
show_thread = Thread(target=show_output, args=())
show_thread.start()

# 读取JSON格式的词汇与词汇对应的文件名的内容
with open("test4.json", "r", encoding='UTF-8') as read_file:
    file_list = json.load(read_file)

# 把文字添加到输出画面的底部（半废弃状态）
def draw_text(text):
    textColor = (255, 255, 255)
    textSize = 20
    img = Image.fromarray(cv2.cvtColor(np.zeros([50, 720, 3]).astype(np.uint8), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontText = ImageFont.truetype("arial.ttf", textSize, encoding="utf-8")

    draw.text((10, 10), text, textColor, font=fontText)

    text_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return text_img


def moving_average_smoothing(data, window_size=5):
    """
    滑动窗口均值轨迹平滑函数，用于在时域范围内对骨架坐标突变或抖动噪声的抑制
        :param data: 输入的数据，形状为[frames（总帧数）,points_range（选取的坐标点的范围）,2（x，y的值）]（numpy array）
        :param window_size: 滑动处理窗口大小
        :return: 平滑后的数据，形状为[frames（总帧数）,points_range（选取的坐标点的范围）,2（x，y的值）]（numpy array）
    - 窗口大小应为奇数
    """
    n_frames, n_points, _ = data.shape
    smoothed = np.zeros_like(data)

    # 确保窗口大小为奇数
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2

    for point_idx in range(n_points):
        for coord in range(2):
            # 扩展边界处理
            extended = np.pad(data[:, point_idx, coord], (half_window, half_window), 'edge')

            # 应用移动平均
            smoothed_vals = np.convolve(extended, np.ones(window_size) / window_size, 'valid')
            smoothed[:, point_idx, coord] = smoothed_vals

    return smoothed


def is_in_expanded_range(theta1, theta2, theta3, extend=math.pi / 3):
    """
         输入三个弧度制的角度，以及一个扩展角度，求角3是否处在角1扩展后的角度和角2扩展后的角度的夹角范围（锐角）内。
         :param theta1: 角度1（弧度）
         :param theta2: 角度2（弧度）
         :param theta3: 角度3（弧度）
         :param extend: 扩展角度（弧度），比如角1和角2之间30度，扩展角度为10度，最后计算出的夹角则为：10+30+10=50度
         :return: 判断后的布尔值，真则为角3在夹角范围中，假则不在。
    """
    # 将角度的值域从[-pi,pi]转化为[0,2pi]
    theta1 = theta1 + np.pi
    theta2 = theta2 + np.pi
    theta3 = theta3 + np.pi
    # print("theta3", theta3)
    # 情况1：角1大于角2，角1往值变大方向扩展，角2往值变小方向扩展
    if theta1 >= theta2:
        start = theta1 + extend
        start_extened = start

        if start > 2 * np.pi:
            start = start - 2 * np.pi
        end = theta2 - extend
        if end < 0:
            end = 2 * np.pi + end
        # print("t1>t2")
        # print("start", start)
        # print("end", end)
        # 如果扩展后的角度没有突破值域，则进行直接比较
        if start >= end:
            if start >= theta3 >= end:
                return True
            else:
                return False
        #如果扩展后突破了值域，则进行将突破部分特殊判断
        else:
            if theta3 >= end or theta3 <= start or (theta3 < 2 * np.pi and theta3 < start_extened):
                return True
            else:
                return False
    # 情况2：角2大于角1，角2往值变大方向扩展，角1往值变小方向扩展
    elif theta1 < theta2:
        start = theta2 + extend
        start_extened = start
        if start > 2 * np.pi:
            start = start - 2 * np.pi
        end = theta1 - extend
        if end < 0:
            end = 2 * np.pi + end
        # print("t1<t2")
        # print("start", start)
        # print("end", end)
        # 如果扩展后的角度没有突破值域，则进行直接比较
        if start >= end:
            if start >= theta3 >= end:
                return True
            else:
                return False
        # 如果扩展后突破了值域，则进行将突破部分特殊判断
        else:
            if theta3 >= end or theta3 <= start or (theta3 < 2 * np.pi and theta3 < start_extened):
                return True
            else:
                return False


def interpolate_angles(theta_start, theta_end, mode, n):
    """
           输入角1（theta_start）、角2（theta_end）、模式（mode）、帧数（n），求出角1旋转到角2的过程，在n帧范围内进行插值运算，以mode来决定插值锐角方向还是钝角方向。
           :param theta_start: 角度1（弧度）
           :param theta_end: 角度2（弧度）
           :param mode: 插值的模式，可选锐角（acute）和钝角（delta_acute）
           :param n: 需要插值的帧数
           :return: 每一帧的角度值（弧度），数据结构是List
    """
    # 将角度归一化到0-360范围
    theta_start = theta_start % (2 * np.pi)
    theta_end = theta_end % (2 * np.pi)

    # 计算原始角度差
    delta_raw = theta_end - theta_start

    # 计算锐角方向的角度差（最短路径）
    delta_acute = (delta_raw + np.pi) % (2 * np.pi) - np.pi

    # 根据模式确定最终的角度差
    if mode == 'acute':
        delta = delta_acute
    elif mode == 'obtuse':
        if delta_acute > 0:
            delta = delta_acute - (2 * np.pi)
        else:
            delta = delta_acute + (2 * np.pi)
    else:
        raise ValueError("mode must be 'acute' or 'obtuse'")

    # 生成每一帧的角度值
    frames = []
    for i in range(n):
        if n == 1:
            t = 0.0
        else:
            t = i / (n - 1)
        theta = theta_start + delta * t
        theta %= (2 * np.pi)  # 确保角度在0-360之间
        frames.append(theta)

    return frames


def rotate_points(points, theta):
    """
    将二维点集绕原点旋转指定角度
    :param points: 输入点集，形状为(N, 2)的numpy数组
    :param theta: 旋转角度（弧度）
    :return: 旋转后的点集
    """
    # 构造旋转矩阵
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # 应用旋转矩阵
    return points @ rotation_matrix.T  # 等价于np.dot(points, rotation_matrix.T)


def calculate_rotation(B_point):
    """
    计算将B点旋转到y轴正方向所需的旋转矩阵
    :param B_point: B点原始坐标 [x, y]
    :return: 旋转矩阵(2x2), 旋转角度(弧度)
    """
    # 计算当前B点的极坐标
    current_angle = np.arctan2(B_point[1], B_point[0])

    # 计算需要旋转的角度（到y轴正方向）
    target_angle = np.pi / 2  # 90度
    theta = target_angle - current_angle

    # 构造旋转矩阵
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    return rotation_matrix, theta, current_angle


def apply_rotation(points, rotation_matrix):
    """
    应用旋转矩阵到多个点
    :param points: 要旋转的点集合，形状为(n, 2)
    :param rotation_matrix: 旋转矩阵(2x2)
    :return: 旋转后的点坐标
    """
    return np.dot(points, rotation_matrix.T)  # 矩阵乘法


def inverse_rotation(points, rotation_matrix):
    """
    逆旋转（恢复原始坐标）
    :param points: 旋转后的点集合
    :param rotation_matrix: 原始旋转矩阵
    :return: 原始坐标点集合
    """
    inv_matrix = rotation_matrix.T  # 旋转矩阵的逆是它的转置
    return np.dot(points, inv_matrix.T)


def batch_transition_root(A, B, A_root, B_root, n_frames):
    """
    批量计算多个点从A到B的过渡动画（旋转+缩放）

    参数：
    A: 起始点坐标数组，形状为(batch_size, 2)
    B: 结束点坐标数组，形状为(batch_size, 2)
    n_frames: 过渡帧数（不包括起始帧）

    返回：
    过渡动画数组，形状为(batch_size, n_frames, 2)
    """
    # 转换为极坐标
    r_A = np.linalg.norm(A, axis=1)  # 各点的起始模长 [batch_size]
    theta_A = np.arctan2(A[:, 1], A[:, 0])  # 起始角度 [batch_size]

    r_B = np.linalg.norm(B, axis=1)  # 结束模长 [batch_size]
    theta_B = np.arctan2(B[:, 1], B[:, 0])  # 结束角度 [batch_size]

    # 计算角度差值（保证走最短路径）
    delta_theta = (theta_B - theta_A + np.pi) % (2 * np.pi) - np.pi

    print(delta_theta)

    # 生成时间参数 [n_frames, 1]
    t = np.linspace(0, 1, n_frames + 2)[1:-1, None]  # 排除起点和终点

    # 批量插值计算 [n_frames, batch_size]
    r = r_A + (r_B - r_A) * t  # 模长插值
    theta = theta_A + delta_theta * t  # 角度插值

    # 转换回笛卡尔坐标 [n_frames, batch_size, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    frames = np.stack([x, y], axis=-1)

    # 调整维度为 [batch_size, n_frames, 2]
    return np.transpose(frames, (1, 0, 2)), frames


def batch_transition_2(A, B, n_frames):
    """
    批量计算多个点从A到B的过渡动画（旋转+缩放）

    参数：
    A: 起始点坐标数组，形状为(batch_size, 2)
    B: 结束点坐标数组，形状为(batch_size, 2)
    n_frames: 过渡帧数（不包括起始帧）

    返回：
    过渡动画数组，形状为(batch_size, n_frames, 2)
    """
    # 转换为极坐标
    r_A = np.linalg.norm(A, axis=1)  # 各点的起始模长 [batch_size]
    theta_A = np.arctan2(A[:, 1], A[:, 0])  # 起始角度 [batch_size]

    r_B = np.linalg.norm(B, axis=1)  # 结束模长 [batch_size]
    theta_B = np.arctan2(B[:, 1], B[:, 0])  # 结束角度 [batch_size]

    # 计算角度差值（保证走最短路径）
    delta_theta = (theta_B - theta_A + np.pi) % (2 * np.pi) - np.pi
    # print(delta_theta)

    # 生成时间参数 [n_frames, 1]
    t = np.linspace(0, 1, n_frames + 2)[1:-1, None]  # 排除起点和终点

    # 批量插值计算 [n_frames, batch_size]
    r = r_A + (r_B - r_A) * t  # 模长插值
    theta = theta_A + delta_theta * t  # 角度插值

    # 转换回笛卡尔坐标 [n_frames, batch_size, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    frames = np.stack([x, y], axis=-1)

    # 调整维度为 [batch_size, n_frames, 2]
    return np.transpose(frames, (1, 0, 2)), frames


def visualize_batch_transition(A, B, frames):
    """可视化批量过渡动画"""
    plt.figure(figsize=(10, 10))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    batch_size = A.shape[0]
    colors = plt.cm.rainbow(np.linspace(0, 1, batch_size))

    for i in range(batch_size):
        # 绘制起点终点
        plt.scatter(A[i, 0], A[i, 1], c=[colors[i]], s=100, marker='o', edgecolors='k')
        plt.scatter(B[i, 0], B[i, 1], c=[colors[i]], s=100, marker='s', edgecolors='k')

        # 绘制路径
        path = np.vstack([A[i], frames[i], B[i]])
        plt.plot(path[:, 0], path[:, 1], c=colors[i], alpha=0.3, linestyle='--')

        # 绘制过渡帧
        plt.scatter(frames[i, :, 0], frames[i, :, 1], c=[colors[i]], s=30, alpha=0.5)

    plt.title(f'Batch Transition Animation ({batch_size} points)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


for file_idx in range(len(file_list)):
    # 需要推理的内容中词汇列表（已废弃）
    gloss = file_list[file_idx]["glosses_merged"]
    # 需要推理的内容中文件名列表（读取每个词汇对应的坐标文件）
    file_names = file_list[file_idx]["pred_merged"]
    # 以句子为元素的书面原文（已废弃）
    inputs = file_list[file_idx]["input"]
    # 创建推理结果输出的文件夹
    if not os.path.exists(f'demos/{file_idx}'):
        os.makedirs(f'demos/{file_idx}')
    gloss_list = []
    file_names_list = []
    # 视频输出的每秒帧数
    fps = 45

    # 将JSON文件的内容转移到数组上
    for input_idx in range(len(inputs)):
        # inputs_text = inputs[input_idx]
        # print(inputs_text)
        gloss_list.extend(gloss[input_idx])
        file_names_list.extend(file_names[input_idx])
    # 推理结果的视频写入器
    videoWrite = cv2.VideoWriter(f'./demos/{file_idx}/ajit.webm',
                                 cv2.VideoWriter_fourcc(*'vp09'),
                                 fps,
                                 size)
    # 输入内容的视频写入器（debug用）
    if need_output_keypoints_data:
        videoWrite_input = cv2.VideoWriter(f'./demos/{file_idx}/ajit_input.webm',
                                           cv2.VideoWriter_fourcc(*'vp09'),
                                           fps,
                                           size_input)
    # 输入的坐标文件内容中，以720*720作为从坐标文件绘制输入的骨架图像的基准分辨率
    thispart_resolution = 720
    # 将词汇写在输出的图像上的控制（已废弃）
    inputs_sentence_Gidx = 0
    inputs_sentence_Iidx = 0

    # 上一次词汇的坐标数据数组中割出来的屁股的帧，作为下一个词汇的开头与上一个词汇进行动作过度的数据（最大值应为process_frame_max帧）
    rebuild_data_temp_last_list = []
    # 需要推理的文件列表中的当前位置
    idx_out = 0
    # print(file_names_list)
    # 读取标准姿势，在推理的开头和结尾使用
    rebuild_data_start_end_frame = np.load(f"./7275-frame_0.npy", allow_pickle=True).item()
    # 进行坐标位置的标准化，将坐标按照基准点偏移
    point_5_1 = rebuild_data_start_end_frame['pose'][5, 1]
    point_6_1 = rebuild_data_start_end_frame['pose'][6, 1]
    mean_high = (point_5_1 + point_6_1) / 2
    # 纵向以y=430为人体肩旁中点的基准
    standard_point_mean_high = 430.0
    diff_mean_high = standard_point_mean_high - mean_high

    # print("mean_high:", mean_high)

    point_0_0 = rebuild_data_start_end_frame['pose'][0, 0]
    # 横向以x=340为人体横向中点的基准
    standard_point_0_0 = 340.0
    diff = standard_point_0_0 - point_0_0
    rebuild_data_start_end_frame['pose'][:, 0] = rebuild_data_start_end_frame['pose'][:, 0] + diff
    rebuild_data_start_end_frame['pose'][:, 1] = rebuild_data_start_end_frame['pose'][:, 1] + diff_mean_high
    # print("rebuild_data_list:", rebuild_data_list[i]['pose'][0, 0])
    # 将人体腰部位置的坐标进行范围约束，不让其发生剧烈抖动，从而提高腰部位置生成质量
    rebuild_data_start_end_frame['pose'][11, 1] = ((rebuild_data_start_end_frame['pose'][
                                                        11, 1] - 660.) / 80) * 10 + 690.
    rebuild_data_start_end_frame['pose'][12, 1] = ((rebuild_data_start_end_frame['pose'][
                                                        12, 1] - 660.) / 80) * 10 + 690.
    # 开始对文件列表进行迭代，推理
    while idx_out < len(file_names_list):


        # 读取这个词汇文件夹里的帧内容文件
        frames_dir = os.listdir(f'./RTM_keypoint_frames_npy_all_det/{file_names_list[idx_out]}')


        print(idx_out)
        # 这个词汇文件夹的推理进度
        real_idx = 0
        # 词汇字符串（已废弃）
        this_gloss = gloss_list[idx_out]
        # 上一词汇和下一词汇之间进行过渡动作计算的最大帧数
        process_frame_max = 15
        # 计算过渡动作已经到第几帧了
        last_idx_start=0

        # 过滤完成后，最终用来进一步计算的列表，比如过度动作计算
        rebuild_data_list = []
        # 过滤器的临时变量
        rebuild_data_list_pre = []
        rebuild_data_last_list = []
        rebuild_data_last_list_pre = []
        # 全部处理完成后的最终列表
        rebuild_data_processed_list = []
        # 过滤器1：将文件列表中的坏帧去掉，平均置信度超0.9的个数低于35%，存在巨大噪声（欧几里得距离超出人体应该有的正常值）
        for idx in range(len(frames_dir)):
            rebuild_data = np.load(
                f"./RTM_keypoint_frames_npy_all_det/{file_names_list[idx_out]}/frame_{idx}.npy",
                allow_pickle=True).item()
            threshold = 0.35

            pose = rebuild_data['pose']
            pose_subset = rebuild_data['pose_subset']
            if not np.any(pose[91:]):
                continue
            l_c = np.mean(pose_subset[91:112] >= 0.9)
            r_c = np.mean(pose_subset[112:] >= 0.9)
            if l_c >= threshold:
                left_hand = pose[91:112] / 720.
                left_dis_list = []
                for idx_1 in range(left_hand.shape[0]):
                    base_point = pose[9] / 720.
                    next_point = left_hand[idx_1]
                    dis = eucliDist(base_point, next_point)
                    left_dis_list.append(dis)
                dist_l_m = np.max(np.array(left_dis_list))
                if dist_l_m > 0.2:
                    continue

            if r_c >= threshold:
                right_hand = pose[112:] / 720.
                right_dis_list = []
                for idx_1 in range(right_hand.shape[0]):
                    base_point = pose[10] / 720.
                    next_point = right_hand[idx_1]
                    dis = eucliDist(base_point, next_point)

                    right_dis_list.append(dis)

                dist_r_m = np.max(np.array(right_dis_list))
                if dist_r_m > 0.2:
                    continue

            rebuild_data_list_pre.append(rebuild_data)
        # 过滤器2，将手部坐标的y低于590作为整个动作的开始，将手部坐标高于560作为整个动作的结束，并往后抠出最大15帧去做下一个词的过渡动作计算
        # （y值坐标越在屏幕下面值越大，x从画面左边到右边（0-720），y从画面上边到下边（0-720））
        has_skiped = False
        rebuild_data_list_pre_temp = []
        if idx_out > 0:
            for idx in range(len(rebuild_data_list_pre)):
                pose = rebuild_data_list_pre[idx]['pose']
                pose_subset = rebuild_data_list_pre[idx]['pose_subset']
                if (pose[9, 1] > 590.0 and pose[10, 1] > 590.0) and has_skiped == False:
                    continue
                else:
                    has_skiped = True
                    rebuild_data_list_pre_temp.append(rebuild_data_list_pre[idx])
        else:
            rebuild_data_list_pre_temp = rebuild_data_list_pre
        print("rebuild_data_list_pre_temp:", len(rebuild_data_list_pre_temp))

        has_skiped = False
        selected_idx = -1
        if idx_out < len(file_names_list) - 1:
            for idx in range(len(rebuild_data_list_pre_temp)):
                pose = rebuild_data_list_pre_temp[len(rebuild_data_list_pre_temp) - 1 - idx]['pose']
                pose_subset = rebuild_data_list_pre_temp[len(rebuild_data_list_pre_temp) - 1 - idx]['pose_subset']
                if (pose[9, 1] > 560.0 and pose[10, 1] > 560.0) and has_skiped == False:
                    continue
                else:
                    selected_idx = len(rebuild_data_list_pre_temp) - 1 - idx
                    has_skiped = True
                    break

            rebuild_data_list = rebuild_data_list_pre_temp[0:selected_idx + 1]
        else:
            rebuild_data_list = rebuild_data_list_pre_temp
        if idx_out > 0:
            rebuild_data_last_list = rebuild_data_temp_last_list
        print("selected_idx:", selected_idx)
        print("rebuild_data_list:", len(rebuild_data_list))
        print("rebuild_data_last_list:", len(rebuild_data_last_list))
        rebuild_data_temp_last_list = rebuild_data_list_pre_temp[selected_idx:selected_idx + process_frame_max]
        print("rebuild_data_temp_last_list:", len(rebuild_data_temp_last_list))
        process_frame = len(rebuild_data_last_list) if len(
            rebuild_data_last_list) < process_frame_max else process_frame_max


        # 对所有的帧的坐标值都进行以标准点为基准的偏移
        for i in range(len(rebuild_data_list)):
            # print("rebuild_data_list:", rebuild_data_list[i]['pose'][0, 0])
            # print("rebuild_data_list:", rebuild_data_list[i]['pose'][5, 1])
            # print("rebuild_data_list:", rebuild_data_list[i]['pose'][6, 1])
            point_5_1 = rebuild_data_list[i]['pose'][5, 1]
            point_6_1 = rebuild_data_list[i]['pose'][6, 1]
            mean_high = (point_5_1 + point_6_1) / 2
            standard_point_mean_high = 430.0
            diff_mean_high = standard_point_mean_high - mean_high

            # print("mean_high:", mean_high)

            point_0_0 = rebuild_data_list[i]['pose'][0, 0]
            standard_point_0_0 = 340.0
            diff = standard_point_0_0 - point_0_0
            rebuild_data_list[i]['pose'][:, 0] = rebuild_data_list[i]['pose'][:, 0] + diff
            rebuild_data_list[i]['pose'][:, 1] = rebuild_data_list[i]['pose'][:, 1] + diff_mean_high
            # print("rebuild_data_list:", rebuild_data_list[i]['pose'][0, 0])
            rebuild_data_list[i]['pose'][11, 1] = ((rebuild_data_list[i]['pose'][11, 1] - 660.) / 80) * 10 + 690.
            rebuild_data_list[i]['pose'][12, 1] = ((rebuild_data_list[i]['pose'][12, 1] - 660.) / 80) * 10 + 690.
            # rebuild_data['pose'][91:112, 1] = rebuild_data['pose'][91:112, 1] + diff_y
        for i in range(len(rebuild_data_last_list)):
            # print("rebuild_data_last_list:", rebuild_data_last_list[i]['pose'][0, 0])
            point_0_0 = rebuild_data_last_list[i]['pose'][0, 0]

            point_5_1 = rebuild_data_last_list[i]['pose'][5, 1]
            point_6_1 = rebuild_data_last_list[i]['pose'][6, 1]
            mean_high = (point_5_1 + point_6_1) / 2
            standard_point_mean_high = 430.0
            diff_mean_high = standard_point_mean_high - mean_high

            standard_point_0_0 = 340.0
            diff = standard_point_0_0 - point_0_0
            rebuild_data_last_list[i]['pose'][:, 0] = rebuild_data_last_list[i]['pose'][:, 0] + diff
            rebuild_data_last_list[i]['pose'][:, 1] = rebuild_data_last_list[i]['pose'][:, 1] + diff_mean_high
            rebuild_data_last_list[i]['pose'][11, 1] = ((rebuild_data_last_list[i]['pose'][
                                                             11, 1] - 660.) / 80) * 10 + 690.
            rebuild_data_last_list[i]['pose'][12, 1] = ((rebuild_data_last_list[i]['pose'][
                                                             12, 1] - 660.) / 80) * 10 + 690.
            # print("rebuild_data_last_list:", rebuild_data_last_list[i]['pose'][0, 0])

        def gen_style_smooth(pose_start, pose_end, seconds, rebuild_data_start_end_frame,thispart_resolution):
            gen_frames = int(seconds * fps)
            gen_style_smooth_rebuild_data_list = []
            blink_frames = 8
            blink_frames_gap = 24
            blink_frames_all = blink_frames + blink_frames_gap
            blink_frames_idx = 0
            can_blink = True
            blink_times = 0

            def calculate_point_D(A, B, C, k):
                # 解构各点坐标
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C

                # 计算向量AB的分量
                dx = Bx - Ax
                dy = By - Ay

                # 检查AB是否为同一个点
                if dx == 0 and dy == 0:
                    raise ValueError("点A和点B重合，无法计算垂线。")

                # 计算向量AC的分量
                acx = Cx - Ax
                acy = Cy - Ay

                # 计算点积和AB长度的平方
                dot_product = acx * dx + acy * dy
                ab_length_sq = dx ** 2 + dy ** 2

                # 计算参数t
                t = dot_product / ab_length_sq

                # 计算垂足P的坐标
                Px = Ax + t * dx
                Py = Ay + t * dy

                # 计算点D的坐标
                Dx = Px + k * (Cx - Px)
                Dy = Py + k * (Cy - Py)

                return (Dx, Dy)

            gen_style_smooth_rebuild_data_temp_list = []
            for frame_idx in range(gen_frames):
                gen_style_smooth_rebuild_data = copy.deepcopy(rebuild_data_start_end_frame)
                for c_idx in range(0, 13):
                    # if c_idx==9 or c_idx==10:
                    #     continue
                    x_end = pose_end[c_idx, 0]
                    y_end = pose_end[c_idx, 1]

                    x_start = pose_start[c_idx, 0]
                    y_start = pose_start[c_idx, 1]

                    x_end = (x_start * (1 - (frame_idx / gen_frames)) + x_end * (
                            frame_idx / gen_frames))
                    y_end = (y_start * (1 - (frame_idx / gen_frames)) + y_end * (
                            frame_idx / gen_frames))

                    gen_style_smooth_rebuild_data['pose'][c_idx, 0] = x_end
                    gen_style_smooth_rebuild_data['pose'][c_idx, 1] = y_end

                for face_idx in range(23, 91):
                    x_end = pose_end[face_idx, 0]
                    y_end = pose_end[face_idx, 1]

                    x_start = pose_start[face_idx, 0]
                    y_start = pose_start[face_idx, 1]

                    x_end = (x_start * (1 - (frame_idx / gen_frames)) + x_end * (
                            frame_idx / gen_frames))
                    y_end = (y_start * (1 - (frame_idx / gen_frames)) + y_end * (
                            frame_idx / gen_frames))

                    gen_style_smooth_rebuild_data['pose'][face_idx, 0] = x_end
                    gen_style_smooth_rebuild_data['pose'][face_idx, 1] = y_end
                if can_blink:
                    left_eye = [66, 67, 69, 70]
                    right_eye = [60, 61, 63, 64]
                    if blink_frames_idx <= blink_frames_gap:
                        blink_frames_idx = blink_frames_idx + 1
                    else:
                        blink_offset = blink_frames_idx - blink_frames_gap
                        for idx in left_eye:
                            pose = gen_style_smooth_rebuild_data['pose']
                            point_left_eye_left_side = pose[68]
                            point_left_eye_right_side = pose[65]
                            point_eye_for_calculation = pose[idx]
                            stretch_value = np.abs((4 - blink_offset) / 4)
                            print("stretch_value:", stretch_value)
                            point_eye_after_calculation = calculate_point_D(point_left_eye_left_side,
                                                                            point_left_eye_right_side,
                                                                            point_eye_for_calculation, stretch_value)
                            gen_style_smooth_rebuild_data['pose'][idx] = point_eye_after_calculation
                        for idx in right_eye:
                            pose = gen_style_smooth_rebuild_data['pose']
                            point_right_eye_left_side = pose[62]
                            point_right_eye_right_side = pose[59]
                            point_eye_for_calculation = pose[idx]
                            stretch_value = np.abs((4 - blink_offset) / 4)
                            point_eye_after_calculation = calculate_point_D(point_right_eye_left_side,
                                                                            point_right_eye_right_side,
                                                                            point_eye_for_calculation, stretch_value)
                            gen_style_smooth_rebuild_data['pose'][idx] = point_eye_after_calculation
                        blink_frames_idx = blink_frames_idx + 1
                        if blink_frames_idx % blink_frames_all == 0:
                            blink_frames_idx = 0
                            blink_times = blink_times + 1
                            if (gen_frames - frame_idx) <= blink_frames_all:
                                can_blink = False
                # gen_style_smooth_rebuild_data_list.append(gen_style_smooth_rebuild_data)
                gen_style_smooth_rebuild_data_list.append([gen_style_smooth_rebuild_data, thispart_resolution])
                gen_style_smooth_rebuild_data_temp_list.append(gen_style_smooth_rebuild_data['pose'])

            gen_style_smooth_rebuild_data_list_npy = np.array(gen_style_smooth_rebuild_data_temp_list)
            need_filter_data = gen_style_smooth_rebuild_data_list_npy[:, 5:13, :]
            filtered_data = moving_average_smoothing(need_filter_data)
            gen_style_smooth_rebuild_data_list_npy[:, 5:13, :] = filtered_data

            for idx in range(len(gen_style_smooth_rebuild_data_list_npy)):
                gen_style_smooth_rebuild_data_list[idx][0]["pose"][5:13, :] = gen_style_smooth_rebuild_data_list_npy[
                                                                              idx, 5:13, :]

            return gen_style_smooth_rebuild_data_list, blink_times

        # 对开头和结尾从标准姿势到正常姿势或者从正常姿势到标准姿势的过渡动作计算
        gen_style_smooth_rebuild_data_list = None
        blink_times = 0
        if idx_out == 0 or idx_out == len(file_names_list) - 1:
            if idx_out == 0:
                pose_start = rebuild_data_start_end_frame['pose']
                pose_end = rebuild_data_list[0]['pose']
                gen_style_smooth_rebuild_data_list, blink_times = gen_style_smooth(pose_start, pose_end, 1,
                                                                                   rebuild_data_start_end_frame,thispart_resolution)

            else:
                pose_start = rebuild_data_list[-1]['pose']
                pose_end = rebuild_data_start_end_frame['pose']
                gen_style_smooth_rebuild_data_list, blink_times = gen_style_smooth(pose_start, pose_end, 1,
                                                                                   rebuild_data_start_end_frame,thispart_resolution)
        rebuild_data_list_raw = copy.deepcopy(rebuild_data_list)
        half_process_frame = int(math.ceil(process_frame * 0.5))

        for idx in range(len(rebuild_data_list)):

            rebuild_data = copy.deepcopy(rebuild_data_list[idx])

            threshold = 0.35
            single_threshold = 0.85

            pose = rebuild_data['pose']

            try:

                if last_idx_start < process_frame and idx_out > 0:

                    pose = rebuild_data["pose"]
                    pose_back = rebuild_data["pose"].copy()
                    pose_subset = rebuild_data["pose_subset"]

                    l_c = np.mean(pose_subset[91:112] >= 0.9)
                    r_c = np.mean(pose_subset[112:] >= 0.9)

                    rebuild_data_last = copy.deepcopy(rebuild_data_last_list[last_idx_start])

                    pose_last = rebuild_data_last["pose"]
                    pose_last_back = rebuild_data_last["pose"].copy()
                    pose_subset_last = rebuild_data_last["pose_subset"]
                    # print(pose_last[91:112])
                    # print(pose_subset_last[91:112])

                    need_continue1 = False
                    need_continue2 = False

                    need_recover1 = False
                    need_use_last1 = False

                    list_x_this = []
                    list_y_this = []
                    list_x_last = []
                    list_y_last = []

                    if l_c < threshold:
                        need_continue1 = True

                    mean_x_this = pose[9, 0]
                    mean_y_this = pose[9, 1]
                    mean_x_last = pose_last[9, 0]
                    mean_y_last = pose_last[9, 1]
                    final_mean_x = (mean_x_last * (1 - (last_idx_start / process_frame)) + mean_x_this * (
                            last_idx_start / process_frame))
                    final_mean_y = (mean_y_last * (1 - (last_idx_start / process_frame)) + mean_y_this * (
                            last_idx_start / process_frame))

                    if last_idx_start > process_frame * 0.5:
                        diff_x = final_mean_x - mean_x_this
                        diff_y = final_mean_y - mean_y_this
                    else:
                        diff_x = final_mean_x - mean_x_last
                        diff_y = final_mean_y - mean_y_last

                    diff_x = np.nan_to_num(diff_x, nan=0.)
                    diff_y = np.nan_to_num(diff_y, nan=0.)

                    if last_idx_start > process_frame * 0.5:
                        rebuild_data['pose'][91:112, 0] = rebuild_data['pose'][91:112, 0] + diff_x
                        rebuild_data['pose'][91:112, 1] = rebuild_data['pose'][91:112, 1] + diff_y

                        offset_x = rebuild_data['pose'][91, 0] - final_mean_x
                        offset_y = rebuild_data['pose'][91, 1] - final_mean_y

                        rebuild_data['pose'][91:112, 0] = rebuild_data['pose'][91:112, 0] - offset_x
                        rebuild_data['pose'][91:112, 1] = rebuild_data['pose'][91:112, 1] - offset_y

                        pose[91:112] = rebuild_data['pose'][91:112]
                        rebuild_data['pose'][9, 0] = final_mean_x
                        rebuild_data['pose'][9, 1] = final_mean_y
                        pose[9] = rebuild_data['pose'][9]




                    else:
                        rebuild_data['pose'][91:112, 0] = rebuild_data_last['pose'][91:112, 0] + diff_x
                        rebuild_data['pose'][91:112, 1] = rebuild_data_last['pose'][91:112, 1] + diff_y

                        offset_x = rebuild_data['pose'][91, 0] - final_mean_x
                        offset_y = rebuild_data['pose'][91, 1] - final_mean_y

                        rebuild_data['pose'][91:112, 0] = rebuild_data['pose'][91:112, 0] - offset_x
                        rebuild_data['pose'][91:112, 1] = rebuild_data['pose'][91:112, 1] - offset_y

                        rebuild_data['pose_subset'][91:112:] = pose_subset_last[91:112:]
                        pose[91:112] = rebuild_data['pose'][91:112]
                        pose_subset[91:112] = rebuild_data['pose_subset'][91:112]
                        rebuild_data['pose'][9, 0] = final_mean_x
                        rebuild_data['pose'][9, 1] = final_mean_y
                        pose[9] = rebuild_data['pose'][9]

                    need_recover2 = False
                    need_use_last2 = False

                    list_x_this = []
                    list_y_this = []
                    list_x_last = []
                    list_y_last = []
                    if r_c < threshold:
                        need_continue2 = True

                    mean_x_this = pose[10, 0]
                    mean_y_this = pose[10, 1]
                    mean_x_last = pose_last[10, 0]
                    mean_y_last = pose_last[10, 1]

                    final_mean_x = (mean_x_last * (1 - (last_idx_start / process_frame)) + mean_x_this * (
                            last_idx_start / process_frame))
                    final_mean_y = (mean_y_last * (1 - (last_idx_start / process_frame)) + mean_y_this * (
                            last_idx_start / process_frame))

                    if last_idx_start > process_frame * 0.5:
                        diff_x = final_mean_x - mean_x_this
                        diff_y = final_mean_y - mean_y_this
                    else:
                        diff_x = final_mean_x - mean_x_last
                        diff_y = final_mean_y - mean_y_last

                    if last_idx_start > process_frame * 0.5:
                        rebuild_data['pose'][112:, 0] = rebuild_data['pose'][112:, 0] + diff_x
                        rebuild_data['pose'][112:, 1] = rebuild_data['pose'][112:, 1] + diff_y

                        offset_x = rebuild_data['pose'][112, 0] - final_mean_x
                        offset_y = rebuild_data['pose'][112, 1] - final_mean_y

                        rebuild_data['pose'][112:, 0] = rebuild_data['pose'][112:, 0] - offset_x
                        rebuild_data['pose'][112:, 1] = rebuild_data['pose'][112:, 1] - offset_y

                        pose[112:] = rebuild_data['pose'][112:]
                        rebuild_data['pose'][10, 0] = final_mean_x
                        rebuild_data['pose'][10, 1] = final_mean_y
                        pose[10] = rebuild_data['pose'][10]

                    else:
                        rebuild_data['pose'][112:, 0] = rebuild_data_last['pose'][112:, 0] + diff_x
                        rebuild_data['pose'][112:, 1] = rebuild_data_last['pose'][112:, 1] + diff_y

                        offset_x = rebuild_data['pose'][112, 0] - final_mean_x
                        offset_y = rebuild_data['pose'][112, 1] - final_mean_y

                        rebuild_data['pose'][112:, 0] = rebuild_data['pose'][112:, 0] - offset_x
                        rebuild_data['pose'][112:, 1] = rebuild_data['pose'][112:, 1] - offset_y

                        rebuild_data['pose_subset'][112:] = pose_subset_last[112:]
                        pose[112:] = rebuild_data['pose'][112:]
                        pose_subset[112:] = rebuild_data['pose_subset'][112:]
                        rebuild_data['pose'][10, 0] = final_mean_x
                        rebuild_data['pose'][10, 1] = final_mean_y
                        pose[10] = rebuild_data['pose'][10]

                    if need_continue1 and need_continue2:
                        # print("continue")
                        continue

                    need_recover = False
                    need_use_last = False
                    for c_idx in range(0, 13):

                        x_this = pose[c_idx, 0]
                        y_this = pose[c_idx, 1]
                        this_subset = pose_subset[c_idx]

                        x_last = pose_last[c_idx, 0]
                        y_last = pose_last[c_idx, 1]
                        last_subset = pose_subset_last[c_idx]

                        # if this_subset <= single_threshold:
                        #     continue
                        if c_idx == 0 or c_idx == 1 or c_idx == 2 or c_idx == 3 or c_idx == 4:
                            x_this = (x_last * (1 - (last_idx_start / process_frame)) + x_this * (
                                    last_idx_start / process_frame))
                            y_this = (y_last * (1 - (last_idx_start / process_frame)) + y_this * (
                                    last_idx_start / process_frame))
                        elif c_idx == 9:
                            pass
                            # x_this = pose[91, 0]
                            # y_this = pose[91, 1]
                        elif c_idx == 10:
                            pass
                            # x_this = pose[112, 0]
                            # y_this = pose[112, 1]
                        else:
                            # if this_subset > single_threshold and last_subset > single_threshold:
                            x_this = (x_last * (1 - (last_idx_start / process_frame)) + x_this * (
                                    last_idx_start / process_frame))
                            y_this = (y_last * (1 - (last_idx_start / process_frame)) + y_this * (
                                    last_idx_start / process_frame))

                        rebuild_data['pose'][c_idx, 0] = x_this
                        rebuild_data['pose'][c_idx, 1] = y_this
                    if need_recover:
                        rebuild_data['pose'][0:13] = pose_back[0:13]

                    need_recover_face = False
                    need_use_last_face = False
                    for face_idx in range(23, 91):
                        x_this = pose[face_idx, 0]
                        y_this = pose[face_idx, 1]
                        this_subset = pose_subset[face_idx]

                        x_last = pose_last[face_idx, 0]
                        y_last = pose_last[face_idx, 1]
                        last_subset = pose_subset_last[face_idx]

                        x_this = (x_last * (1 - (last_idx_start / process_frame)) + x_this * (
                                last_idx_start / process_frame))
                        y_this = (y_last * (1 - (last_idx_start / process_frame)) + y_this * (
                                last_idx_start / process_frame))

                        rebuild_data['pose'][face_idx, 0] = x_this
                        rebuild_data['pose'][face_idx, 1] = y_this

                        # if c_idx == 27:
                        #     print(rebuild_data['faces'][0][c_idx][0])

                    if need_recover_face:
                        rebuild_data['pose'][23:91] = pose_back[23:91]
                    last_idx_start = last_idx_start + 1
                # print("hand2:", rebuild_data['hands'][1][0])

                thispart_resolution = 720
                scale_ori_rate = 0

                rebuild_data_processed_list.append([rebuild_data.copy(), thispart_resolution])

            except Exception as e:
                info = traceback.format_exc()
                print(info)

            real_idx = real_idx + 1

        if idx_out > 0:
            start_frame_left = deepcopy(rebuild_data_processed_list[0][0])
            start_frame_hand_to_0 = start_frame_left['pose']
            start_frame_hand_to_0[91:112, 0] = start_frame_hand_to_0[91:112, 0] - start_frame_hand_to_0[9, 0]
            start_frame_hand_to_0[91:112, 1] = start_frame_hand_to_0[91:112, 1] - start_frame_hand_to_0[9, 1]
            # start_frame_hand_to_0_cut = start_frame_hand_to_0[91:112]

            end_frame_left = deepcopy(rebuild_data_processed_list[half_process_frame][0])
            end_frame_hand_to_0 = end_frame_left['pose']
            end_frame_hand_to_0[91:112, 0] = end_frame_hand_to_0[91:112, 0] - end_frame_hand_to_0[9, 0]
            end_frame_hand_to_0[91:112, 1] = end_frame_hand_to_0[91:112, 1] - end_frame_hand_to_0[9, 1]
            # end_frame_hand_to_0_cut = end_frame_hand_to_0[91:112]

            start_frame_hand_to_0 = np.array(start_frame_hand_to_0)
            end_frame_hand_to_0 = np.array(end_frame_hand_to_0)
            if end_frame_hand_to_0[9, 1] < 660:
                start_point = 91
                level_points = [[92, 96, 100, 104, 108], [93, 97, 101, 105, 109], [94, 98, 102, 106, 110],
                                [95, 99, 103, 107, 111]]

                start_frame_hand_to_0_cut = start_frame_hand_to_0[91:112]
                end_frame_hand_to_0_cut = end_frame_hand_to_0[91:112]
                rotation_matrix_start, theta_start, angle_start = calculate_rotation(start_frame_hand_to_0[103, :])
                start_frame_hand_to_0_cut = apply_rotation(start_frame_hand_to_0_cut, rotation_matrix_start)
                rotation_matrix_end, theta_end, angle_end = calculate_rotation(end_frame_hand_to_0[103, :])
                end_frame_hand_to_0_cut = apply_rotation(end_frame_hand_to_0_cut, rotation_matrix_end)
                start_frame_hand_to_0[91:112] = start_frame_hand_to_0_cut
                end_frame_hand_to_0[91:112] = end_frame_hand_to_0_cut

                transition_trans, transitions = batch_transition_2(start_frame_hand_to_0_cut,
                                                                   end_frame_hand_to_0_cut,
                                                                   half_process_frame)
                # visualize_batch_transition(start_frame_hand_to_0_cut, end_frame_hand_to_0_cut, transition_trans)

                raw_delta = theta_start - theta_end
                raw_delta = (raw_delta + np.pi) % (2 * np.pi) - np.pi
                middle_frame_left = deepcopy(rebuild_data_processed_list[int((half_process_frame + 1) // 2)][0])
                middle_frame = middle_frame_left['pose']
                middle_frame = np.array(middle_frame)
                _, _, arm_middle_angle = calculate_rotation(middle_frame[9] - middle_frame[7])
                _, _, arm_start_angle = calculate_rotation(start_frame_hand_to_0[9] - start_frame_hand_to_0[7])
                _, _, arm_end_angle = calculate_rotation(end_frame_hand_to_0[9] - end_frame_hand_to_0[7])
                # print("arm_start_angle", arm_start_angle)
                # print("arm_middle_angle", arm_middle_angle)
                # print("arm_end_angle", arm_end_angle)
                # print("raw_delta", raw_delta)
                check_interval = is_in_expanded_range(arm_start_angle, arm_end_angle, arm_middle_angle)
                if check_interval:
                    angles = interpolate_angles(angle_start, angle_end, 'acute', half_process_frame)
                    print('acute')

                else:
                    angles = interpolate_angles(angle_start, angle_end, 'obtuse', half_process_frame)
                    print('obtuse')

                for n in range(1, len(transitions) + 1):
                    transition = transitions[n - 1, :, :]

                    target_angle = np.pi / 2  # 90度
                    theta = target_angle - angles[n - 1]

                    # 构造旋转矩阵
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                                [sin_theta, cos_theta]])
                    transition = inverse_rotation(transition, rotation_matrix)
                    transition = np.reshape(transition, [transition.shape[0], transition.shape[1]])
                    # print(transition.shape)

                    frame_right = rebuild_data_processed_list[n][0]["pose"]
                    frame_right = np.array(frame_right)
                    frame_right_basepoint = frame_right[9:10, :]
                    transition = transition + frame_right_basepoint
                    rebuild_data_processed_list[n][0]['pose'][91:112, :] = transition

            start_frame_right = deepcopy(rebuild_data_processed_list[0][0])
            start_frame_hand_to_0 = start_frame_right['pose']
            start_frame_hand_to_0[112:, 0] = start_frame_hand_to_0[112:, 0] - start_frame_hand_to_0[10, 0]
            start_frame_hand_to_0[112:, 1] = start_frame_hand_to_0[112:, 1] - start_frame_hand_to_0[10, 1]

            end_frame_right = deepcopy(rebuild_data_processed_list[half_process_frame + 1][0])
            end_frame_hand_to_0 = end_frame_right['pose']
            end_frame_hand_to_0[112:, 0] = end_frame_hand_to_0[112:, 0] - end_frame_hand_to_0[10, 0]
            end_frame_hand_to_0[112:, 1] = end_frame_hand_to_0[112:, 1] - end_frame_hand_to_0[10, 1]

            start_frame_hand_to_0 = np.array(start_frame_hand_to_0)
            end_frame_hand_to_0 = np.array(end_frame_hand_to_0)
            if end_frame_hand_to_0[10, 1] < 660:
                start_point = 112
                level_points = [[113, 117, 121, 125, 129], [114, 118, 122, 126, 130], [115, 119, 123, 127, 131],
                                [116, 120, 124, 128, 132]]
                transitions_groups = []

                init_state_ = 0
                start_frame_hand_to_0_cut = start_frame_hand_to_0[112:]
                end_frame_hand_to_0_cut = end_frame_hand_to_0[112:]
                rotation_matrix_start, theta_start, angle_start = calculate_rotation(start_frame_hand_to_0[124, :])
                start_frame_hand_to_0_cut = apply_rotation(start_frame_hand_to_0_cut, rotation_matrix_start)
                rotation_matrix_end, theta_end, angle_end = calculate_rotation(end_frame_hand_to_0[124, :])
                end_frame_hand_to_0_cut = apply_rotation(end_frame_hand_to_0_cut, rotation_matrix_end)
                start_frame_hand_to_0[112:] = start_frame_hand_to_0_cut
                end_frame_hand_to_0[112:] = end_frame_hand_to_0_cut
                print("angle_start", angle_start)
                print("angle_end", angle_end)

                transition_trans, transitions = batch_transition_2(start_frame_hand_to_0_cut,
                                                                   end_frame_hand_to_0_cut,
                                                                   half_process_frame)

                raw_delta = angle_start - angle_end
                raw_delta = (raw_delta + np.pi) % (2 * np.pi) - np.pi

                middle_frame_right = deepcopy(rebuild_data_processed_list[int((half_process_frame + 1) // 2)][0])
                middle_frame = middle_frame_right['pose']
                middle_frame = np.array(middle_frame)
                _, _, arm_middle_angle = calculate_rotation(middle_frame[10] - middle_frame[8])
                _, _, arm_start_angle = calculate_rotation(start_frame_hand_to_0[10] - start_frame_hand_to_0[8])
                _, _, arm_end_angle = calculate_rotation(end_frame_hand_to_0[10] - end_frame_hand_to_0[8])
                # print("arm_start_angle", arm_start_angle)
                # print("arm_middle_angle", arm_middle_angle)
                # print("arm_end_angle", arm_end_angle)
                # print("raw_delta", raw_delta)

                check_interval = is_in_expanded_range(arm_start_angle, arm_end_angle, arm_middle_angle)
                if check_interval:
                    angles = interpolate_angles(angle_start, angle_end, 'acute', half_process_frame)
                    print('acute')

                else:
                    angles = interpolate_angles(angle_start, angle_end, 'obtuse', half_process_frame)
                    print('obtuse')

                for n in range(1, len(transitions) + 1):
                    transition = transitions[n - 1, :, :]
                    # print(transition.shape)

                    target_angle = np.pi / 2  # 90度
                    theta = target_angle - angles[n - 1]

                    # 构造旋转矩阵
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                                [sin_theta, cos_theta]])

                    transition = inverse_rotation(transition, rotation_matrix)
                    transition = np.reshape(transition, [transition.shape[0], transition.shape[1]])
                    # print(transition.shape)

                    frame_right = rebuild_data_processed_list[n][0]["pose"]
                    frame_right = np.array(frame_right)
                    frame_right_basepoint = frame_right[10:11, :]
                    transition = transition + frame_right_basepoint
                    rebuild_data_processed_list[n][0]['pose'][112:, :] = transition

                    middle_frame_right = deepcopy(rebuild_data_processed_list[n][0])
                    middle_frame = middle_frame_right['pose']
                    middle_frame = np.array(middle_frame)

        input_image_list = []
        npy_list = []

        print(len(rebuild_data_processed_list))
        if idx_out == 0 and gen_style_smooth_rebuild_data_list is not None:
            gen_style_smooth_rebuild_data_list.extend(rebuild_data_processed_list)
            rebuild_data_processed_list = gen_style_smooth_rebuild_data_list
            print(len(rebuild_data_processed_list))
            print("blink_times:", blink_times)
        threshold = 0.35
        for idx in range(len(rebuild_data_processed_list)):
            rebuild_data = rebuild_data_processed_list[idx][0]
            pose = rebuild_data["pose"]
            # print("pose:", pose)
            pose_subset = rebuild_data["pose_subset"]
            thispart_resolution = rebuild_data_processed_list[idx][1]
            out_test = draw_pose_after_instance(pose, pose_subset, thispart_resolution, thispart_resolution, threshold)

            base_background = np.full([720, 720, 3], 255, np.uint8)
            final_img_3c = base_background.copy()
            if thispart_resolution > 720:
                gap = (out_test.shape[0] - base_background.shape[0]) / 2
                left_bound = int(gap)
                right_buond = int(out_test.shape[0] - gap)
                final_img_3c = out_test[left_bound:right_buond, left_bound:right_buond, :]
            elif thispart_resolution < 720:
                gap = (base_background.shape[0] - out_test.shape[0]) / 2
                left_bound = int(gap)
                right_buond = int(base_background.shape[0] - gap)

                final_img_3c[left_bound:right_buond, left_bound:right_buond, :] = out_test
            elif thispart_resolution == 720:
                final_img_3c = out_test

            input_image_b, input_image_lh, input_image_rh, input_image_f = draw_pose_split_after_instance(pose,
                                                                                                          pose_subset,
                                                                                                          thispart_resolution,
                                                                                                          thispart_resolution,
                                                                                                          threshold)

            base_background = np.zeros([720, 720, 3], np.uint8)
            final_img_b = base_background.copy()
            final_img_lh = base_background.copy()
            final_img_rh = base_background.copy()
            final_img_f = base_background.copy()
            if thispart_resolution > 720:
                gap = (out_test.shape[0] - base_background.shape[0]) / 2
                left_bound = int(gap)
                right_buond = int(out_test.shape[0] - gap)
                final_img_b = input_image_b[left_bound:right_buond, left_bound:right_buond, :]
                final_img_lh = input_image_lh[left_bound:right_buond, left_bound:right_buond, :]
                final_img_rh = input_image_rh[left_bound:right_buond, left_bound:right_buond, :]
                final_img_f = input_image_f[left_bound:right_buond, left_bound:right_buond, :]
            elif thispart_resolution < 720:
                gap = (base_background.shape[0] - out_test.shape[0]) / 2
                left_bound = int(gap)
                right_buond = int(base_background.shape[0] - gap)
                final_img_b[left_bound:right_buond, left_bound:right_buond, :] = input_image_b
                final_img_lh[left_bound:right_buond, left_bound:right_buond, :] = input_image_lh
                final_img_rh[left_bound:right_buond, left_bound:right_buond, :] = input_image_rh
                final_img_f[left_bound:right_buond, left_bound:right_buond, :] = input_image_f
            elif thispart_resolution == 720:
                final_img_b = input_image_b
                final_img_lh = input_image_lh
                final_img_rh = input_image_rh
                final_img_f = input_image_f

            base_background = np.zeros([720, 720, 3], np.uint8)
            final_yscale_img_b = base_background.copy()
            final_yscale_img_lh = base_background.copy()
            final_yscale_img_rh = base_background.copy()
            final_yscale_img_f = base_background.copy()

            input_image_b = final_img_b
            input_image_lh = final_img_lh
            input_image_rh = final_img_rh
            input_image_f = final_img_f

            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            input_image_b = transform(input_image_b)
            input_image_lh = transform(input_image_lh)
            input_image_rh = transform(input_image_rh)
            input_image_f = transform(input_image_f)

            input_image_b = (input_image_b * 2) - 1
            input_image_lh = (input_image_lh * 2) - 1
            input_image_rh = (input_image_rh * 2) - 1
            input_image_f = (input_image_f * 2) - 1

            input_body_and_fance_img = ((input_image_b + 1) + (input_image_f + 1)) - 1
            input_body_and_fance_img = torch.clip(input_body_and_fance_img, -1., 1.)

            input_image = torch.cat((input_body_and_fance_img, input_image_lh, input_image_rh), 0)
            input_image_list.append(input_image)
            pose = torch.from_numpy(pose / 720.)
            npy_list.append(pose)
            input_images_list.append(final_img_3c)
            if (idx + 1) % batch_size == 0 or len(rebuild_data_processed_list) - idx < batch_size:
                input_images = torch.stack(input_image_list, dim=0).cuda()
                npy_poses = torch.stack(npy_list, dim=0).cuda()
                npy_poses = npy_poses

                outputs = generate_images(generator, input_images, style_images, npy_poses)

                outputs = (outputs + 1.0) * 127.5
                outputs = outputs.detach().cpu()
                numpy_images = outputs.numpy()
                cv2_images = np.transpose(numpy_images, (0, 2, 3, 1))
                cv2_images_list = cv2_images
                cv2_images_list_l = [cv2_images_list[i] for i in range(cv2_images_list.shape[0])]
                outputs_list.extend(cv2_images_list_l)

                input_image_list = []
                npy_list = []

            inputs_sentence_Iidx = inputs_sentence_Iidx + 1

        idx_out = idx_out + 1
    # videoWrite.release()
    # videoWrite_input.release()
finish = True
