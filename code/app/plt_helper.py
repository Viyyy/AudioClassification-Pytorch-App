import matplotlib.pyplot as plt
import librosa

from io import BytesIO
from typing import List, Literal
from PIL import Image
import numpy as np
from copy import deepcopy

# 设置中文
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

from typing import List, Tuple, Dict
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 统一修改字体
plt.rcParams['font.family'] = ['STSong']


def get_draw_data(
    data: List[Tuple[str, float]], threshold: float = 0.05
) -> Dict[str, float]:
    result = {}
    for key, value in data:
        result[key] = value

    for key in list(result.keys()):
        if result[key] < threshold:
            result["其他"] = result.pop(key)

    # 如果其他太小，则删除
    if result["其他"] < 1e-6:
        result.pop("其他")

    return result


def get_labels_and_sizes(data: Dict[str, float]) -> Tuple[List[str], List[float]]:
    labels = list(data.keys())
    sizes = list(data.values())
    return labels, sizes


def get_colors(labels: List[str], color_dict: Dict[str, str]) -> List[str]:
    colors = [color_dict[label] for label in labels]
    return colors


def draw_legend(color_dict: Dict[str, str]):
    # 创建一个单独的图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in color_dict
    ]
    labels = list(color_dict.keys())

    legend = plt.figure(figsize=(1, int(len(color_dict) / 3)))
    plt.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5))

    plt.axis("off")  # 关闭图例的轴线

    return legend


def draw_pie(
    datas: List[List[Tuple[str, float]]],
    color_dict: Dict[str, str],
    width = 6,
    height = 6,
    fontsize: int = 10,
    threshold: float = 0.05,
):
    '''
    绘制饼图
    :param datas: 包含多个数据列表的列表
    :param color_dict: 颜色字典
    :param width: 图的宽度
    :param height: 图的高度
    :param threshold: 阈值，小于阈值的类别归为其他类别
    '''
    nrows = len(datas)

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(width, height * nrows))
    if nrows == 1:
        axes = [axes]

    for i, data in enumerate(datas):
        data = get_draw_data(data, threshold)
        labels, sizes = get_labels_and_sizes(data)
        colors = get_colors(labels, color_dict)
        ax = axes[i]
        ax.pie(
            sizes,
            labels=labels,
            startangle=90,
            colors=colors,
            pctdistance=0.8,
            labeldistance=1.1,
            textprops={"fontsize": fontsize},
            autopct="%1.1f%%",
        )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    return fig

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", fig_size=(8,4)):
    '''
    绘制频谱图
    :param specgram: 频谱图
    :param title: 图的标题
    :param ylabel: y轴的标签
    :param fig_size: 图的大小
    '''
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("frame")
    ax.imshow(librosa.power_to_db(specgram) + 80 ,origin="lower", aspect="auto", interpolation="nearest")
    
    ax.set_title(title) if title is not None else None
    
    fig.colorbar(mappable=ax.images[0], ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    
    return fig

def plt2ndarray(figure, *, format: Literal["jpg", "png"]="png", dpi:int=200):
    '''
    将 Matplotlib 图像转换为 NumPy 数组
    :param figure: Matplotlib 图像
    :param format: 图像格式，支持 jpg 和 png
    :param dpi: 图像分辨率
    :return: NumPy 数组
    '''
    assert format in ["jpg", "png"], f"format must be jpg or png, but got {format}"
    with BytesIO() as buf:
        figure.savefig(
            buf, format=format, dpi=dpi
        )  # 将 Matplotlib 图像保存到内存中的一个缓冲区
        buf.seek(0)
        image = Image.open(buf)  # 使用 Pillow 打开缓冲区中的图像
        image_array = np.array(image)  # 将 Pillow 图像转换为 NumPy 数组
        return deepcopy(image_array)  # jpg: RGB, png: RGBA