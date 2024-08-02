import os
import yaml
import random

COLOR_YAML_PATH = os.path.join(os.path.dirname(__file__), "colors_unique.yaml")


def __get_colors_configs() -> dict:
    with open(COLOR_YAML_PATH, "r") as f:
        colors = yaml.load(f, Loader=yaml.FullLoader)
    return colors


def get_color_dict(lst: list, random_state: int = 1202, add_else: bool = True) -> dict:
    """
    随机生成颜色字典
    :param lst: 列表
    :param random_state: 随机种子
    :param add_else: 是否添加其他的颜色
    :return: 颜色字典
    """
    try:
        random.seed(random_state)  # 设置随机种子
        colors_configs = __get_colors_configs()  # 读取颜色配置
        len_colors = len(colors_configs)
        colors_keys = list(colors_configs.keys())
        colors = []
        idx = 0
        while idx < len(lst):
            random_color_idx = random.randint(0, len_colors - 1)  # 随机选择一种颜色
            random_color_name = colors_keys[random_color_idx]  # 随机选择一种颜色
            if (
                "black" in random_color_name.lower()
                or "white" in random_color_name.lower()
                or "gray" in random_color_name.lower()
                or "dark" in random_color_name.lower()
            ):  # 不要选择黑白灰暗的颜色
                continue
            random_color = colors_configs[random_color_name]  # 随机选择一种颜色
            len_colors1 = len(random_color)  # 随机颜色的数量
            random_color_idx1 = random.randint(0, len_colors1 - 1)  # 随机选择一种颜色
            random_color1 = random_color[random_color_idx1]
            if random_color1 not in colors:  # 保证颜色不重复
                colors.append(random_color1)
                idx += 1
        if add_else:
            lst.append("其他")
            colors.append(colors_configs['LightGray'][0])
        return {lst[i]: colors[i] for i in range(len(lst))}
    except Exception as e:
        print(e)
        return {}
    finally:
        random.seed()
