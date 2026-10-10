# -*- coding: utf-8 -*-
"""
可视化带图例 + 多层超点 + 预测正确性分析 + 迭代超点
+ Oracle预测 + 初始超点 + 特征PCA

适配最新的 EvalVisSaver：
按 Epoch 和 Scene 分层保存数据。
"""

import os
import math
import hashlib
from datetime import datetime

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


# ============================================================
# 1. 数据集、场景与实验路径配置
# ============================================================

# 支持：
#   "s3dis"
#   "dales"
dataset_name = "s3dis"

# 当前需要可视化的场景
scene_name = "Area_5-conferenceRoom_3"

# 当前需要可视化的 Epoch
target_epoch = 1


# -------------------- 根据实验选择一个路径 --------------------

# S3DIS
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/s3dis/s3dis-area5-best-test/eval_vis_compare"
# )
eval_vis_root = (
    "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/dales/dales-var-thresh0.003-1/eval_vis_compare"
)

# eval_vis_root = (
#     "/media/cvmaster/P106/SPNet-main/exp/s3dis/results/vis"
# )
# Toronto3D
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/toronto3d1/toronto3d-SPCNet/eval_vis_compare"
# )

# DALES
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/dales/dales-noaug-1-1-vis/eval_vis_compare"
# )

# ScanNet
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/scannet/scanenet-SPCNet-1/eval_vis_compare"
# )
# eval_vis_root = (
#     "/media/cvmaster/P106/PCA-Vis/eval_vis_compare"
# )
# SemanticKITTI
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/semantic_kitti/"
#     "semantic_kitti-noaug-1-vis/"
#     "eval_vis_compare"
# )

# KITTI360
# eval_vis_root = (
#     "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/"
#     "exp/kitti360/kitti360-noaug-1-1/eval_vis_compare"
# )

# NuScenes
# eval_vis_root = (
#     "/media/cvmaster/P106/Vis-Data/"
#     "nuscenes-noaug-1-vis/eval_vis_compare"
# )


# 可视化结果保存位置
BASE_DIR = (
    "/media/cvmaster/1C0329EB7F42121A/"
    "xlh/Vis/vis-eval-new"
)


# ============================================================
# 2. 可视化开关
# ============================================================

SHOW_SERIALIZE_PATH = False
SHOW_SERIALIZE_PATH_PREFIX = False

SERIALIZE_PATH_STRIDE = 20
SERIALIZE_PATH_MAX_POINTS = 3000
SERIALIZE_PREFIX_POINTS = 1000

SHOW_ORIGINAL_SCENE = True

SHOW_GT = True
SHOW_PRED = True
SHOW_ORACLE = False
SHOW_ACC = True

SHOW_INITIAL_AFFILIATION = True
SHOW_INIT_SEED = True

SHOW_PCA_FEAT = True
SHOW_SP_POINT_FEAT_PCA = True

SHOW_SUPERPOINT_LEVELS = True
SHOW_ITERATIVE = False

APPEND_LEGEND_GT_PRED = True
APPEND_LEGEND_ACC = True

# True：
# GT、Prediction 和 Oracle 使用类别并集生成同一张图例。
#
# False：
# 每张图只显示自身包含的类别。
USE_UNION_LEGEND = True


# ============================================================
# 3. 数据集类别和颜色配置
# ============================================================

ignore_index = -1

# ignore_index 对应颜色：灰色
IGNORE_COLOR_01 = np.array(
    [0.5, 0.5, 0.5],
    dtype=np.float64
)


DATASET_PALETTE = {

    # --------------------------------------------------------
    # S3DIS
    # --------------------------------------------------------
    "s3dis": {
        "num_classes": 20,

        "names": [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refridgerator",
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "otherfurniture",
        ],

        # Open3D 使用 0～1 颜色
        "colors": np.array([
            [0.6, 0.8, 0.6],
            [0.6, 0.7, 0.9],
            [0.2, 0.4, 0.7],
            [0.9, 0.7, 0.5],
            [0.7, 0.7, 0.4],
            [0.5, 0.3, 0.2],
            [0.9, 0.6, 0.6],
            [0.7, 0.3, 0.2],
            [0.8, 0.7, 0.9],
            [0.6, 0.4, 0.7],
            [0.5, 0.8, 0.8],
            [0.4, 0.6, 0.9],
            [0.9, 0.7, 0.8],
            [0.9, 0.9, 0.6],
            [0.9, 0.6, 0.3],
            [0.8, 0.6, 0.8],
            [0.7, 0.8, 0.8],
            [0.3, 0.6, 0.3],
            [0.5, 0.5, 0.6],
            [0.4, 0.4, 0.6],
        ], dtype=np.float64),
    },

    # --------------------------------------------------------
    # DALES
    # --------------------------------------------------------
    "dales": {
        "num_classes": 9,

        "names": [
            "Ground",
            "Vegetation",
            "Cars",
            "Trucks",
            "Power lines",
            "Fences",
            "Poles",
            "Buildings",
            "Unknown",
        ],

        # DALES / EZ-SPC 配色
        # 原始值为 RGB 0～255，这里转为 Open3D 使用的 0～1。
        "colors": np.array([
            [243, 214, 171],   # 0 Ground
            [70, 115, 66],     # 1 Vegetation
            [233, 50, 239],    # 2 Cars
            [243, 238, 0],     # 3 Trucks
            [190, 153, 153],   # 4 Power lines
            [0, 233, 11],      # 5 Fences
            [239, 114, 0],     # 6 Poles
            [214, 66, 54],     # 7 Buildings
            [0, 8, 116],       # 8 Unknown
        ], dtype=np.float64) / 255.0,
    },
    # --------------------------------------------------------
    # SemanticKITTI
    # --------------------------------------------------------
    "semantic_kitti": {
        "num_classes": 19,

        # 这里的顺序必须和训练时 segment / pred 中的类别 ID 一致。
        # 对应 Pointcept / SemanticKITTI 常用的 19 类学习标签：
        # 0～18
        "names": [
            "car",  # 0
            "bicycle",  # 1
            "motorcycle",  # 2
            "truck",  # 3
            "other-vehicle",  # 4
            "person",  # 5
            "bicyclist",  # 6
            "motorcyclist",  # 7
            "road",  # 8
            "parking",  # 9
            "sidewalk",  # 10
            "other-ground",  # 11
            "building",  # 12
            "fence",  # 13
            "vegetation",  # 14
            "trunk",  # 15
            "terrain",  # 16
            "pole",  # 17
            "traffic-sign",  # 18
        ],

        # 前 13 类沿用现有 S3DIS 调色板的前 13 种颜色；
        # 后 6 类使用额外补充的高区分度颜色。
        #
        # Open3D 使用 0～1 浮点颜色。
        "colors": np.array([
            [0.6, 0.8, 0.6],  # 0  car
            [0.6, 0.7, 0.9],  # 1  bicycle
            [0.2, 0.4, 0.7],  # 2  motorcycle
            [0.9, 0.7, 0.5],  # 3  truck
            [0.7, 0.7, 0.4],  # 4  other-vehicle
            [0.5, 0.3, 0.2],  # 5  person
            [0.9, 0.6, 0.6],  # 6  bicyclist
            [0.7, 0.3, 0.2],  # 7  motorcyclist
            [0.8, 0.7, 0.9],  # 8  road
            [0.6, 0.4, 0.7],  # 9  parking
            [0.5, 0.8, 0.8],  # 10 sidewalk
            [0.4, 0.6, 0.9],  # 11 other-ground
            [0.9, 0.7, 0.8],  # 12 building

            # 新增的后 6 类颜色
            [0.95, 0.85, 0.20],  # 13 fence
            [0.20, 0.70, 0.25],  # 14 vegetation
            [0.55, 0.35, 0.15],  # 15 trunk
            [0.60, 0.85, 0.45],  # 16 terrain
            [0.20, 0.85, 0.90],  # 17 pole
            [1.00, 0.25, 0.20],  # 18 traffic-sign
        ], dtype=np.float64),
    },

}


# ============================================================
# 4. 根据 dataset_name 自动选择类别和颜色
# ============================================================

dataset_key = dataset_name.strip().lower()

if dataset_key not in DATASET_PALETTE:
    raise ValueError(
        f"不支持的数据集：{dataset_name}\n"
        f"当前支持的数据集：{list(DATASET_PALETTE.keys())}"
    )

dataset_cfg = DATASET_PALETTE[dataset_key]

num_classes = int(dataset_cfg["num_classes"])
names = list(dataset_cfg["names"])
fixed_colors = np.asarray(
    dataset_cfg["colors"],
    dtype=np.float64
)

if len(names) != num_classes:
    raise ValueError(
        f"[{dataset_key}] 类别名称数量错误："
        f"num_classes={num_classes}，"
        f"names数量={len(names)}"
    )

if fixed_colors.shape != (num_classes, 3):
    raise ValueError(
        f"[{dataset_key}] 调色板尺寸错误："
        f"期望 ({num_classes}, 3)，"
        f"实际 {fixed_colors.shape}"
    )

fixed_colors = np.clip(
    fixed_colors,
    0.0,
    1.0
)

print(
    f"🎨 已加载数据集配色："
    f"{dataset_key}，类别数量={num_classes}"
)


# ============================================================
# 5. 预测正确性颜色
# ============================================================

CORRECT_COLOR_U8 = np.array(
    [0, 200, 0],
    dtype=np.uint8
)

WRONG_COLOR_U8 = np.array(
    [220, 20, 60],
    dtype=np.uint8
)

IGNORE_COLOR_ACC_U8 = np.array(
    [169, 169, 169],
    dtype=np.uint8
)


# ============================================================
# 6. 通用工具函数
# ============================================================

def stable_seed(*parts):
    """
    根据输入字符串生成稳定的随机种子。

    不直接使用 Python 的 hash()，因为不同进程中的 hash 值
    可能发生变化。
    """
    text = "|".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def validate_coords(coords, name="coords"):
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"{name} 应为 N×3，实际形状为 {coords.shape}"
        )

    return coords.astype(
        np.float64,
        copy=False
    )


def validate_labels(labels, point_count, name="labels"):
    labels = np.asarray(labels).reshape(-1)

    if labels.shape[0] != point_count:
        raise ValueError(
            f"{name} 点数不匹配："
            f"标签数={labels.shape[0]}，"
            f"坐标数={point_count}"
        )

    return labels.astype(
        np.int64,
        copy=False
    )


def compact_ids(arr):
    """
    将任意离散 ID 映射成连续 ID：0, 1, 2, ...
    """
    arr = np.asarray(arr).reshape(-1)
    _, inv = np.unique(
        arr,
        return_inverse=True
    )
    return inv.astype(np.int64), int(np.unique(arr).size)


def ids_present(
    arr,
    ignore_idx=None,
    class_count=None
):
    """
    获取当前标签数组中实际存在的合法类别 ID。
    """
    if arr is None:
        return []

    if ignore_idx is None:
        ignore_idx = ignore_index

    if class_count is None:
        class_count = num_classes

    arr = np.asarray(arr).reshape(-1)
    ids = np.unique(arr)

    valid = (
        (ids != ignore_idx)
        & (ids >= 0)
        & (ids < class_count)
    )

    return ids[valid].astype(int).tolist()


def _load_font(size=16):
    """
    加载图例字体。
    """
    candidate_fonts = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in candidate_fonts:
        try:
            return ImageFont.truetype(
                font_path,
                size
            )
        except Exception:
            continue

    return ImageFont.load_default()


# ============================================================
# 7. 图例相关函数
# ============================================================

def build_legend_image(
    ids,
    class_names,
    palette01,
    cols=5,
    swatch=28,
    gap=8,
    bg=(255, 255, 255)
):
    """
    根据类别 ID 生成图例。
    """
    ids = [
        int(cid)
        for cid in ids
        if 0 <= int(cid) < len(class_names)
    ]

    palette_u8 = (
        np.clip(
            palette01,
            0.0,
            1.0
        ) * 255
    ).astype(np.uint8)

    font = _load_font(size=16)

    # 没有合法类别时，返回小型空白图例
    if len(ids) == 0:
        legend = Image.new(
            "RGB",
            (300, 50),
            bg
        )
        draw = ImageDraw.Draw(legend)
        draw.text(
            (10, 15),
            "No valid class",
            fill=(0, 0, 0),
            font=font
        )
        return legend

    labels = [
        class_names[cid]
        for cid in ids
    ]

    tmp = Image.new(
        "RGB",
        (1, 1),
        bg
    )
    tmp_draw = ImageDraw.Draw(tmp)

    text_widths = []

    for label in labels:
        try:
            width = tmp_draw.textlength(
                label,
                font=font
            )
        except Exception:
            bbox = tmp_draw.textbbox(
                (0, 0),
                label,
                font=font
            )
            width = bbox[2] - bbox[0]

        text_widths.append(width)

    max_text_w = int(
        max(text_widths)
        if text_widths
        else 0
    )

    actual_cols = min(
        max(1, cols),
        len(ids)
    )

    cell_w = swatch + 8 + max_text_w
    cell_h = max(
        swatch,
        int(16 * 1.6)
    )

    rows = max(
        1,
        math.ceil(len(ids) / actual_cols)
    )

    img_w = (
        cell_w * actual_cols
        + gap * (actual_cols + 1)
    )

    img_h = (
        cell_h * rows
        + gap * (rows + 1)
    )

    legend = Image.new(
        "RGB",
        (img_w, img_h),
        bg
    )

    draw = ImageDraw.Draw(legend)

    for k, cid in enumerate(ids):
        row, col = divmod(
            k,
            actual_cols
        )

        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)

        color = tuple(
            palette_u8[cid].tolist()
        )

        draw.rectangle(
            [
                x,
                y,
                x + swatch,
                y + swatch,
            ],
            fill=color,
            outline=(0, 0, 0)
        )

        text_y = y + max(
            0,
            (swatch - 16) // 2
        )

        draw.text(
            (
                x + swatch + 6,
                text_y
            ),
            class_names[cid],
            fill=(0, 0, 0),
            font=font
        )

    return legend


def append_legend_below(
    image_path,
    legend_img,
    out_path=None,
    pad=12,
    bg=(255, 255, 255)
):
    """
    将图例拼接到可视化结果图片下方。
    """
    if not os.path.exists(image_path):
        print(
            f"⚠️ 图例拼接失败，找不到图片："
            f"{image_path}"
        )
        return

    base = Image.open(
        image_path
    ).convert("RGB")

    legend_img = legend_img.copy()

    if legend_img.width != base.width:
        ratio = base.width / float(
            legend_img.width
        )

        resized_height = max(
            1,
            int(round(
                legend_img.height * ratio
            ))
        )

        legend_img = legend_img.resize(
            (
                base.width,
                resized_height
            ),
            Image.Resampling.NEAREST
        )

    combined = Image.new(
        "RGB",
        (
            base.width,
            base.height + pad + legend_img.height
        ),
        bg
    )

    combined.paste(
        base,
        (0, 0)
    )

    combined.paste(
        legend_img,
        (
            0,
            base.height + pad
        )
    )

    if out_path is None:
        root, ext = os.path.splitext(
            image_path
        )
        out_path = root + "_wlegend" + ext

    combined.save(
        out_path,
        quality=95
    )

    print(
        f"✅ 图例图片保存完成：{out_path}"
    )


# ============================================================
# 8. Open3D 可视化函数
# ============================================================

def create_visualizer_fixed_colors(
    coords,
    labels,
    title,
    colors,
    class_count=None,
    ignore_idx=None,
    point_size=5
):
    """
    根据语义标签和固定调色板生成点云。
    """
    if class_count is None:
        class_count = len(colors)

    if ignore_idx is None:
        ignore_idx = ignore_index

    coords = validate_coords(
        coords,
        "coords"
    )

    labels = validate_labels(
        labels,
        coords.shape[0],
        "labels"
    )

    colors = np.asarray(
        colors,
        dtype=np.float64
    )

    if colors.shape != (class_count, 3):
        raise ValueError(
            f"颜色矩阵尺寸错误："
            f"期望 ({class_count}, 3)，"
            f"实际 {colors.shape}"
        )

    point_colors = np.tile(
        IGNORE_COLOR_01,
        (labels.shape[0], 1)
    )

    valid_mask = (
        (labels >= 0)
        & (labels < class_count)
    )

    point_colors[valid_mask] = colors[
        labels[valid_mask]
    ]

    # ignore_index 显示为灰色
    point_colors[
        labels == ignore_idx
    ] = IGNORE_COLOR_01

    invalid_mask = (
        (labels != ignore_idx)
        & (
            (labels < 0)
            | (labels >= class_count)
        )
    )

    if np.any(invalid_mask):
        invalid_ids = np.unique(
            labels[invalid_mask]
        )

        print(
            f"⚠️ {title} 中存在超出类别范围的标签："
            f"{invalid_ids.tolist()}。"
            f"这些点将显示为灰色。"
        )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        coords
    )

    pcd.colors = o3d.utility.Vector3dVector(
        np.clip(
            point_colors,
            0.0,
            1.0
        )
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)

    vis.get_render_option().point_size = (
        point_size
    )

    return vis, pcd


def create_visualizer_random_colors(
    coords,
    groups,
    title,
    point_size=7,
    seed=0
):
    """
    使用随机颜色可视化超点或分组。
    """
    coords = validate_coords(
        coords,
        "coords"
    )

    groups = validate_labels(
        groups,
        coords.shape[0],
        "groups"
    )

    # 允许输入不连续 ID，先压缩成连续 ID
    groups_compact, num_groups = compact_ids(
        groups
    )

    rng = np.random.default_rng(seed)

    if num_groups > 0:
        palette = rng.random(
            (num_groups, 3)
        )
    else:
        palette = np.zeros(
            (1, 3),
            dtype=np.float64
        )

    palette = np.clip(
        palette,
        0.0,
        1.0
    )

    if num_groups > 0:
        mapped = palette[
            groups_compact
        ]
    else:
        mapped = np.zeros(
            (coords.shape[0], 3),
            dtype=np.float64
        )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        coords
    )

    pcd.colors = o3d.utility.Vector3dVector(
        mapped
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)

    vis.get_render_option().point_size = (
        point_size
    )

    return vis, pcd


def create_visualizer_original_colors(
    coords,
    colors,
    title,
    point_size=5
):
    """
    使用原始 RGB 颜色创建点云。

    colors 可为：
        N×3，范围 0～255
    或：
        N×3，范围 0～1
    """
    coords = validate_coords(
        coords,
        "coords"
    )

    colors = np.asarray(
        colors,
        dtype=np.float64
    )

    if (
        colors.ndim != 2
        or colors.shape[1] != 3
    ):
        raise ValueError(
            f"colors 应为 N×3，"
            f"实际形状为 {colors.shape}"
        )

    if colors.shape[0] != coords.shape[0]:
        raise ValueError(
            f"颜色点数与坐标点数不一致："
            f"colors={colors.shape[0]}，"
            f"coords={coords.shape[0]}"
        )

    # 如果最大值大于 1，则认为颜色范围为 0～255。
    finite_colors = colors[
        np.isfinite(colors)
    ]

    if finite_colors.size > 0:
        color_max = float(
            np.max(finite_colors)
        )
    else:
        color_max = 0.0

    if color_max > 1.0:
        colors = colors / 255.0

    colors = np.nan_to_num(
        colors,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    colors = np.clip(
        colors,
        0.0,
        1.0
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        coords
    )

    pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)

    vis.get_render_option().point_size = (
        point_size
    )

    return vis, pcd


def create_visualizer_iterative_superpoints(
    coords,
    mapping,
    title,
    point_size=7,
    seed=0,
    palette=None
):
    """
    可视化迭代超点，并允许同一层不同迭代复用同一调色板。
    """
    coords = validate_coords(
        coords,
        "coords"
    )

    mapping = validate_labels(
        mapping,
        coords.shape[0],
        "mapping"
    )

    valid_mapping = mapping[
        mapping >= 0
    ]

    max_id = (
        int(valid_mapping.max())
        if valid_mapping.size > 0
        else -1
    )

    num_ids = max_id + 1

    if palette is None:
        rng = np.random.default_rng(seed)

        palette = rng.random(
            (max(1, num_ids), 3)
        )

    else:
        palette = np.asarray(
            palette,
            dtype=np.float64
        )

        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(
                f"palette 应为 K×3，"
                f"实际形状为 {palette.shape}"
            )

        if palette.shape[0] < max(1, num_ids):
            rng = np.random.default_rng(seed)

            extra_count = (
                max(1, num_ids)
                - palette.shape[0]
            )

            extra = rng.random(
                (extra_count, 3)
            )

            palette = np.vstack([
                palette,
                extra
            ])

    palette = np.clip(
        palette,
        0.0,
        1.0
    )

    point_colors = np.tile(
        IGNORE_COLOR_01,
        (mapping.shape[0], 1)
    )

    valid_mask = mapping >= 0

    if num_ids > 0:
        point_colors[valid_mask] = palette[
            mapping[valid_mask]
        ]

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        coords
    )

    pcd.colors = o3d.utility.Vector3dVector(
        point_colors
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)

    vis.get_render_option().point_size = (
        point_size
    )

    return vis, pcd, palette


def create_visualizer_with_path(
    full_coords,
    path_coords,
    title,
    point_color=(0.75, 0.75, 0.75),
    line_color=(1.0, 0.0, 0.0),
    point_size=3
):
    """
    full_coords：
        全部点云，用灰色显示背景。

    path_coords：
        按序列化顺序采样后的轨迹点，用折线连接。
    """
    full_coords = validate_coords(
        full_coords,
        "full_coords"
    )

    path_coords = validate_coords(
        path_coords,
        "path_coords"
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        full_coords
    )

    pcd.paint_uniform_color(
        point_color
    )

    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(
        path_coords
    )

    if len(path_coords) >= 2:
        lines = [
            [i, i + 1]
            for i in range(
                len(path_coords) - 1
            )
        ]

        line_colors = [
            line_color
            for _ in lines
        ]

    else:
        lines = []
        line_colors = []

    line_set.lines = o3d.utility.Vector2iVector(
        lines
    )

    line_set.colors = o3d.utility.Vector3dVector(
        line_colors
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    vis.get_render_option().point_size = (
        point_size
    )

    return vis, pcd, line_set


# ============================================================
# 9. PCA 工具函数
# ============================================================

def pca_features_to_rgb(features):
    """
    将 N×C 特征通过 PCA 降维为 N×3 RGB。

    返回范围：
        0～255
    """
    features = np.asarray(
        features,
        dtype=np.float64
    )

    if features.ndim != 2:
        raise ValueError(
            f"PCA特征应为 N×C，"
            f"实际形状为 {features.shape}"
        )

    num_points, num_channels = features.shape

    if num_points == 0:
        raise ValueError(
            "PCA特征为空，无法可视化。"
        )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # PCA 的成分数量不能超过样本数和特征维数。
    n_components = min(
        3,
        num_points,
        num_channels
    )

    if n_components <= 0:
        raise ValueError(
            f"无法执行PCA："
            f"特征形状={features.shape}"
        )

    pca = PCA(
        n_components=n_components
    )

    pca_feat = pca.fit_transform(
        features
    )

    # 若只能得到 1 或 2 个主成分，则补零到 3 通道。
    if n_components < 3:
        padding = np.zeros(
            (
                num_points,
                3 - n_components
            ),
            dtype=np.float64
        )

        pca_feat = np.concatenate(
            [
                pca_feat,
                padding
            ],
            axis=1
        )

    val_min = pca_feat.min(
        axis=0
    )

    val_max = pca_feat.max(
        axis=0
    )

    norm_feat = (
        pca_feat - val_min
    ) / (
        val_max - val_min + 1e-8
    )

    return np.clip(
        norm_feat * 255.0,
        0.0,
        255.0
    )


# ============================================================
# 10. 保存函数
# ============================================================

def save_vis_and_pcd(
    vis,
    pcd,
    output_dir,
    name
):
    vis.poll_events()
    vis.update_renderer()

    img_path = os.path.join(
        output_dir,
        f"{name}.png"
    )

    ply_path = os.path.join(
        output_dir,
        f"{name}.ply"
    )

    vis.capture_screen_image(
        img_path,
        do_render=True
    )

    o3d.io.write_point_cloud(
        ply_path,
        pcd,
        write_ascii=False,
        compressed=True
    )

    print(
        f"✅ 保存完成：{img_path}"
    )


def save_vis_with_path(
    vis,
    pcd,
    line_set,
    output_dir,
    name
):
    """
    保存序列化轨迹图片。

    PLY 文件只保存背景点云，
    折线不会写入 PointCloud 文件。
    """
    del line_set

    vis.poll_events()
    vis.update_renderer()

    img_path = os.path.join(
        output_dir,
        f"{name}.png"
    )

    ply_path = os.path.join(
        output_dir,
        f"{name}.ply"
    )

    vis.capture_screen_image(
        img_path,
        do_render=True
    )

    o3d.io.write_point_cloud(
        ply_path,
        pcd,
        write_ascii=False,
        compressed=True
    )

    print(
        f"✅ 保存完成：{img_path}"
    )


# ============================================================
# 11. 原始点云 RGB 路径
# ============================================================

def get_raw_scene_paths(
    current_dataset,
    current_scene
):
    """
    返回原始点云的 coord.npy 和 color.npy 路径。

    路径需要与你实际的数据预处理目录保持一致。
    """

    # --------------------------------------------------------
    # S3DIS
    # --------------------------------------------------------
    if current_dataset == "s3dis":

        # 优先使用 scene_name 形式，例如：
        # Area_5-office_1
        if "-" in current_scene:
            area_name, room_name = current_scene.split(
                "-",
                1
            )

            base_path = (
                "/media/cvmaster/1C0329EB7F42121A/"
                f"xlh/Data/s3dis/{area_name}/{room_name}"
            )

        else:
            # 保留你原来测试用的固定路径
            base_path = (
                "/media/cvmaster/1C0329EB7F42121A/"
                "xlh/Data/s3dis/Area_1/office_1"
            )

        return (
            os.path.join(
                base_path,
                "coord.npy"
            ),
            os.path.join(
                base_path,
                "color.npy"
            ),
        )

    # --------------------------------------------------------
    # DALES
    # --------------------------------------------------------
    if current_dataset == "dales":

        # DALES 预处理目录可能有 train、val 或 test。
        # 这里逐个查找。
        data_root = (
            "/media/cvmaster/1C0329EB7F42121A/"
            "xlh/Data/dales"
        )

        split_candidates = [
            "train",
            "val",
            "test",
        ]

        for split in split_candidates:
            base_path = os.path.join(
                data_root,
                split,
                current_scene
            )

            coord_path = os.path.join(
                base_path,
                "coord.npy"
            )

            color_path = os.path.join(
                base_path,
                "color.npy"
            )

            if os.path.exists(coord_path):
                return (
                    coord_path,
                    color_path
                )

        # 没找到时返回默认 val 路径，
        # 后面会自动检查文件是否存在。
        base_path = os.path.join(
            data_root,
            "val",
            current_scene
        )

        return (
            os.path.join(
                base_path,
                "coord.npy"
            ),
            os.path.join(
                base_path,
                "color.npy"
            ),
        )

    return None, None


# ============================================================
# 12. 构造场景目录并读取基本数据
# ============================================================

scene_dir = os.path.join(
    eval_vis_root,
    f"epoch_{target_epoch:03d}",
    scene_name
)

if not os.path.exists(scene_dir):
    raise FileNotFoundError(
        "找不到数据文件夹，请检查以下配置：\n"
        f"dataset_name = {dataset_name}\n"
        f"scene_name = {scene_name}\n"
        f"target_epoch = {target_epoch}\n"
        f"scene_dir = {scene_dir}"
    )


basic_data_path = os.path.join(
    scene_dir,
    "basic.npy"
)

if not os.path.exists(basic_data_path):
    raise FileNotFoundError(
        f"找不到 basic.npy："
        f"{basic_data_path}"
    )

basic_data = np.load(
    basic_data_path,
    allow_pickle=True
).item()


required_basic_keys = [
    "coord",
    "segment",
    "pred",
]

missing_basic_keys = [
    key
    for key in required_basic_keys
    if key not in basic_data
]

if missing_basic_keys:
    raise KeyError(
        f"basic.npy 缺少字段："
        f"{missing_basic_keys}"
    )


data_coord = np.asarray(
    basic_data["coord"]
)

data_gt = np.asarray(
    basic_data["segment"]
).reshape(-1).astype(np.int64)

data_pred = np.asarray(
    basic_data["pred"]
).reshape(-1).astype(np.int64)

data_oracle = basic_data.get(
    "oracle_pred",
    None
)

if data_oracle is not None:
    data_oracle = np.asarray(
        data_oracle
    ).reshape(-1).astype(np.int64)


data_initial_aff = basic_data.get(
    "initial_affiliation",
    None
)

data_init_seed_mask = basic_data.get(
    "init_seed_mask",
    None
)

data_init_seed_rank = basic_data.get(
    "init_seed_rank",
    None
)


data_serialize_rank_raw = basic_data.get(
    "serialize_rank_raw",
    None
)

data_serialize_order_raw = basic_data.get(
    "serialize_order_raw",
    None
)

data_serialize_coord = basic_data.get(
    "serialize_coord",
    None
)


# 检查坐标与标签点数
data_coord = validate_coords(
    data_coord,
    "data_coord"
)

data_gt = validate_labels(
    data_gt,
    data_coord.shape[0],
    "data_gt"
)

data_pred = validate_labels(
    data_pred,
    data_coord.shape[0],
    "data_pred"
)

if data_oracle is not None:
    data_oracle = validate_labels(
        data_oracle,
        data_coord.shape[0],
        "data_oracle"
    )


# 同步 ignore_index
data_pred = data_pred.copy()

data_pred[
    data_gt == ignore_index
] = ignore_index

if data_oracle is not None:
    data_oracle = data_oracle.copy()

    data_oracle[
        data_gt == ignore_index
    ] = ignore_index


print(
    f"📦 场景点数：{data_coord.shape[0]}"
)

print(
    f"🏷️ GT类别：{np.unique(data_gt).tolist()}"
)

print(
    f"🔮 Pred类别：{np.unique(data_pred).tolist()}"
)


# ============================================================
# 13. 加载原始 RGB 点云
# ============================================================

data_coord_color = None
data_color = None

if SHOW_ORIGINAL_SCENE:

    raw_coord_path, raw_color_path = (
        get_raw_scene_paths(
            dataset_key,
            scene_name
        )
    )

    coord_exists = (
        raw_coord_path is not None
        and os.path.exists(raw_coord_path)
    )

    color_exists = (
        raw_color_path is not None
        and os.path.exists(raw_color_path)
    )

    if coord_exists and color_exists:
        data_coord_color = np.load(
            raw_coord_path
        )

        data_color = np.load(
            raw_color_path
        )

        if (
            data_coord_color.shape[0]
            != data_color.shape[0]
        ):
            print(
                "⚠️ 原始坐标与颜色点数不同，"
                "已自动关闭 SHOW_ORIGINAL_SCENE。"
            )

            SHOW_ORIGINAL_SCENE = False

    else:
        print(
            "⚠️ 找不到原始 RGB 点云，"
            "已自动关闭 SHOW_ORIGINAL_SCENE。"
        )

        print(
            f"   coord路径：{raw_coord_path}"
        )

        print(
            f"   color路径：{raw_color_path}"
        )

        SHOW_ORIGINAL_SCENE = False


# ============================================================
# 14. 加载 Backbone 特征
# ============================================================

data_backbone_feat = None

if SHOW_PCA_FEAT:
    feat_data_path = os.path.join(
        scene_dir,
        "backbone_feat.npy"
    )

    if os.path.exists(feat_data_path):
        feat_data_dict = np.load(
            feat_data_path,
            allow_pickle=True
        ).item()

        data_backbone_feat = feat_data_dict.get(
            "feat",
            None
        )

        if data_backbone_feat is None:
            print(
                "⚠️ backbone_feat.npy 中没有 feat，"
                "已跳过 Backbone PCA。"
            )

            SHOW_PCA_FEAT = False

        else:
            data_backbone_feat = np.asarray(
                data_backbone_feat
            )

            if (
                data_backbone_feat.shape[0]
                != data_coord.shape[0]
            ):
                print(
                    "⚠️ Backbone特征点数与坐标点数不同："
                    f"{data_backbone_feat.shape[0]} != "
                    f"{data_coord.shape[0]}，"
                    "已跳过 Backbone PCA。"
                )

                SHOW_PCA_FEAT = False

    else:
        print(
            f"⚠️ 找不到 {feat_data_path}，"
            "已跳过 Backbone PCA。"
        )

        SHOW_PCA_FEAT = False


# ============================================================
# 15. 加载超点层级数据
# ============================================================

groups_per_level = []

if SHOW_SUPERPOINT_LEVELS:
    group_data_path = os.path.join(
        scene_dir,
        "group.npy"
    )

    if not os.path.exists(group_data_path):
        print(
            f"⚠️ 找不到 {group_data_path}，"
            "已关闭 SHOW_SUPERPOINT_LEVELS。"
        )

        SHOW_SUPERPOINT_LEVELS = False

    else:
        group_data = np.load(
            group_data_path,
            allow_pickle=True
        ).item()

        per_level_mappings = group_data.get(
            "per_level_mappings",
            None
        )

        if per_level_mappings is None:
            print(
                "⚠️ group.npy 中没有 "
                "per_level_mappings，"
                "已关闭 SHOW_SUPERPOINT_LEVELS。"
            )

            SHOW_SUPERPOINT_LEVELS = False

        else:
            print(
                f"🔢 检测到超点层数："
                f"{len(per_level_mappings)}"
            )

            for level, groups_raw in enumerate(
                per_level_mappings
            ):
                groups_raw = np.asarray(
                    groups_raw
                ).reshape(-1)

                if (
                    groups_raw.shape[0]
                    != data_coord.shape[0]
                ):
                    print(
                        f"⚠️ Level {level} 的映射长度错误："
                        f"{groups_raw.shape[0]} != "
                        f"{data_coord.shape[0]}，"
                        "已跳过该层。"
                    )
                    continue

                groups_compact, num_sp = compact_ids(
                    groups_raw
                )

                groups_per_level.append(
                    (
                        groups_compact,
                        num_sp
                    )
                )

                print(
                    f"   Level {level}："
                    f"超点数={num_sp}"
                )


# ============================================================
# 16. 加载迭代数据
# ============================================================

affiliations_history = []

if SHOW_ITERATIVE:
    iter_data_path = os.path.join(
        scene_dir,
        "iter.npy"
    )

    if os.path.exists(iter_data_path):
        data_iter = np.load(
            iter_data_path,
            allow_pickle=True
        )

        if isinstance(
            data_iter,
            np.ndarray
        ):
            affiliations_history = (
                data_iter.tolist()
            )
        else:
            affiliations_history = data_iter

        print(
            f"🔄 检测到迭代历史层数："
            f"{len(affiliations_history)}"
        )

    else:
        print(
            f"⚠️ 找不到 {iter_data_path}，"
            "已关闭 SHOW_ITERATIVE。"
        )

        SHOW_ITERATIVE = False


# ============================================================
# 17. 加载超点映射特征
# ============================================================

data_sp_point_feat = None

if SHOW_SP_POINT_FEAT_PCA:
    sp_point_feat_path = os.path.join(
        scene_dir,
        "sp_point_feat.npy"
    )

    if os.path.exists(sp_point_feat_path):
        sp_point_feat_dict = np.load(
            sp_point_feat_path,
            allow_pickle=True
        ).item()

        data_sp_point_feat = sp_point_feat_dict.get(
            "SPfeat_Pointfeat",
            None
        )

        if data_sp_point_feat is None:
            print(
                "⚠️ sp_point_feat.npy 中没有 "
                "SPfeat_Pointfeat，"
                "已跳过超点特征PCA。"
            )

            SHOW_SP_POINT_FEAT_PCA = False

        else:
            data_sp_point_feat = np.asarray(
                data_sp_point_feat
            )

            if (
                data_sp_point_feat.shape[0]
                != data_coord.shape[0]
            ):
                print(
                    "⚠️ 超点映射特征点数与坐标点数不同："
                    f"{data_sp_point_feat.shape[0]} != "
                    f"{data_coord.shape[0]}，"
                    "已跳过超点特征PCA。"
                )

                SHOW_SP_POINT_FEAT_PCA = False

    else:
        print(
            f"⚠️ 找不到 {sp_point_feat_path}，"
            "已跳过超点特征PCA。"
        )

        SHOW_SP_POINT_FEAT_PCA = False


# ============================================================
# 18. 生成输出目录
# ============================================================

def extract_experiment_name(
    eval_root_path
):
    normalized_path = os.path.normpath(
        eval_root_path
    )

    exp_dir = os.path.dirname(
        normalized_path
    )

    return os.path.basename(
        exp_dir
    )


experiment_name = extract_experiment_name(
    eval_vis_root
)

timestamp = datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    experiment_name,
    (
        f"Epoch{target_epoch}_"
        f"{scene_name}_"
        f"{timestamp}"
    )
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 19. 创建可视化结果
# ============================================================

all_vis = []


# ------------------------------------------------------------
# 00 原始 RGB 场景
# ------------------------------------------------------------

if (
    SHOW_ORIGINAL_SCENE
    and data_coord_color is not None
    and data_color is not None
):
    vis_origin, pcd_origin = (
        create_visualizer_original_colors(
            data_coord_color,
            data_color,
            f"Original Scene ({scene_name})",
            point_size=5
        )
    )

    save_vis_and_pcd(
        vis_origin,
        pcd_origin,
        OUTPUT_DIR,
        "00_original"
    )

    all_vis.append(
        vis_origin
    )


# ------------------------------------------------------------
# 01 Ground Truth
# ------------------------------------------------------------

if SHOW_GT:
    vis_groundtruth, pcd_gt = (
        create_visualizer_fixed_colors(
            data_coord,
            data_gt,
            f"Ground Truth ({scene_name})",
            fixed_colors,
            class_count=num_classes,
            ignore_idx=ignore_index,
            point_size=5
        )
    )

    save_vis_and_pcd(
        vis_groundtruth,
        pcd_gt,
        OUTPUT_DIR,
        "01_ground_truth"
    )

    all_vis.append(
        vis_groundtruth
    )


# ------------------------------------------------------------
# 02 Prediction
# ------------------------------------------------------------

if SHOW_PRED:
    vis_pred, pcd_pred = (
        create_visualizer_fixed_colors(
            data_coord,
            data_pred,
            f"Prediction ({scene_name})",
            fixed_colors,
            class_count=num_classes,
            ignore_idx=ignore_index,
            point_size=5
        )
    )

    save_vis_and_pcd(
        vis_pred,
        pcd_pred,
        OUTPUT_DIR,
        "02_prediction"
    )

    all_vis.append(
        vis_pred
    )


# ------------------------------------------------------------
# 02 Oracle Prediction
# ------------------------------------------------------------

if SHOW_ORACLE:
    if data_oracle is not None:
        vis_oracle, pcd_oracle = (
            create_visualizer_fixed_colors(
                data_coord,
                data_oracle,
                f"Oracle Prediction ({scene_name})",
                fixed_colors,
                class_count=num_classes,
                ignore_idx=ignore_index,
                point_size=5
            )
        )

        save_vis_and_pcd(
            vis_oracle,
            pcd_oracle,
            OUTPUT_DIR,
            "02_oracle_prediction"
        )

        all_vis.append(
            vis_oracle
        )

    else:
        print(
            "⚠️ basic.npy 中没有 oracle_pred，"
            "已跳过 Oracle 可视化。"
        )


# ------------------------------------------------------------
# 03 Prediction Accuracy
# ------------------------------------------------------------

if SHOW_ACC:
    num_points = data_coord.shape[0]

    acc_colors_u8 = np.tile(
        IGNORE_COLOR_ACC_U8,
        (num_points, 1)
    ).astype(np.uint8)

    valid_mask = (
        data_gt != ignore_index
    )

    correct_mask = (
        (data_pred == data_gt)
        & valid_mask
    )

    wrong_mask = (
        (data_pred != data_gt)
        & valid_mask
    )

    acc_colors_u8[
        correct_mask
    ] = CORRECT_COLOR_U8

    acc_colors_u8[
        wrong_mask
    ] = WRONG_COLOR_U8

    vis_acc, pcd_acc = (
        create_visualizer_original_colors(
            data_coord,
            acc_colors_u8,
            (
                "Accuracy "
                "(Green=Correct, Red=Wrong) "
                f"({scene_name})"
            ),
            point_size=5
        )
    )

    save_vis_and_pcd(
        vis_acc,
        pcd_acc,
        OUTPUT_DIR,
        "03_prediction_accuracy"
    )

    all_vis.append(
        vis_acc
    )


# ------------------------------------------------------------
# 04 Initial Affiliation
# ------------------------------------------------------------

if (
    SHOW_INITIAL_AFFILIATION
    and data_initial_aff is not None
):
    data_initial_aff = np.asarray(
        data_initial_aff
    ).reshape(-1)

    if (
        data_initial_aff.shape[0]
        != data_coord.shape[0]
    ):
        print(
            "⚠️ initial_affiliation 长度与坐标点数不一致，"
            "已跳过初始化分块可视化。"
        )

    else:
        aff_compact, num_sp_init = compact_ids(
            data_initial_aff
        )

        seed = stable_seed(
            scene_name,
            "INIT_AFF"
        )

        vis_init, pcd_init = (
            create_visualizer_random_colors(
                data_coord,
                aff_compact,
                (
                    "Initial Affiliation "
                    f"(N={num_sp_init})"
                ),
                point_size=7,
                seed=seed
            )
        )

        # seed 点高亮
        if (
            SHOW_INIT_SEED
            and data_init_seed_mask is not None
        ):
            seed_mask = np.asarray(
                data_init_seed_mask
            ).reshape(-1).astype(bool)

            if (
                seed_mask.shape[0]
                == data_coord.shape[0]
            ):
                init_colors = np.asarray(
                    pcd_init.colors
                ).copy()

                init_colors[
                    seed_mask
                ] = np.array(
                    [1.0, 0.0, 0.0]
                )

                pcd_init.colors = (
                    o3d.utility.Vector3dVector(
                        init_colors
                    )
                )

                vis_init.update_geometry(
                    pcd_init
                )

            else:
                print(
                    "⚠️ init_seed_mask 长度与点数不一致，"
                    "未进行 seed 高亮。"
                )

        save_vis_and_pcd(
            vis_init,
            pcd_init,
            OUTPUT_DIR,
            "04_initial_affiliation"
        )

        all_vis.append(
            vis_init
        )


# ------------------------------------------------------------
# 05 Backbone Feature PCA
# ------------------------------------------------------------

if (
    SHOW_PCA_FEAT
    and data_backbone_feat is not None
):
    print(
        "🚀 正在计算 Backbone 特征 PCA..."
    )

    try:
        pca_colors_u8 = pca_features_to_rgb(
            data_backbone_feat
        )

        vis_pca, pcd_pca = (
            create_visualizer_original_colors(
                data_coord,
                pca_colors_u8,
                (
                    "Backbone Feature PCA "
                    f"({scene_name})"
                ),
                point_size=5
            )
        )

        save_vis_and_pcd(
            vis_pca,
            pcd_pca,
            OUTPUT_DIR,
            "05_backbone_feat_pca"
        )

        all_vis.append(
            vis_pca
        )

        print(
            "✅ Backbone 特征 PCA 完成。"
        )

    except Exception as error:
        print(
            f"⚠️ Backbone PCA 失败：{error}"
        )


# ------------------------------------------------------------
# 05 SPfeat Pointfeat PCA
# ------------------------------------------------------------

if (
    SHOW_SP_POINT_FEAT_PCA
    and data_sp_point_feat is not None
):
    print(
        "🚀 正在计算 SPfeat_Pointfeat PCA..."
    )

    try:
        pca_colors_u8_sp = pca_features_to_rgb(
            data_sp_point_feat
        )

        vis_pca_sp, pcd_pca_sp = (
            create_visualizer_original_colors(
                data_coord,
                pca_colors_u8_sp,
                (
                    "SPfeat_Pointfeat PCA "
                    f"({scene_name})"
                ),
                point_size=5
            )
        )

        save_vis_and_pcd(
            vis_pca_sp,
            pcd_pca_sp,
            OUTPUT_DIR,
            "05_sp_point_feat_pca"
        )

        all_vis.append(
            vis_pca_sp
        )

        print(
            "✅ SPfeat_Pointfeat PCA 完成。"
        )

    except Exception as error:
        print(
            f"⚠️ SPfeat_Pointfeat PCA 失败：{error}"
        )


# ------------------------------------------------------------
# 06+ 多层超点
# ------------------------------------------------------------

if SHOW_SUPERPOINT_LEVELS:
    for level_index, (
        groups_level,
        num_sp_level
    ) in enumerate(groups_per_level):

        seed = stable_seed(
            scene_name,
            f"SP_LEVEL_{level_index}"
        )

        vis_sp, pcd_sp = (
            create_visualizer_random_colors(
                data_coord,
                groups_level,
                (
                    f"Superpoint Level {level_index} "
                    f"(N={num_sp_level})"
                ),
                point_size=7,
                seed=seed
            )
        )

        output_name = (
            f"{level_index + 6:02d}_"
            f"superpoint_level{level_index}"
        )

        save_vis_and_pcd(
            vis_sp,
            pcd_sp,
            OUTPUT_DIR,
            output_name
        )

        all_vis.append(
            vis_sp
        )


# ------------------------------------------------------------
# 序列化完整轨迹
# ------------------------------------------------------------

if (
    SHOW_SERIALIZE_PATH
    and data_serialize_coord is not None
):
    serial_coord = np.asarray(
        data_serialize_coord
    )

    if (
        serial_coord.ndim == 2
        and serial_coord.shape[1] == 3
    ):
        path_coords = serial_coord[
            ::SERIALIZE_PATH_STRIDE
        ]

        if (
            path_coords.shape[0]
            > SERIALIZE_PATH_MAX_POINTS
        ):
            path_coords = path_coords[
                :SERIALIZE_PATH_MAX_POINTS
            ]

        if path_coords.shape[0] >= 1:
            vis_serial, pcd_serial, line_serial = (
                create_visualizer_with_path(
                    full_coords=data_coord,
                    path_coords=path_coords,
                    title=(
                        f"Serialize Path ({scene_name}) "
                        f"stride={SERIALIZE_PATH_STRIDE}"
                    ),
                    point_color=(
                        0.8,
                        0.8,
                        0.8
                    ),
                    line_color=(
                        1.0,
                        0.0,
                        0.0
                    ),
                    point_size=3
                )
            )

            save_vis_with_path(
                vis_serial,
                pcd_serial,
                line_serial,
                OUTPUT_DIR,
                "05_serialize_path"
            )

            all_vis.append(
                vis_serial
            )


# ------------------------------------------------------------
# 序列化前缀轨迹
# ------------------------------------------------------------

if (
    SHOW_SERIALIZE_PATH_PREFIX
    and data_serialize_coord is not None
):
    serial_coord = np.asarray(
        data_serialize_coord
    )

    if (
        serial_coord.ndim == 2
        and serial_coord.shape[1] == 3
    ):
        prefix_coords = serial_coord[
            :SERIALIZE_PREFIX_POINTS
        ]

        if prefix_coords.shape[0] >= 2:
            vis_prefix, pcd_prefix, line_prefix = (
                create_visualizer_with_path(
                    full_coords=data_coord,
                    path_coords=prefix_coords,
                    title=(
                        f"Serialize Prefix Path "
                        f"({scene_name}) "
                        f"first "
                        f"{SERIALIZE_PREFIX_POINTS}"
                    ),
                    point_color=(
                        0.8,
                        0.8,
                        0.8
                    ),
                    line_color=(
                        0.0,
                        0.2,
                        1.0
                    ),
                    point_size=3
                )
            )

            save_vis_with_path(
                vis_prefix,
                pcd_prefix,
                line_prefix,
                OUTPUT_DIR,
                "06_serialize_prefix_path"
            )

            all_vis.append(
                vis_prefix
            )


# ------------------------------------------------------------
# 迭代超点
# ------------------------------------------------------------

if SHOW_ITERATIVE:
    for level, level_list in enumerate(
        affiliations_history
    ):
        if level_list is None:
            continue

        level_mappings = []

        for mapping in level_list:
            mapping = np.asarray(
                mapping
            ).reshape(-1)

            if (
                mapping.shape[0]
                == data_coord.shape[0]
            ):
                level_mappings.append(
                    mapping
                )

        if len(level_mappings) == 0:
            continue

        max_id_level = max(
            (
                int(mapping[mapping >= 0].max())
                if np.any(mapping >= 0)
                else -1
            )
            for mapping in level_mappings
        )

        seed_level = stable_seed(
            scene_name,
            f"ITER_LEVEL_{level}"
        )

        rng = np.random.default_rng(
            seed_level
        )

        palette_level = np.clip(
            rng.random(
                (
                    max(
                        1,
                        max_id_level + 1
                    ),
                    3
                )
            ),
            0.0,
            1.0
        )

        for iteration, mapping_iteration in enumerate(
            level_mappings
        ):
            vis_iteration, pcd_iteration, _ = (
                create_visualizer_iterative_superpoints(
                    data_coord,
                    mapping_iteration,
                    (
                        f"Iter Level {level} "
                        f"| Iter {iteration}"
                    ),
                    point_size=5,
                    seed=seed_level,
                    palette=palette_level
                )
            )

            output_name = (
                f"iter_level{level}_"
                f"iter{iteration}"
            )

            save_vis_and_pcd(
                vis_iteration,
                pcd_iteration,
                OUTPUT_DIR,
                output_name
            )

            all_vis.append(
                vis_iteration
            )


# ============================================================
# 20. 生成 GT、Pred 和 Oracle 图例
# ============================================================

if (
    APPEND_LEGEND_GT_PRED
    and (
        SHOW_GT
        or SHOW_PRED
        or SHOW_ORACLE
    )
):
    ids_gt = (
        ids_present(data_gt)
        if SHOW_GT
        else []
    )

    ids_pred = (
        ids_present(data_pred)
        if SHOW_PRED
        else []
    )

    ids_oracle = (
        ids_present(data_oracle)
        if (
            SHOW_ORACLE
            and data_oracle is not None
        )
        else []
    )

    if USE_UNION_LEGEND:
        ids_union = sorted(
            set(ids_gt)
            | set(ids_pred)
            | set(ids_oracle)
        )

        union_legend = build_legend_image(
            ids_union,
            names,
            fixed_colors,
            cols=5,
            swatch=28
        )

        if SHOW_GT:
            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "01_ground_truth.png"
                ),
                union_legend
            )

        if SHOW_PRED:
            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "02_prediction.png"
                ),
                union_legend
            )

        if (
            SHOW_ORACLE
            and data_oracle is not None
        ):
            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "02_oracle_prediction.png"
                ),
                union_legend
            )

    else:
        if SHOW_GT:
            legend_gt = build_legend_image(
                ids_gt,
                names,
                fixed_colors,
                cols=5,
                swatch=28
            )

            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "01_ground_truth.png"
                ),
                legend_gt
            )

        if SHOW_PRED:
            legend_pred = build_legend_image(
                ids_pred,
                names,
                fixed_colors,
                cols=5,
                swatch=28
            )

            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "02_prediction.png"
                ),
                legend_pred
            )

        if (
            SHOW_ORACLE
            and data_oracle is not None
        ):
            legend_oracle = build_legend_image(
                ids_oracle,
                names,
                fixed_colors,
                cols=5,
                swatch=28
            )

            append_legend_below(
                os.path.join(
                    OUTPUT_DIR,
                    "02_oracle_prediction.png"
                ),
                legend_oracle
            )


# ============================================================
# 21. 生成预测正确性图例
# ============================================================

if APPEND_LEGEND_ACC and SHOW_ACC:
    acc_legend_palette = np.array([
        CORRECT_COLOR_U8,
        WRONG_COLOR_U8,
        IGNORE_COLOR_ACC_U8,
    ], dtype=np.float64) / 255.0

    legend_acc_img = build_legend_image(
        [0, 1, 2],
        [
            "Correct",
            "Wrong",
            "Ignore",
        ],
        acc_legend_palette,
        cols=3,
        swatch=28
    )

    append_legend_below(
        os.path.join(
            OUTPUT_DIR,
            "03_prediction_accuracy.png"
        ),
        legend_acc_img
    )


# ============================================================
# 22. 启动所有 Open3D 窗口
# ============================================================

print(
    f"\n🎯 所有结果已保存至："
    f"{OUTPUT_DIR}"
)

if len(all_vis) == 0:
    print(
        "⚠️ 当前没有创建任何可视化窗口，"
        "程序直接结束。"
    )

else:
    print(
        "👀 正在显示可视化窗口。"
    )

    print(
        "关闭窗口或按 Q 键退出。"
    )

    while True:
        active_windows = []

        for visualizer in all_vis:
            try:
                is_active = visualizer.poll_events()
            except Exception:
                is_active = False

            active_windows.append(
                bool(is_active)
            )

        if not any(active_windows):
            break

        for visualizer, is_active in zip(
            all_vis,
            active_windows
        ):
            if is_active:
                visualizer.update_renderer()

    for visualizer in all_vis:
        try:
            visualizer.destroy_window()
        except Exception:
            pass

print(
    "👋 可视化结束。"
)
