# -*- coding: utf-8 -*-
"""
可视化带图例 + 多层超点 + 预测正确性分析 + 迭代超点 + Oracle预测 + 初始超点 + 特征PCA
- 适配最新的 EvalVisSaver (按 Epoch 和 Scene 分层保存) 的数据格式
"""

import os
from datetime import datetime
import math

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

# ====================================================
# [新增] 引入 PCA 用于将高维特征降维为 RGB 颜色
# ====================================================
from sklearn.decomposition import PCA

# ==================== 【可视化开关：你只需要改这里】 ====================
SHOW_SERIALIZE_PATH = False          # 显示序列化轨迹
SHOW_SERIALIZE_PATH_PREFIX = False   # 显示前缀轨迹（前一小段更清楚）
SERIALIZE_PATH_STRIDE = 20      # 每隔多少个序列点取一个
SERIALIZE_PATH_MAX_POINTS = 3000
SERIALIZE_PREFIX_POINTS = 1000  # 前缀轨迹长度

SHOW_ORIGINAL_SCENE = True  # 00_original (需要依赖原始数据集color.npy)
SHOW_GT = True  # 01_ground_truth
SHOW_PRED = True  # 02_prediction
SHOW_ORACLE = False  # 02_oracle_prediction
SHOW_ACC = True  # 03_prediction_accuracy

SHOW_INITIAL_AFFILIATION = True  # [新增] 04_initial_affiliation (显示初始化时的方形生硬分块)
SHOW_INIT_SEED = True  # 是否显示初始化种子点

SHOW_PCA_FEAT = True  # [新增] 05_backbone_feat_pca (显示特征PCA后的颜色)
SHOW_SP_POINT_FEAT_PCA = True  # [新增] 05_sp_point_feat_pca (显示映射回点云的超点特征PCA)v
SHOW_SUPERPOINT_LEVELS = True  # 06.. superpoint_level*
SHOW_ITERATIVE = False  # iter_level*_iter* (如果没有存iter数据，会自动跳过)

# 图例拼接也可单独控制（可选）
APPEND_LEGEND_GT_PRED = True
APPEND_LEGEND_ACC = True

# 若希望 GT 与 Pred 共用并集图例
USE_UNION_LEGEND = True
# =====================================================


# ==================== 类别与颜色映射 ====================
num_classes = 20
ignore_index = -1
names = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]

fixed_colors = np.array([
    [0.6, 0.8, 0.6], [0.6, 0.7, 0.9], [0.2, 0.4, 0.7], [0.9, 0.7, 0.5],
    [0.7, 0.7, 0.4], [0.5, 0.3, 0.2], [0.9, 0.6, 0.6], [0.7, 0.3, 0.2],
    [0.8, 0.7, 0.9], [0.6, 0.4, 0.7], [0.5, 0.8, 0.8], [0.4, 0.6, 0.9],
    [0.9, 0.7, 0.8], [0.9, 0.9, 0.6], [0.9, 0.6, 0.3], [0.8, 0.6, 0.8],
    [0.7, 0.8, 0.8], [0.3, 0.6, 0.3], [0.5, 0.5, 0.6], [0.4, 0.4, 0.6],
    [0.1, 0.1, 0.1]  # ignore = 黑色（末尾）
])

# ===== 预测正确性可视化颜色定义 (RGB 0-255) =====
CORRECT_COLOR_U8 = np.array([0, 200, 0])  # 绿色表示正确
WRONG_COLOR_U8 = np.array([220, 20, 60])  # 红色表示错误
IGNORE_COLOR_ACC_U8 = np.array([169, 169, 169])  # 灰色表示忽略区域


# ==================== 可视化与工具函数 (保持原样) ====================
def create_visualizer_iterative_superpoints(coords, mapping, title, point_size=7, seed=0, palette=None):
    coords = np.asarray(coords)
    mapping = np.asarray(mapping).reshape(-1)
    if mapping.shape[0] != coords.shape[0]:
        raise ValueError(f"[IterVis] mapping长度({mapping.shape[0]})与点数({coords.shape[0]})不一致")
    max_id = int(mapping.max()) if mapping.size > 0 else -1
    num_ids = max_id + 1

    if palette is None:
        rng = np.random.default_rng(seed)
        palette = rng.random((max(1, num_ids), 3))
        palette = np.clip(palette, 0, 1)
    else:
        palette = np.asarray(palette)
        if palette.shape[0] < max(1, num_ids):
            rng = np.random.default_rng(seed)
            extra = rng.random((max(1, num_ids) - palette.shape[0], 3))
            palette = np.vstack([palette, extra])
            palette = np.clip(palette, 0, 1)

    safe_mapping = mapping.copy()
    safe_mapping[safe_mapping < 0] = 0
    point_colors = palette[safe_mapping]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    return vis, pcd, palette


def compact_ids(arr: np.ndarray):
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64), uniq.size


def ids_present(arr, ignore_idx=-1, num_classes=20):
    ids = np.unique(arr)
    ids = ids[(ids != ignore_idx) & (ids >= 0) & (ids < num_classes)]
    return ids.tolist()


def _load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_legend_image(ids, class_names, palette01, cols=5, swatch=28, gap=8, bg=(255, 255, 255)):
    palette = (np.clip(palette01, 0, 1) * 255).astype(np.uint8)
    font = _load_font(size=16)
    labels = [class_names[i] for i in ids]
    _tmp = Image.new("RGB", (1, 1), bg)
    _draw = ImageDraw.Draw(_tmp)
    try:
        text_widths = [_draw.textlength(lbl, font=font) for lbl in labels]
    except Exception:
        text_widths = [_draw.textsize(lbl, font=font)[0] for lbl in labels]

    max_text_w = int(max(text_widths) if text_widths else 0)
    cell_w = swatch + 8 + max_text_w
    cell_h = max(swatch, int(font.size * 1.6))
    rows = max(1, math.ceil(len(ids) / max(1, cols)))
    img_w = cell_w * min(cols, max(1, len(ids))) + gap * (min(cols, max(1, len(ids))) + 1)
    img_h = cell_h * rows + gap * (rows + 1)

    legend = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(legend)
    for k, cid in enumerate(ids):
        r, c = divmod(k, cols)
        x = gap + c * (cell_w + gap)
        y = gap + r * (cell_h + gap)
        color = tuple(palette[cid].tolist()) if cid < len(palette) else (0, 0, 0)
        draw.rectangle([x, y, x + swatch, y + swatch], fill=color, outline=(0, 0, 0))
        ty = y + (swatch - font.size) // 2
        draw.text((x + swatch + 6, ty), class_names[cid], fill=(0, 0, 0), font=font)
    return legend


def append_legend_below(image_path, legend_img, out_path=None, pad=12, bg=(255, 255, 255)):
    if not os.path.exists(image_path):
        return
    base = Image.open(image_path).convert("RGB")
    if legend_img.width != base.width:
        ratio = base.width / float(legend_img.width)
        legend_img = legend_img.resize((base.width, int(round(legend_img.height * ratio))), Image.NEAREST)
    combined = Image.new("RGB", (base.width, base.height + pad + legend_img.height), bg)
    combined.paste(base, (0, 0))
    combined.paste(legend_img, (0, base.height + pad))
    if out_path is None:
        root, ext = os.path.splitext(image_path)
        out_path = root + "_wlegend" + ext
    combined.save(out_path, quality=95)


def create_visualizer_fixed_colors(coords, labels, title, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    point_colors = np.zeros((labels.shape[0], 3))
    for i in range(num_classes):
        point_colors[labels == i] = colors[i]
    point_colors[labels == ignore_index] = colors[-1]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 5
    return vis, pcd


def create_visualizer_random_colors(coords, groups, title, point_size=7, seed=0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    num_groups = int(np.max(groups)) + 1 if groups.size > 0 else 0
    rng = np.random.default_rng(seed)
    colors = rng.random((num_groups, 3)) if num_groups > 0 else np.zeros((1, 3))
    colors = np.clip(colors, 0, 1)
    mapped = colors[groups] if num_groups > 0 else np.zeros((coords.shape[0], 3))
    pcd.colors = o3d.utility.Vector3dVector(mapped)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    return vis, pcd


def create_visualizer_original_colors(coords, colors, title):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = colors / 255.0  # 注意：这里期望输入的是 0-255 的颜色矩阵
    colors = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 5
    return vis, pcd
#保存序列化顺序
def save_vis_with_path(vis, pcd, line_set, name):
    vis.poll_events()
    vis.update_renderer()
    img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    ply_path = os.path.join(OUTPUT_DIR, f"{name}.ply")
    vis.capture_screen_image(img_path, do_render=True)

    # 这里只保存背景点云，折线本身不写入 point cloud 文件
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False, compressed=True)
    print(f"✅ 保存完成: {img_path}")


def create_visualizer_with_path(
    full_coords,
    path_coords,
    title,
    point_color=(0.75, 0.75, 0.75),
    line_color=(1.0, 0.0, 0.0),
    point_size=3,
):
    """
    full_coords: 全部点云，用灰色显示背景
    path_coords: 按序列化顺序采样后的轨迹点，用折线连接
    """
    full_coords = np.asarray(full_coords)
    path_coords = np.asarray(path_coords)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_coords)
    pcd.paint_uniform_color(point_color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path_coords)

    if len(path_coords) >= 2:
        lines = [[i, i + 1] for i in range(len(path_coords) - 1)]
        colors = [line_color for _ in range(len(lines))]
    else:
        lines = []
        colors = []

    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.get_render_option().point_size = point_size

    return vis, pcd, line_set


# ====================================================
# [新增] 将特征通过PCA降维为RGB的工具函数
# ====================================================
def pca_features_to_rgb(features):
    """
    将 N x C 的特征矩阵，通过 PCA 降维到 N x 3，
    并归一化到 [0, 255] 以便当作颜色显示。
    """
    pca = PCA(n_components=3)
    # fit_transform 后特征变成 N x 3
    pca_feat = pca.fit_transform(features)

    # 按照每一个通道进行 Min-Max 归一化到 [0, 1]
    val_min = pca_feat.min(axis=0)
    val_max = pca_feat.max(axis=0)
    norm_feat = (pca_feat - val_min) / (val_max - val_min + 1e-8)

    # 转换到 [0, 255] 并返回
    return norm_feat * 255.0

#可视化结果还行scene0378_00,scene0025_00，scene0088_00，scene0153_00，scene0217_00,scene0221_00
# ==================== 【路径与数据读取配置】 ====================
dataset_name = "s3dis"          #5080_54400_new_1-3，5080_54400_new_2-4，5080_54400_new_2-6
scene_name = "Area_5-conferenceRoom_3"  # 你想查看的场景名#Area_5-conferenceRoom_1，Area_5-hallway_1，Area_5-office_1    #scene0011_00,scene0064_00,L002_block00000,L002_block00001，5080_54400_new_1-1，5080_54400_new_8-4
target_epoch = 1  # 想要查看的 epoch，例如 100 对应 epoch_100

# 新版的保存根目录
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/s3dis/s3dis-area5-best-vis/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/toronto3d1/toronto3d-SPCNet/eval_vis_compare"
eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/dales/dales-var-thresh0.003-1/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/scannet/scannet-noaug-1-3-vis/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/dales/dales-noaug-1-1-vis/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/semantic_kitti/semantic_kitti-encoder-cluster-pointmeta-crossblock/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/kitti360/kitti360-noaug-1-1/eval_vis_compare"
# eval_vis_root = "/media/cvmaster/P106/Vis-Data/nuscenes-noaug-1-vis/eval_vis_compare"

scene_dir = os.path.join(eval_vis_root, f"epoch_{target_epoch:03d}", scene_name)

if not os.path.exists(scene_dir):
    raise FileNotFoundError(f"找不到数据文件夹，请检查 Epoch 或场景名是否正确: {scene_dir}")

# 1. 加载基础数据 (coord, gt, pred, oracle_pred, initial_affiliation)
basic_data_path = os.path.join(scene_dir, "basic.npy")
basic_data = np.load(basic_data_path, allow_pickle=True).item()

data_coord = basic_data["coord"]
data_gt = basic_data["segment"]
data_pred = basic_data["pred"]
data_oracle = basic_data.get("oracle_pred", None)
data_initial_aff = basic_data.get("initial_affiliation", None)  # [新增] 获取初始化超点分配
data_init_seed_mask = basic_data.get("init_seed_mask", None)
data_init_seed_rank = basic_data.get("init_seed_rank", None)

#序列化顺序的数据读取
data_serialize_rank_raw = basic_data.get("serialize_rank_raw", None)
data_serialize_order_raw = basic_data.get("serialize_order_raw", None)
data_serialize_coord = basic_data.get("serialize_coord", None)

# 同步 ignore_index
data_pred[data_gt == ignore_index] = ignore_index
if data_oracle is not None:
    data_oracle[data_gt == ignore_index] = ignore_index

# 2. 加载原始RGB颜色 (如果需要看原始场景，依然从原数据集读)
if SHOW_ORIGINAL_SCENE:
    # raw_coord_path = f"/media/cvmaster/1C0329EB7F42121A/xlh/Data/{dataset_name}/val/{scene_name}/coord.npy"
    # raw_color_path = f"/media/cvmaster/1C0329EB7F42121A/xlh/Data/{dataset_name}/val/{scene_name}/color.npy"
    raw_coord_path = f"/media/cvmaster/1C0329EB7F42121A/xlh/Data/{dataset_name}/Area_1/office_1/coord.npy"

    raw_color_path = f"/media/cvmaster/1C0329EB7F42121A/xlh/Data/{dataset_name}/Area_1/office_1/color.npy"

    if os.path.exists(raw_color_path):
        data_coord_color = np.load(raw_coord_path)
        data_color = np.load(raw_color_path)
    else:
        print(f"⚠️ 找不到原始数据集 {raw_color_path}，已自动关闭 SHOW_ORIGINAL_SCENE")
        SHOW_ORIGINAL_SCENE = False

# [新增] 2.5 加载特征数据用于 PCA 可视化
data_backbone_feat = None
if SHOW_PCA_FEAT:
    feat_data_path = os.path.join(scene_dir, "backbone_feat.npy")
    if os.path.exists(feat_data_path):
        feat_data_dict = np.load(feat_data_path, allow_pickle=True).item()
        data_backbone_feat = feat_data_dict.get("feat", None)
    else:
        print(f"⚠️ 找不到 backbone_feat.npy，已自动跳过特征PCA可视化。")
        SHOW_PCA_FEAT = False

# 3. 加载超点层级数据 (group.npy)
group_data_path = os.path.join(scene_dir, "group.npy")
group_data = np.load(group_data_path, allow_pickle=True).item()
per_level_mappings = group_data["per_level_mappings"]

print(f"🔢 检测到超点层数: {len(per_level_mappings)}")
groups_per_level = []
for lvl, groups_raw in enumerate(per_level_mappings):
    groups_raw = np.asarray(groups_raw).reshape(-1)
    groups_compact, num_sp = compact_ids(groups_raw)
    groups_per_level.append((groups_compact, num_sp))
    print(f"  - Level {lvl}: 超点数 = {num_sp}")

# 4. 加载迭代数据 (iter.npy) (如果存在)
affiliations_history = []
if SHOW_ITERATIVE:
    iter_data_path = os.path.join(scene_dir, "iter.npy")
    if os.path.exists(iter_data_path):
        data_iter = np.load(iter_data_path, allow_pickle=True)
        affiliations_history = data_iter.tolist() if isinstance(data_iter, np.ndarray) else data_iter
        print(f"🔄 检测到迭代历史层数: {len(affiliations_history)}")
    else:
        print(f"⚠️ 找不到 iter.npy，已自动跳过迭代可视化。")
        SHOW_ITERATIVE = False
# [新增] 2.6 加载超点映射特征 (SPfeat_Pointfeat) 用于 PCA 可视化
data_sp_point_feat = None
if SHOW_SP_POINT_FEAT_PCA:
    sp_point_feat_path = os.path.join(scene_dir, "sp_point_feat.npy")
    if os.path.exists(sp_point_feat_path):
        sp_point_feat_dict = np.load(sp_point_feat_path, allow_pickle=True).item()
        data_sp_point_feat = sp_point_feat_dict.get("SPfeat_Pointfeat", None)
    else:
        print(f"⚠️ 找不到 sp_point_feat.npy，已自动跳过超点映射特征PCA可视化。")
        SHOW_SP_POINT_FEAT_PCA = False

# ==================== 动态提取实验名称 ====================
def extract_experiment_name(eval_root_path):
    exp_dir = os.path.dirname(eval_root_path)
    return os.path.basename(exp_dir)


experiment_name = extract_experiment_name(eval_vis_root)

# ==================== 保存输出路径配置 ====================
BASE_DIR = "/media/cvmaster/1C0329EB7F42121A/xlh/Vis/vis-eval-new"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

OUTPUT_DIR = os.path.join(BASE_DIR, experiment_name, f"Epoch{target_epoch}_{scene_name}_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_vis_and_pcd(vis, pcd, name):
    vis.poll_events()
    vis.update_renderer()
    img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    ply_path = os.path.join(OUTPUT_DIR, f"{name}.ply")
    vis.capture_screen_image(img_path, do_render=True)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False, compressed=True)
    print(f"✅ 保存完成: {img_path}")


# ==================== 创建可视化窗口 ====================
all_vis = []
vis_origin = vis_groundtruth = vis_pred = vis_oracle = vis_acc = None

if SHOW_ORIGINAL_SCENE:
    vis_origin, pcd_origin = create_visualizer_original_colors(
        data_coord_color, data_color, f"Original Scene ({scene_name})"
    )
    save_vis_and_pcd(vis_origin, pcd_origin, "00_original")
    all_vis.append(vis_origin)

if SHOW_GT:
    vis_groundtruth, pcd_gt = create_visualizer_fixed_colors(
        data_coord, data_gt, f"Ground Truth ({scene_name})", fixed_colors
    )
    save_vis_and_pcd(vis_groundtruth, pcd_gt, "01_ground_truth")
    all_vis.append(vis_groundtruth)

if SHOW_PRED:
    vis_pred, pcd_pred = create_visualizer_fixed_colors(
        data_coord, data_pred, f"Prediction ({scene_name})", fixed_colors
    )
    save_vis_and_pcd(vis_pred, pcd_pred, "02_prediction")
    all_vis.append(vis_pred)

if SHOW_ORACLE and data_oracle is not None:
    vis_oracle, pcd_oracle = create_visualizer_fixed_colors(
        data_coord, data_oracle, f"Oracle Prediction ({scene_name})", fixed_colors
    )
    save_vis_and_pcd(vis_oracle, pcd_oracle, "02_oracle_prediction")
    all_vis.append(vis_oracle)

if SHOW_ACC:
    num_points = data_coord.shape[0]
    acc_colors_u8 = np.tile(IGNORE_COLOR_ACC_U8, (num_points, 1)).astype(np.uint8)
    valid_mask = data_gt != ignore_index
    correct_mask = (data_pred == data_gt) & valid_mask
    wrong_mask = (data_pred != data_gt) & valid_mask
    acc_colors_u8[correct_mask] = CORRECT_COLOR_U8
    acc_colors_u8[wrong_mask] = WRONG_COLOR_U8

    vis_acc, pcd_acc = create_visualizer_original_colors(
        data_coord, acc_colors_u8, f"Accuracy (Green=Correct, Red=Wrong) ({scene_name})"
    )
    save_vis_and_pcd(vis_acc, pcd_acc, "03_prediction_accuracy")
    all_vis.append(vis_acc)

# ====================================================
# [增强] 初始分块 + seed 高亮 + seed rank
# ====================================================
if SHOW_INITIAL_AFFILIATION and data_initial_aff is not None:

    aff_compact, num_sp_init = compact_ids(data_initial_aff)
    seed = abs(hash((scene_name, "INIT_AFF"))) % (2 ** 32)

    vis_init, pcd_init = create_visualizer_random_colors(
        data_coord,
        aff_compact,
        f"Initial Affiliation (N={num_sp_init})",
        point_size=7,
        seed=seed
    )

    # ==============================
    # ⭐关键：seed 高亮叠加
    # ==============================
    if SHOW_INIT_SEED and data_init_seed_mask is not None:

        seed_mask = np.asarray(data_init_seed_mask).astype(bool)

        # 红色高亮 seed
        colors = np.asarray(pcd_init.colors)
        colors[seed_mask] = np.array([1.0, 0.0, 0.0])  # Open3D 是 0~1

        pcd_init.colors = o3d.utility.Vector3dVector(colors)

    save_vis_and_pcd(vis_init, pcd_init, "04_initial_affiliation")
    all_vis.append(vis_init)


# ====================================================
# [新增] 可视化特征 PCA 降维效果
# ====================================================
if SHOW_PCA_FEAT and data_backbone_feat is not None:
    print("🚀 正在计算 Backbone 特征的 PCA降维颜色...")
    pca_colors_u8 = pca_features_to_rgb(data_backbone_feat)
    vis_pca, pcd_pca = create_visualizer_original_colors(
        data_coord, pca_colors_u8, f"Backbone Feature PCA ({scene_name})"
    )
    save_vis_and_pcd(vis_pca, pcd_pca, "05_backbone_feat_pca")
    all_vis.append(vis_pca)
    print("✅ PCA 特征颜色计算并挂载完毕！")
# ====================================================
# [新增] 可视化 超点映射特征(SPfeat_Pointfeat) PCA 降维效果
# ====================================================
if SHOW_SP_POINT_FEAT_PCA and data_sp_point_feat is not None:
    print("�� 正在计算 超点映射特征(SPfeat_Pointfeat) 的 PCA降维颜色...")
    pca_colors_u8_sp = pca_features_to_rgb(data_sp_point_feat)
    vis_pca_sp, pcd_pca_sp = create_visualizer_original_colors(
        data_coord, pca_colors_u8_sp, f"SPfeat_Pointfeat PCA ({scene_name})"
    )
    # 为了方便文件排序，命名为 05_sp_point_feat_pca
    save_vis_and_pcd(vis_pca_sp, pcd_pca_sp, "05_sp_point_feat_pca")
    all_vis.append(vis_pca_sp)
    print("✅ SPfeat_Pointfeat PCA 特征颜色计算并挂载完毕！")
if SHOW_SUPERPOINT_LEVELS:
    for idx, (groups_lvl, num_sp_lvl) in enumerate(groups_per_level):
        seed = abs(hash((scene_name, f"SP_LEVEL_{idx}"))) % (2 ** 32)
        vis_sp, pcd_sp = create_visualizer_random_colors(
            data_coord, groups_lvl, f"Superpoint Level {idx} (N={num_sp_lvl})", point_size=7, seed=seed
        )
        name = f"{idx + 6:02d}_superpoint_level{idx}"  # 名字加了点偏置，排在04,05后面
        save_vis_and_pcd(vis_sp, pcd_sp, name)
        all_vis.append(vis_sp)

# ====================================================
# [新增] 可视化序列化轨迹
# ====================================================
if SHOW_SERIALIZE_PATH and data_serialize_coord is not None:
    serial_coord = np.asarray(data_serialize_coord)

    path_coords = serial_coord[::SERIALIZE_PATH_STRIDE]
    if path_coords.shape[0] > SERIALIZE_PATH_MAX_POINTS:
        path_coords = path_coords[:SERIALIZE_PATH_MAX_POINTS]

    vis_serial, pcd_serial, line_serial = create_visualizer_with_path(
        full_coords=data_coord,
        path_coords=path_coords,
        title=f"Serialize Path ({scene_name}) stride={SERIALIZE_PATH_STRIDE}",
        point_color=(0.8, 0.8, 0.8),
        line_color=(1.0, 0.0, 0.0),
        point_size=3,
    )
    save_vis_with_path(vis_serial, pcd_serial, line_serial, "05_serialize_path")
    all_vis.append(vis_serial)

# ====================================================
# [新增] 可视化序列化前缀轨迹（更清楚）
# ====================================================
if SHOW_SERIALIZE_PATH_PREFIX and data_serialize_coord is not None:
    serial_coord = np.asarray(data_serialize_coord)

    prefix_coords = serial_coord[:SERIALIZE_PREFIX_POINTS]
    if prefix_coords.shape[0] >= 2:
        vis_prefix, pcd_prefix, line_prefix = create_visualizer_with_path(
            full_coords=data_coord,
            path_coords=prefix_coords,
            title=f"Serialize Prefix Path ({scene_name}) first {SERIALIZE_PREFIX_POINTS}",
            point_color=(0.8, 0.8, 0.8),
            line_color=(0.0, 0.2, 1.0),
            point_size=3,
        )
        save_vis_with_path(vis_prefix, pcd_prefix, line_prefix, "06_serialize_prefix_path")
        all_vis.append(vis_prefix)



if SHOW_ITERATIVE:
    for lvl, lvl_list in enumerate(affiliations_history):
        max_id_lvl = max([int(np.asarray(m).max()) for m in lvl_list if np.asarray(m).size > 0], default=-1)
        seed_lvl = abs(hash((scene_name, f"ITER_LEVEL_{lvl}"))) % (2 ** 32)
        rng = np.random.default_rng(seed_lvl)
        palette_lvl = np.clip(rng.random((max(1, max_id_lvl + 1), 3)), 0, 1)

        for it, mapping_it in enumerate(lvl_list):
            vis_it, pcd_it, _ = create_visualizer_iterative_superpoints(
                data_coord, mapping_it, f"Iter Level {lvl} | Iter {it}",
                point_size=5, seed=seed_lvl, palette=palette_lvl
            )
            name = f"iter_level{lvl}_iter{it}"
            save_vis_and_pcd(vis_it, pcd_it, name)
            all_vis.append(vis_it)

# ==================== 生成与拼接图例 ====================
if APPEND_LEGEND_GT_PRED and (SHOW_GT or SHOW_PRED or SHOW_ORACLE):
    ids_gt = ids_present(data_gt) if SHOW_GT else []
    ids_prd = ids_present(data_pred) if SHOW_PRED else []
    ids_oracle = ids_present(data_oracle) if (SHOW_ORACLE and data_oracle is not None) else []

    ids_union = sorted(set(ids_gt) | set(ids_prd) | set(ids_oracle))
    legend_img = build_legend_image(ids_union, names, fixed_colors, cols=5, swatch=28)

    if SHOW_GT: append_legend_below(os.path.join(OUTPUT_DIR, "01_ground_truth.png"), legend_img)
    if SHOW_PRED: append_legend_below(os.path.join(OUTPUT_DIR, "02_prediction.png"), legend_img)
    if SHOW_ORACLE and data_oracle is not None:
        append_legend_below(os.path.join(OUTPUT_DIR, "02_oracle_prediction.png"), legend_img)

if APPEND_LEGEND_ACC and SHOW_ACC:
    acc_legend_palette = np.array([CORRECT_COLOR_U8, WRONG_COLOR_U8, IGNORE_COLOR_ACC_U8]) / 255.0
    legend_acc_img = build_legend_image([0, 1, 2], ["Correct", "Wrong", "Ignore"], acc_legend_palette, cols=3)
    append_legend_below(os.path.join(OUTPUT_DIR, "03_prediction_accuracy.png"), legend_acc_img)

# ==================== 启动所有窗口事件循环 ====================
print(f"\n🎯 所有结果已保存至: {OUTPUT_DIR}")
print("👀 正在显示可视化窗口 (按下键盘 Q 键 或 关闭窗口退出)...")

while all(v.poll_events() for v in all_vis):
    for v in all_vis:
        v.update_renderer()

for v in all_vis:
    v.destroy_window()

print("👋 可视化结束。")