# -*- coding: utf-8 -*-
"""
Test阶段保存数据可视化脚本
- 读取 SemSegTester 保存的 original / result / superpoint_result
- 可视化最终超点、初始化 Hilbert block、初始化种子点、seed rank
- 不需要命令行输入，直接修改 SCENE_NAME 后运行即可
"""

import os
from datetime import datetime
import math

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 【只需要改这里】
# ============================================================

SCENE_NAME = "Area_5-conferenceRoom_3"
# 例如：
# SCENE_NAME = "Area_5-office_1"
# SCENE_NAME = "Area_5-hallway_1"

SUPERPOINT_RESULT_ROOT = "/media/cvmaster/1C0329EB7F42121A/xlh/gjl/Pointcept/exp/s3dis/s3dis-area5-best-test/result"

# 输出可视化结果保存位置
BASE_OUTPUT_DIR = "/media/cvmaster/1C0329EB7F42121A/xlh/Vis/test_init_seed_vis"

# 是否显示窗口
SHOW_WINDOWS = True

# 是否保存 png 和 ply
SAVE_RESULT = True

# 点大小
POINT_SIZE = 5
SUPERPOINT_POINT_SIZE = 7
SEED_POINT_SIZE = 10

# 可视化开关
SHOW_GT = True
SHOW_PRED = True
SHOW_ACC = True
SHOW_FINAL_SUPERPOINT = True

SHOW_INIT_BLOCK = False
SHOW_INIT_BLOCK_SEED_RED = False
SHOW_INIT_BLOCK_SEED_RANK = False
SHOW_SEEDS_ONLY = False

# 是否给 seed 额外加小球，论文图更明显，但点多时会慢
ADD_SEED_SPHERES = False
SEED_SPHERE_RADIUS = 0.04

# ============================================================


# ============================================================
# S3DIS 类别与颜色
# ============================================================

num_classes = 13
ignore_index = -1

names = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

# fixed_colors = np.array([
#     [0.60, 0.80, 0.60],  # ceiling
#     [0.60, 0.70, 0.90],  # floor
#     [0.70, 0.70, 0.70],  # wall
#     [0.80, 0.55, 0.30],  # beam
#     [0.65, 0.45, 0.25],  # column
#     [0.50, 0.80, 0.90],  # window
#     [0.75, 0.35, 0.20],  # door
#     [0.90, 0.60, 0.60],  # table
#     [0.90, 0.80, 0.30],  # chair
#     [0.55, 0.35, 0.25],  # sofa
#     [0.65, 0.45, 0.75],  # bookcase
#     [0.20, 0.55, 0.90],  # board
#     [0.40, 0.40, 0.40],  # clutter
#     [0.05, 0.05, 0.05],  # ignore
# ], dtype=np.float32)
fixed_colors = np.array([
    [0.6, 0.8, 0.6], [0.6, 0.7, 0.9], [0.2, 0.4, 0.7], [0.9, 0.7, 0.5],
    [0.7, 0.7, 0.4], [0.5, 0.3, 0.2], [0.9, 0.6, 0.6], [0.7, 0.3, 0.2],
    [0.8, 0.7, 0.9], [0.6, 0.4, 0.7], [0.5, 0.8, 0.8], [0.4, 0.6, 0.9],
    [0.9, 0.7, 0.8], [0.9, 0.9, 0.6], [0.9, 0.6, 0.3], [0.8, 0.6, 0.8],
    [0.7, 0.8, 0.8], [0.3, 0.6, 0.3], [0.5, 0.5, 0.6], [0.4, 0.4, 0.6],
    [0.1, 0.1, 0.1]  # ignore
], dtype=np.float32)

CORRECT_COLOR_U8 = np.array([0, 200, 0], dtype=np.uint8)
WRONG_COLOR_U8 = np.array([220, 20, 60], dtype=np.uint8)
IGNORE_COLOR_ACC_U8 = np.array([169, 169, 169], dtype=np.uint8)


# ============================================================
# 路径自动推导
# ============================================================

EXP_ROOT = os.path.dirname(SUPERPOINT_RESULT_ROOT)
ORIGINAL_ROOT = os.path.join(EXP_ROOT, "original")
RESULT_ROOT = os.path.join(EXP_ROOT, "result")

GROUP_PATH = os.path.join(SUPERPOINT_RESULT_ROOT, f"{SCENE_NAME}_group.npy")
ORIGINAL_PATH = os.path.join(ORIGINAL_ROOT, f"{SCENE_NAME}_original.npy")
PRED_PATH = os.path.join(RESULT_ROOT, f"{SCENE_NAME}_pred.npy")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, SCENE_NAME + "_" + timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================

def load_npy_dict(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    return data


def normalize_color01(colors):
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size > 0 and colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def compact_ids_keep_invalid(ids, invalid_value=-1):
    """
    将任意超点 / block id 压缩成连续 id，避免 id offset 后颜色表过大。
    invalid_value 保留为 -1。
    """
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    valid = ids != invalid_value

    compact = np.full_like(ids, fill_value=invalid_value, dtype=np.int64)

    if valid.sum() == 0:
        return compact, 0

    uniq, inv = np.unique(ids[valid], return_inverse=True)
    compact[valid] = inv.astype(np.int64)

    return compact, len(uniq)


def make_random_palette(num_ids, seed=0):
    rng = np.random.default_rng(seed)
    palette = rng.random((max(1, num_ids), 3)).astype(np.float32)
    palette = palette * 0.75 + 0.20
    return np.clip(palette, 0.0, 1.0)


def colors_by_ids(ids, seed=0, invalid_color=(0.25, 0.25, 0.25)):
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    valid = ids >= 0

    colors = np.zeros((ids.shape[0], 3), dtype=np.float32)
    colors[:] = np.asarray(invalid_color, dtype=np.float32)

    if valid.sum() == 0:
        return colors

    num_ids = int(ids[valid].max()) + 1
    palette = make_random_palette(num_ids, seed=seed)
    colors[valid] = palette[ids[valid]]

    return colors


def create_pcd(coords, colors):
    coords = np.asarray(coords, dtype=np.float32)
    colors = normalize_color01(colors)

    if coords.shape[0] != colors.shape[0]:
        raise ValueError(f"点数和颜色数量不一致: coord={coords.shape}, colors={colors.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_visualizer(coords, colors, title, point_size=5, extra_geometries=None):
    pcd = create_pcd(coords, colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=960)
    vis.add_geometry(pcd)

    if extra_geometries is not None:
        for geo in extra_geometries:
            vis.add_geometry(geo)

    vis.get_render_option().point_size = point_size
    return vis, pcd


def save_vis_and_pcd(vis, pcd, name):
    if not SAVE_RESULT:
        return

    vis.poll_events()
    vis.update_renderer()

    img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    ply_path = os.path.join(OUTPUT_DIR, f"{name}.ply")

    vis.capture_screen_image(img_path, do_render=True)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False, compressed=True)

    print(f"✅ 保存完成: {img_path}")
    print(f"✅ 保存完成: {ply_path}")


def create_seed_spheres(coords, seed_mask, color=(1.0, 0.0, 0.0), radius=0.04):
    coords = np.asarray(coords, dtype=np.float32)
    seed_mask = np.asarray(seed_mask).reshape(-1).astype(bool)

    seed_coords = coords[seed_mask]
    spheres = []

    for p in seed_coords:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        spheres.append(sphere)

    return spheres


def create_visualizer_fixed_labels(coords, labels, title, point_size=5):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    for cid in range(num_classes):
        colors[labels == cid] = fixed_colors[cid]

    colors[labels == ignore_index] = fixed_colors[-1]

    return create_visualizer(coords, colors, title, point_size=point_size)


def _load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_legend_image(ids, class_names, palette01, cols=5, swatch=28, gap=8, bg=(255, 255, 255)):
    palette = (np.clip(palette01, 0, 1) * 255).astype(np.uint8)
    font = _load_font(size=16)

    labels = [class_names[i] for i in ids]

    tmp = Image.new("RGB", (1, 1), bg)
    draw = ImageDraw.Draw(tmp)

    try:
        text_widths = [draw.textlength(lbl, font=font) for lbl in labels]
    except Exception:
        text_widths = [draw.textsize(lbl, font=font)[0] for lbl in labels]

    max_text_w = int(max(text_widths) if text_widths else 0)
    cell_w = swatch + 8 + max_text_w
    cell_h = max(swatch, int(font.size * 1.6))

    rows = max(1, math.ceil(len(ids) / max(1, cols)))
    used_cols = min(cols, max(1, len(ids)))

    img_w = cell_w * used_cols + gap * (used_cols + 1)
    img_h = cell_h * rows + gap * (rows + 1)

    legend = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(legend)

    for k, cid in enumerate(ids):
        r, c = divmod(k, cols)
        x = gap + c * (cell_w + gap)
        y = gap + r * (cell_h + gap)

        color = tuple(palette[cid].tolist())
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
    print(f"✅ 图例拼接完成: {out_path}")


# ============================================================
# 颜色生成函数
# ============================================================

def color_prediction_accuracy(gt, pred):
    gt = np.asarray(gt).reshape(-1).astype(np.int64)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)

    colors_u8 = np.tile(IGNORE_COLOR_ACC_U8, (gt.shape[0], 1)).astype(np.uint8)

    valid = gt != ignore_index
    correct = (gt == pred) & valid
    wrong = (gt != pred) & valid

    colors_u8[correct] = CORRECT_COLOR_U8
    colors_u8[wrong] = WRONG_COLOR_U8

    return colors_u8.astype(np.float32) / 255.0


def color_init_block(init_block_id, scene_name):
    init_block_compact, num_blocks = compact_ids_keep_invalid(init_block_id)
    seed = abs(hash((scene_name, "INIT_BLOCK"))) % (2 ** 32)
    colors = colors_by_ids(init_block_compact, seed=seed)
    return colors, init_block_compact, num_blocks


def color_init_block_seed_red(init_block_id, seed_mask, scene_name):
    colors, init_block_compact, num_blocks = color_init_block(init_block_id, scene_name)

    seed_mask = np.asarray(seed_mask).reshape(-1).astype(bool)
    colors[seed_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return colors, init_block_compact, num_blocks


def color_init_block_seed_rank(init_block_id, seed_rank, scene_name):
    colors, init_block_compact, num_blocks = color_init_block(init_block_id, scene_name)

    seed_rank = np.asarray(seed_rank).reshape(-1).astype(np.int64)
    valid_seed = seed_rank >= 0

    seed_palette = np.array([
        [1.0, 0.0, 0.0],    # rank 0 red
        [1.0, 1.0, 0.0],    # rank 1 yellow
        [0.0, 0.2, 1.0],    # rank 2 blue
        [1.0, 0.0, 1.0],    # rank 3 magenta
        [0.0, 1.0, 1.0],    # rank 4 cyan
        [1.0, 0.5, 0.0],    # rank 5 orange
        [0.0, 1.0, 0.0],    # rank 6 green
        [0.6, 0.2, 1.0],    # rank 7 purple
        [1.0, 0.4, 0.6],    # rank 8 pink
        [0.4, 1.0, 0.4],    # rank 9 light green
    ], dtype=np.float32)

    colors[valid_seed] = seed_palette[seed_rank[valid_seed] % len(seed_palette)]

    return colors, init_block_compact, num_blocks


def color_final_superpoint(sp_index, scene_name):
    sp_compact, num_sp = compact_ids_keep_invalid(sp_index)
    seed = abs(hash((scene_name, "FINAL_SP"))) % (2 ** 32)
    colors = colors_by_ids(sp_compact, seed=seed)
    return colors, sp_compact, num_sp


# ============================================================
# 主流程
# ============================================================

print("\n========== 路径检查 ==========")
print(f"SCENE_NAME: {SCENE_NAME}")
print(f"GROUP_PATH: {GROUP_PATH}")
print(f"ORIGINAL_PATH: {ORIGINAL_PATH}")
print(f"PRED_PATH: {PRED_PATH}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print("================================\n")

group_data = load_npy_dict(GROUP_PATH)
original_data = load_npy_dict(ORIGINAL_PATH)
print("========== group_data 检查 ==========")
print("type(group_data):", type(group_data))

if isinstance(group_data, dict):
    print("group_data keys:", group_data.keys())
    for key, value in group_data.items():
        arr = np.asarray(value)
        print(
            f"  {key}: type={type(value)}, "
            f"shape={arr.shape}, dtype={arr.dtype}"
        )

print("====================================")


coord = original_data["coord"]
gt = original_data["groud_turth"]

coord = np.asarray(coord, dtype=np.float32)
gt = np.asarray(gt).reshape(-1).astype(np.int64)

if os.path.exists(PRED_PATH):
    pred = np.load(PRED_PATH)
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
else:
    pred = None
    print(f"⚠️ 未找到预测文件，将跳过 prediction / accuracy: {PRED_PATH}")

required_group_keys = [
    "raw_to_super_index",
    "init_block_id",
    "init_seed_mask",
    "init_seed_rank",
]

for k in required_group_keys:
    if k not in group_data:
        raise KeyError(
            f"group.npy 中缺少字段 {k}。\n"
            f"请确认已经重新跑过 test，并且不是旧的 group.npy。\n"
            f"当前文件: {GROUP_PATH}"
        )

raw_to_super_index = np.asarray(group_data["raw_to_super_index"]).reshape(-1).astype(np.int64)
init_block_id = np.asarray(group_data["init_block_id"]).reshape(-1).astype(np.int64)
init_seed_mask = np.asarray(group_data["init_seed_mask"]).reshape(-1).astype(np.int64)
init_seed_rank = np.asarray(group_data["init_seed_rank"]).reshape(-1).astype(np.int64)

print("========== 数据检查 ==========")
print(f"coord shape              : {coord.shape}")
print(f"gt shape                 : {gt.shape}")
if pred is not None:
    print(f"pred shape               : {pred.shape}")
print(f"raw_to_super_index shape : {raw_to_super_index.shape}")
print(f"init_block_id shape      : {init_block_id.shape}")
print(f"init_seed_mask shape     : {init_seed_mask.shape}")
print(f"init_seed_rank shape     : {init_seed_rank.shape}")

print(f"有效最终超点数量         : {len(np.unique(raw_to_super_index[raw_to_super_index >= 0]))}")
print(f"有效初始化块数量         : {len(np.unique(init_block_id[init_block_id >= 0]))}")
print(f"seed 点数量              : {int((init_seed_mask > 0).sum())}")
print("================================\n")

assert coord.shape[0] == gt.shape[0], "coord 与 gt 点数不一致"
assert coord.shape[0] == raw_to_super_index.shape[0], "coord 与 raw_to_super_index 点数不一致"
assert coord.shape[0] == init_block_id.shape[0], "coord 与 init_block_id 点数不一致"
assert coord.shape[0] == init_seed_mask.shape[0], "coord 与 init_seed_mask 点数不一致"
assert coord.shape[0] == init_seed_rank.shape[0], "coord 与 init_seed_rank 点数不一致"

if pred is not None:
    assert coord.shape[0] == pred.shape[0], "coord 与 pred 点数不一致"


all_vis = []


# ============================================================
# 00 GT
# ============================================================

if SHOW_GT:
    vis_gt, pcd_gt = create_visualizer_fixed_labels(
        coord,
        gt,
        f"Ground Truth - {SCENE_NAME}",
        point_size=POINT_SIZE,
    )
    save_vis_and_pcd(vis_gt, pcd_gt, "00_ground_truth")
    all_vis.append(vis_gt)


# ============================================================
# 01 Prediction
# ============================================================

if SHOW_PRED and pred is not None:
    pred_show = pred.copy()
    pred_show[gt == ignore_index] = ignore_index

    vis_pred, pcd_pred = create_visualizer_fixed_labels(
        coord,
        pred_show,
        f"Prediction - {SCENE_NAME}",
        point_size=POINT_SIZE,
    )
    save_vis_and_pcd(vis_pred, pcd_pred, "01_prediction")
    all_vis.append(vis_pred)


# ============================================================
# 02 Accuracy
# ============================================================

if SHOW_ACC and pred is not None:
    acc_colors = color_prediction_accuracy(gt, pred)

    vis_acc, pcd_acc = create_visualizer(
        coord,
        acc_colors,
        f"Prediction Accuracy - {SCENE_NAME}",
        point_size=POINT_SIZE,
    )
    save_vis_and_pcd(vis_acc, pcd_acc, "02_prediction_accuracy")
    all_vis.append(vis_acc)


# ============================================================
# 03 Final Superpoint
# ============================================================

if SHOW_FINAL_SUPERPOINT:
    sp_colors, sp_compact, num_sp = color_final_superpoint(
        raw_to_super_index,
        SCENE_NAME,
    )

    vis_sp, pcd_sp = create_visualizer(
        coord,
        sp_colors,
        f"Final Superpoint - {SCENE_NAME} | N={num_sp}",
        point_size=SUPERPOINT_POINT_SIZE,
    )
    save_vis_and_pcd(vis_sp, pcd_sp, "03_final_superpoint")
    all_vis.append(vis_sp)


# ============================================================
# 04 Init Blocks
# ============================================================

if SHOW_INIT_BLOCK:
    block_colors, init_block_compact, num_blocks = color_init_block(
        init_block_id,
        SCENE_NAME,
    )

    vis_block, pcd_block = create_visualizer(
        coord,
        block_colors,
        f"Initial Hilbert Blocks - {SCENE_NAME} | N={num_blocks}",
        point_size=SUPERPOINT_POINT_SIZE,
    )
    save_vis_and_pcd(vis_block, pcd_block, "04_init_blocks")
    all_vis.append(vis_block)


# ============================================================
# 05 Init Blocks + Seeds Red
# ============================================================

if SHOW_INIT_BLOCK_SEED_RED:
    block_seed_colors, init_block_compact, num_blocks = color_init_block_seed_red(
        init_block_id,
        init_seed_mask,
        SCENE_NAME,
    )

    extra_geometries = None
    if ADD_SEED_SPHERES:
        extra_geometries = create_seed_spheres(
            coord,
            init_seed_mask > 0,
            color=(1.0, 0.0, 0.0),
            radius=SEED_SPHERE_RADIUS,
        )

    vis_seed_red, pcd_seed_red = create_visualizer(
        coord,
        block_seed_colors,
        f"Initial Blocks + Seeds Red - {SCENE_NAME}",
        point_size=SUPERPOINT_POINT_SIZE,
        extra_geometries=extra_geometries,
    )
    save_vis_and_pcd(vis_seed_red, pcd_seed_red, "05_init_blocks_seed_red")
    all_vis.append(vis_seed_red)


# ============================================================
# 06 Init Blocks + Seed Rank
# ============================================================

if SHOW_INIT_BLOCK_SEED_RANK:
    block_seed_rank_colors, init_block_compact, num_blocks = color_init_block_seed_rank(
        init_block_id,
        init_seed_rank,
        SCENE_NAME,
    )

    vis_seed_rank, pcd_seed_rank = create_visualizer(
        coord,
        block_seed_rank_colors,
        f"Initial Blocks + Seed Rank - {SCENE_NAME}",
        point_size=SUPERPOINT_POINT_SIZE,
    )
    save_vis_and_pcd(vis_seed_rank, pcd_seed_rank, "06_init_blocks_seed_rank")
    all_vis.append(vis_seed_rank)


# ============================================================
# 07 Seeds Only
# ============================================================

if SHOW_SEEDS_ONLY:
    seed_mask_bool = init_seed_mask > 0
    seed_coord = coord[seed_mask_bool]

    seed_rank_valid = init_seed_rank[seed_mask_bool]
    seed_palette = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.2, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.2, 1.0],
    ], dtype=np.float32)

    if seed_coord.shape[0] > 0:
        seed_colors = seed_palette[np.maximum(seed_rank_valid, 0) % len(seed_palette)]
    else:
        seed_colors = np.zeros((0, 3), dtype=np.float32)

    vis_seed_only, pcd_seed_only = create_visualizer(
        seed_coord,
        seed_colors,
        f"Seeds Only - {SCENE_NAME} | N={seed_coord.shape[0]}",
        point_size=SEED_POINT_SIZE,
    )
    save_vis_and_pcd(vis_seed_only, pcd_seed_only, "07_init_seeds_only")
    all_vis.append(vis_seed_only)


# ============================================================
# 图例
# ============================================================

try:
    legend_ids = list(range(num_classes))
    legend_img = build_legend_image(
        legend_ids,
        names,
        fixed_colors,
        cols=5,
        swatch=28,
    )

    if SHOW_GT:
        append_legend_below(os.path.join(OUTPUT_DIR, "00_ground_truth.png"), legend_img)

    if SHOW_PRED and pred is not None:
        append_legend_below(os.path.join(OUTPUT_DIR, "01_prediction.png"), legend_img)

except Exception as e:
    print(f"⚠️ 图例生成失败，但不影响点云可视化: {e}")


# ============================================================
# 窗口显示循环
# ============================================================

print(f"\n🎯 所有结果已保存至: {OUTPUT_DIR}")

if SHOW_WINDOWS and len(all_vis) > 0:
    print("👀 正在显示可视化窗口，关闭窗口或按 Q 退出...")

    while all(v.poll_events() for v in all_vis):
        for v in all_vis:
            v.update_renderer()

    for v in all_vis:
        v.destroy_window()

print("👋 可视化结束。")
