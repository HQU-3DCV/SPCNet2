# -*- coding: utf-8 -*-

import os
import colorsys
import numpy as np
import open3d as o3d

from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. 配置
# ============================================================

VIS_ROOT = "/media/cvmaster/P106/PCA-Vis1/vis"

scene_name = "scene0015_00"

POINT_SIZE = 5

SAVE_RESULT = True

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960


# ============================================================
# 2. 当前场景目录
# ============================================================

scene_dir = os.path.join(
    VIS_ROOT,
    scene_name,
)

if not os.path.exists(scene_dir):
    raise FileNotFoundError(
        f"找不到场景目录：{scene_dir}"
    )


coord_path = os.path.join(
    scene_dir,
    "coord.npy",
)

segment_path = os.path.join(
    scene_dir,
    "segment.npy",
)

# 第一次 Attention 之前
first_before_path = os.path.join(
    scene_dir,
    "feat_first_attn_before.npy",
)

# 第一次 Attention 之后
first_after_path = os.path.join(
    scene_dir,
    "feat_first_attn_after.npy",
)

# 最后一次 Attention 之后
last_after_path = os.path.join(
    scene_dir,
    "feat_last_attn_after.npy",
)

# 第一次迭代得到的最细层超点归属
first_iter_aff_path = os.path.join(
    scene_dir,
    "first_iter_affiliation.npy",
)

# 最终最细层超点归属
final_fine_aff_path = os.path.join(
    scene_dir,
    "final_fine_affiliation.npy",
)


# ============================================================
# 3. 文件检查
# ============================================================

required_paths = {
    "coord": coord_path,
    "segment": segment_path,
    "feat_first_attn_before": first_before_path,
    "feat_first_attn_after": first_after_path,
    "feat_last_attn_after": last_after_path,
    "first_iter_affiliation": first_iter_aff_path,
    "final_fine_affiliation": final_fine_aff_path,
}

for name, path in required_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少文件 {name}：{path}"
        )


# ============================================================
# 4. 加载数据
# ============================================================

coord = np.load(coord_path)
segment = np.load(segment_path)

feat_first_before = np.load(first_before_path)
feat_first_after = np.load(first_after_path)
feat_last_after = np.load(last_after_path)

first_iter_aff = np.load(first_iter_aff_path)
final_fine_aff = np.load(final_fine_aff_path)


print("========================================")
print(f"Scene                   : {scene_name}")
print(f"coord                   : {coord.shape}")
print(f"segment                 : {segment.shape}")
print(f"feat_first_before       : {feat_first_before.shape}")
print(f"feat_first_after        : {feat_first_after.shape}")
print(f"feat_last_after         : {feat_last_after.shape}")
print(f"first_iter_affiliation  : {first_iter_aff.shape}")
print(f"final_fine_affiliation  : {final_fine_aff.shape}")
print("========================================")


# ============================================================
# 5. 数据检查
# ============================================================

if coord.ndim != 2 or coord.shape[1] != 3:
    raise ValueError(
        f"coord 应为 [N,3]，实际为 {coord.shape}"
    )

feature_list = [
    feat_first_before,
    feat_first_after,
    feat_last_after,
]

feature_names = [
    "feat_first_before",
    "feat_first_after",
    "feat_last_after",
]

for name, feat in zip(feature_names, feature_list):
    if feat.ndim != 2:
        raise ValueError(
            f"{name} 应为 [N,C]，实际为 {feat.shape}"
        )

    if feat.shape[0] != coord.shape[0]:
        raise ValueError(
            f"{name} 与坐标点数不一致："
            f"{feat.shape[0]} vs {coord.shape[0]}"
        )

if not (
    feat_first_before.shape
    == feat_first_after.shape
    == feat_last_after.shape
):
    raise ValueError(
        "三组 PCA 特征尺寸不一致：\n"
        f"{feat_first_before.shape}\n"
        f"{feat_first_after.shape}\n"
        f"{feat_last_after.shape}"
    )

if first_iter_aff.reshape(-1).shape[0] != coord.shape[0]:
    raise ValueError(
        "first_iter_affiliation 与坐标点数不一致"
    )

if final_fine_aff.reshape(-1).shape[0] != coord.shape[0]:
    raise ValueError(
        "final_fine_affiliation 与坐标点数不一致"
    )


# ============================================================
# 6. 清理 NaN / Inf
# ============================================================

coord = np.asarray(
    coord,
    dtype=np.float64,
)

feat_first_before = np.asarray(
    feat_first_before,
    dtype=np.float64,
)

feat_first_after = np.asarray(
    feat_first_after,
    dtype=np.float64,
)

feat_last_after = np.asarray(
    feat_last_after,
    dtype=np.float64,
)

feat_first_before = np.nan_to_num(
    feat_first_before,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

feat_first_after = np.nan_to_num(
    feat_first_after,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

feat_last_after = np.nan_to_num(
    feat_last_after,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

first_iter_aff = first_iter_aff.reshape(-1).astype(np.int64)
final_fine_aff = final_fine_aff.reshape(-1).astype(np.int64)


# ============================================================
# 7. 三阶段联合 PCA
#
# 必须把三组特征拼起来共同拟合 PCA。
# 否则三个阶段的 RGB 不在同一个 PCA 坐标系中。
# ============================================================

num_points = coord.shape[0]

feat_all = np.concatenate(
    [
        feat_first_before,
        feat_first_after,
        feat_last_after,
    ],
    axis=0,
)

print("🚀 正在计算三阶段联合 PCA...")

pca = PCA(
    n_components=3,
)

feat_all_pca = pca.fit_transform(
    feat_all
)

first_before_pca = feat_all_pca[
    0:num_points
]

first_after_pca = feat_all_pca[
    num_points:2 * num_points
]

last_after_pca = feat_all_pca[
    2 * num_points:3 * num_points
]


print(
    "PCA explained variance ratio:",
    pca.explained_variance_ratio_,
)

print(
    "PCA total explained variance:",
    pca.explained_variance_ratio_.sum(),
)


# ============================================================
# 8. PCA -> RGB
#
# 三组特征共用同一套颜色范围。
# ============================================================

pca_min = np.percentile(
    feat_all_pca,
    1,
    axis=0,
)

pca_max = np.percentile(
    feat_all_pca,
    99,
    axis=0,
)


def pca_to_rgb(pca_feat):
    rgb = (
        pca_feat - pca_min
    ) / (
        pca_max - pca_min + 1e-8
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


first_before_rgb = pca_to_rgb(
    first_before_pca
)

first_after_rgb = pca_to_rgb(
    first_after_pca
)

last_after_rgb = pca_to_rgb(
    last_after_pca
)


# ============================================================
# 9. 超点归属编号 -> 稳定 RGB
#
# 同一个超点编号始终使用同一种颜色。
# 第一次迭代和最终结果分别生成颜色。
# ============================================================

def affiliation_to_rgb(affiliation):
    affiliation = affiliation.reshape(-1).astype(np.int64)

    unique_ids = np.unique(
        affiliation
    )

    color_table = {}

    golden_ratio = 0.618033988749895

    for rank, sp_id in enumerate(unique_ids):
        hue = (
            rank * golden_ratio
        ) % 1.0

        color_table[int(sp_id)] = colorsys.hsv_to_rgb(
            hue,
            0.70,
            0.95,
        )

    rgb = np.zeros(
        (affiliation.shape[0], 3),
        dtype=np.float64,
    )

    for sp_id, color in color_table.items():
        rgb[affiliation == sp_id] = color

    return rgb


first_iter_aff_rgb = affiliation_to_rgb(
    first_iter_aff
)

final_fine_aff_rgb = affiliation_to_rgb(
    final_fine_aff
)


# ============================================================
# 10. 创建 Open3D 点云
# ============================================================

def create_pcd(xyz, rgb):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        xyz
    )

    pcd.colors = o3d.utility.Vector3dVector(
        rgb
    )

    return pcd


pcd_first_before = create_pcd(
    coord,
    first_before_rgb,
)

pcd_first_after = create_pcd(
    coord,
    first_after_rgb,
)

pcd_last_after = create_pcd(
    coord,
    last_after_rgb,
)

pcd_first_iter_aff = create_pcd(
    coord,
    first_iter_aff_rgb,
)

pcd_final_fine_aff = create_pcd(
    coord,
    final_fine_aff_rgb,
)


# ============================================================
# 11. 相机路径
# ============================================================

camera_path = os.path.join(
    scene_dir,
    "shared_camera.json",
)


# ============================================================
# 12. 输出图片路径
# ============================================================

pca_first_before_png = os.path.join(
    scene_dir,
    "pca_first_attn_before.png",
)

pca_first_after_png = os.path.join(
    scene_dir,
    "pca_first_attn_after.png",
)

pca_last_after_png = os.path.join(
    scene_dir,
    "pca_last_attn_after.png",
)

first_iter_aff_png = os.path.join(
    scene_dir,
    "first_iter_fine_superpoints.png",
)

final_fine_aff_png = os.path.join(
    scene_dir,
    "final_fine_superpoints.png",
)


# ============================================================
# 13. 创建可交互 Open3D 窗口
#
# 使用 VisualizerWithKeyCallback，
# 这样可以：
#
# R：
#   把 Before First Attention 当前视角
#   同步到所有窗口
#
# S：
#   同步视角
#   + 保存 camera
#   + 保存所有 PNG
#
# Q：
#   正常关闭窗口
# ============================================================

def create_interactive_visualizer(
    pcd,
    title,
):
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name=title,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    vis.add_geometry(
        pcd
    )

    render_option = vis.get_render_option()

    render_option.point_size = POINT_SIZE

    render_option.background_color = np.array(
        [1.0, 1.0, 1.0]
    )

    return vis


# ============================================================
# 14. 创建五个窗口
# ============================================================

vis_first_before = create_interactive_visualizer(
    pcd_first_before,
    (
        "MASTER | Before First Attention "
        f"({scene_name})"
    ),
)

vis_first_after = create_interactive_visualizer(
    pcd_first_after,
    (
        "After First Attention "
        f"({scene_name})"
    ),
)

vis_last_after = create_interactive_visualizer(
    pcd_last_after,
    (
        "After Last Attention "
        f"({scene_name})"
    ),
)

vis_first_iter_aff = create_interactive_visualizer(
    pcd_first_iter_aff,
    (
        "First Iteration Fine Superpoints "
        f"({scene_name})"
    ),
)

vis_final_fine_aff = create_interactive_visualizer(
    pcd_final_fine_aff,
    (
        "Final Fine Superpoints "
        f"({scene_name})"
    ),
)


# ============================================================
# 15. 所有窗口列表
# ============================================================

all_vis = [
    vis_first_before,
    vis_first_after,
    vis_last_after,
    vis_first_iter_aff,
    vis_final_fine_aff,
]


other_vis = [
    vis_first_after,
    vis_last_after,
    vis_first_iter_aff,
    vis_final_fine_aff,
]


# ============================================================
# 16. 如果已经有 camera，
# 启动时让所有窗口先加载已有 camera
# ============================================================

if os.path.exists(
    camera_path
):
    print(
        f"📷 检测到已有 Camera：{camera_path}"
    )

    old_camera = (
        o3d.io.read_pinhole_camera_parameters(
            camera_path
        )
    )

    for vis in all_vis:

        vis.get_view_control().convert_from_pinhole_camera_parameters(
            old_camera,
            allow_arbitrary=True,
        )

        vis.poll_events()
        vis.update_renderer()


# ============================================================
# 17. 同步相机函数
#
# MASTER：
# Before First Attention
#
# 另外四个窗口会复制 MASTER 当前 camera。
# ============================================================

def sync_camera_from_master():

    master_camera = (
        vis_first_before
        .get_view_control()
        .convert_to_pinhole_camera_parameters()
    )

    for vis in other_vis:

        vis.get_view_control().convert_from_pinhole_camera_parameters(
            master_camera,
            allow_arbitrary=True,
        )

        vis.poll_events()
        vis.update_renderer()

    vis_first_before.poll_events()
    vis_first_before.update_renderer()

    return master_camera


# ============================================================
# 18. R 键：
# 只同步 Camera，不保存
# ============================================================

def callback_sync_camera(vis):

    del vis

    sync_camera_from_master()

    print(
        "\n🔄 已将 Before First Attention 的视角"
        "同步到全部窗口。"
    )

    return False


# ============================================================
# 19. S 键：
#
# 1. 同步 Camera
# 2. 保存 camera.json
# 3. 保存五张固定视角 PNG
# ============================================================

def callback_sync_and_save(vis):

    del vis

    camera_param = sync_camera_from_master()


    # --------------------------------------------------------
    # 保存 Camera
    # --------------------------------------------------------

    o3d.io.write_pinhole_camera_parameters(
        camera_path,
        camera_param,
    )

    print(
        f"\n✅ Camera 已保存：{camera_path}"
    )


    # --------------------------------------------------------
    # 再刷新所有窗口
    # --------------------------------------------------------

    for visualizer in all_vis:

        for _ in range(3):
            visualizer.poll_events()
            visualizer.update_renderer()


    # --------------------------------------------------------
    # 保存五张截图
    # --------------------------------------------------------

    save_items = [
        (
            vis_first_before,
            pca_first_before_png,
        ),
        (
            vis_first_after,
            pca_first_after_png,
        ),
        (
            vis_last_after,
            pca_last_after_png,
        ),
        (
            vis_first_iter_aff,
            first_iter_aff_png,
        ),
        (
            vis_final_fine_aff,
            final_fine_aff_png,
        ),
    ]


    for visualizer, output_path in save_items:

        visualizer.capture_screen_image(
            output_path,
            do_render=True,
        )

        print(
            f"✅ 保存：{output_path}"
        )


    print(
        "\n========================================"
    )

    print(
        "✅ 五个窗口已使用完全相同 Camera"
    )

    print(
        "你可以继续调整 MASTER 窗口，"
        "然后再次按 R 或 S。"
    )

    print(
        "========================================\n"
    )


    return False


# ============================================================
# 20. 注册快捷键
#
# 注意：
# 快捷键注册在 MASTER 窗口：
#
# Before First Attention
#
# 所以按 R / S 时，
# 请保证 MASTER 窗口处于激活状态。
# ============================================================

vis_first_before.register_key_callback(
    ord("R"),
    callback_sync_camera,
)

vis_first_before.register_key_callback(
    ord("S"),
    callback_sync_and_save,
)


# ============================================================
# 21. 使用说明
# ============================================================

print(
    "\n========================================"
)

print(
    "👀 已同时打开 5 个 Open3D 窗口"
)

print(
    ""
)

print(
    "主窗口："
)

print(
    "MASTER | Before First Attention"
)

print(
    ""
)

print(
    "操作流程："
)

print(
    "1. 先自由查看五个窗口"
)

print(
    "2. 在 MASTER 窗口中调整最终视角"
)

print(
    "3. MASTER 窗口按 R"
)

print(
    "   → 将当前 Camera 同步到其他窗口"
)

print(
    "4. 看看五张图的视角是否满意"
)

print(
    "5. MASTER 窗口按 S"
)

print(
    "   → 同步 Camera"
)

print(
    "   → 保存 Camera"
)

print(
    "   → 保存五张 PNG"
)

print(
    ""
)

print(
    "不满意可以继续调整，再次按 R / S。"
)

print(
    "全部完成后按 Q 或关闭所有窗口。"
)

print(
    "========================================\n"
)


# ============================================================
# 22. 五个窗口同时运行
#
# 不能使用 vis.run()，
# 因为 vis.run() 会阻塞后面窗口。
#
# 所以统一使用 poll_events()。
# ============================================================

while True:

    active_windows = []

    for visualizer in all_vis:

        try:
            active = visualizer.poll_events()

        except Exception:
            active = False

        active_windows.append(
            bool(active)
        )


    if not any(
        active_windows
    ):
        break


    for visualizer, active in zip(
        all_vis,
        active_windows,
    ):

        if active:

            visualizer.update_renderer()


# ============================================================
# 23. 关闭所有窗口
# ============================================================

for visualizer in all_vis:

    try:
        visualizer.destroy_window()

    except Exception:
        pass


print(
    "👋 所有 Open3D 窗口已关闭。"
)



# # ============================================================
# # 15. 渲染第一次迭代和最终最细层超点
# # ============================================================
#
# first_iter_aff_png = os.path.join(
#     scene_dir,
#     "first_iter_fine_superpoints.png",
# )
#
# final_fine_aff_png = os.path.join(
#     scene_dir,
#     "final_fine_superpoints.png",
# )
#
#
# render_with_fixed_camera(
#     pcd=pcd_first_iter_aff,
#     camera_param=camera_param,
#     output_path=first_iter_aff_png,
#     window_name="First Iteration Fine Superpoints",
# )
#
# render_with_fixed_camera(
#     pcd=pcd_final_fine_aff,
#     camera_param=camera_param,
#     output_path=final_fine_aff_png,
#     window_name="Final Fine Superpoints",
# )


# ============================================================
# 16. 保存 PLY
# ============================================================

if SAVE_RESULT:

    ply_items = [
        (
            "pca_first_attn_before.ply",
            pcd_first_before,
        ),
        (
            "pca_first_attn_after.ply",
            pcd_first_after,
        ),
        (
            "pca_last_attn_after.ply",
            pcd_last_after,
        ),
        (
            "first_iter_fine_superpoints.ply",
            pcd_first_iter_aff,
        ),
        (
            "final_fine_superpoints.ply",
            pcd_final_fine_aff,
        ),
    ]

    for filename, pcd in ply_items:
        output_path = os.path.join(
            scene_dir,
            filename,
        )

        o3d.io.write_point_cloud(
            output_path,
            pcd,
            write_ascii=False,
            compressed=True,
        )

        print(
            f"✅ 保存：{output_path}"
        )


# ============================================================
# 17. 图片拼接函数
# ============================================================

def load_font(font_size=30):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            font_size,
        )
    except Exception:
        return ImageFont.load_default()


def concatenate_images(
    image_paths,
    titles,
    output_path,
):
    images = [
        Image.open(path).convert("RGB")
        for path in image_paths
    ]

    base_size = images[0].size

    images = [
        image.resize(base_size)
        if image.size != base_size
        else image
        for image in images
    ]

    width, height = base_size

    title_height = 70

    canvas = Image.new(
        "RGB",
        (
            width * len(images),
            height + title_height,
        ),
        (
            255,
            255,
            255,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = load_font(
        30
    )

    for index, (image, title) in enumerate(
        zip(images, titles)
    ):
        x_start = index * width

        canvas.paste(
            image,
            (
                x_start,
                title_height,
            ),
        )

        bbox = draw.textbbox(
            (0, 0),
            title,
            font=font,
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        draw.text(
            (
                x_start
                + (width - text_width) // 2,
                20,
            ),
            title,
            fill=(0, 0, 0),
            font=font,
        )

        if index > 0:
            draw.line(
                [
                    (
                        x_start,
                        0,
                    ),
                    (
                        x_start,
                        height + title_height,
                    ),
                ],
                fill=(
                    180,
                    180,
                    180,
                ),
                width=2,
            )

    canvas.save(
        output_path
    )

    print(
        f"✅ 拼接图保存：{output_path}"
    )


# ============================================================
# 18. 拼接三阶段 PCA 图
# ============================================================

pca_compare_png = os.path.join(
    scene_dir,
    "pca_three_stage_compare.png",
)

concatenate_images(
    image_paths=[
        pca_first_before_png,
        pca_first_after_png,
        pca_last_after_png,
    ],
    titles=[
        "Before First Attention",
        "After First Attention",
        "After Last Attention",
    ],
    output_path=pca_compare_png,
)


# ============================================================
# 19. 拼接超点结果
# ============================================================

superpoint_compare_png = os.path.join(
    scene_dir,
    "fine_superpoint_compare.png",
)

concatenate_images(
    image_paths=[
        first_iter_aff_png,
        final_fine_aff_png,
    ],
    titles=[
        "First Iteration Fine Superpoints",
        "Final Fine Superpoints",
    ],
    output_path=superpoint_compare_png,
)


# ============================================================
# 20. 完成
# ============================================================

print("\n========================================")
print("✅ 三阶段 PCA 对比图：")
print(pca_compare_png)

print("\n✅ 最细层超点对比图：")
print(superpoint_compare_png)

print("\n✅ 共享相机参数：")
print(camera_path)

print("========================================")
