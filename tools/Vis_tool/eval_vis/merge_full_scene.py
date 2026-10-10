# import os
# import numpy as np
#
# # ============================
# # 路径
# # ============================
#
# PRED_DIR = (
#     "/media/cvmaster/P106/"
#     "task8/dales-noaug-1-1-test/result"
# )
#
# RAW_DIR = (
#     "/media/cvmaster/P106/xlh/Data/dales/"
#     "dales_noDrop_Normalized/test"
# )
#
# SAVE_DIR = "./merged_scene"
#
# os.makedirs(
#     SAVE_DIR,
#     exist_ok=True
# )
#
#
# # ============================
# # fragment -> scene
# # ============================
#
# def get_scene_name(name):
#     # 5155_54335_new_1-1
#     # ->
#     # 5155_54335_new
#
#     return name.rsplit("_", 1)[0]
#
#
# # ============================
# # fragment排序
# # ============================
#
# def fragment_sort_key(name):
#     """
#     保证：
#     1-1
#     1-2
#     ...
#     1-10
#     2-1
#
#     而不是字符串排序
#     """
#
#     rc = name.rsplit("_", 1)[-1]
#
#     row, col = rc.split("-")
#
#     return int(row), int(col)
#
#
# # ============================
# # 恢复fragment空间位置
# # ============================
#
# def restore_global_coord(
#         coord,
#         fragment
# ):
#     """
#     DALES:
#
#     每个fragment:
#         x,y ∈ [-1,1]
#
#     10×10切块
#
#     fragment:
#         scene_row-col
#
#     row:
#         x方向
#
#     col:
#         y方向
#     """
#
#     row, col = fragment.rsplit(
#         "_",
#         1
#     )[1].split("-")
#
#     row = int(row)
#     col = int(col)
#
#     coord = coord.copy()
#
#     # 每个block长度=2
#
#     x_offset = (row - 1) * 2
#     y_offset = (col - 1) * 2
#
#     coord[:, 0] += x_offset
#
#     coord[:, 1] += y_offset
#
#     return coord
#
#
# # ============================
# # 收集预测
# # ============================
#
#
# scene_dict = {}
#
# for f in os.listdir(PRED_DIR):
#
#     if not f.endswith("_pred.npy"):
#         continue
#
#     fragment = f.replace(
#         "_pred.npy",
#         ""
#     )
#
#     scene = get_scene_name(
#         fragment
#     )
#
#     scene_dict.setdefault(
#         scene,
#         []
#     ).append(
#         fragment
#     )
#
# print(
#     "场景数量:",
#     len(scene_dict)
# )
#
# # ============================
# # merge
# # ============================
#
#
# for scene, fragments in scene_dict.items():
#
#     print("\n====================")
#
#     print(
#         "Processing:",
#         scene
#     )
#
#     print(
#         "fragment数量:",
#         len(fragments)
#     )
#
#     # 检查是否10×10
#
#     if len(fragments) != 100:
#         print(
#             "⚠️ 注意，该场景不是100块:"
#             ,
#             len(fragments)
#         )
#
#     coord_all = []
#
#     gt_all = []
#
#     pred_all = []
#
#     # 正确空间顺序
#
#     fragments = sorted(
#         fragments,
#         key=fragment_sort_key
#     )
#
#     for fragment in fragments:
#
#         print(
#             " ",
#             fragment
#         )
#
#         raw_path = os.path.join(
#             RAW_DIR,
#             fragment
#         )
#
#         coord_path = os.path.join(
#             raw_path,
#             "coord.npy"
#         )
#
#         gt_path = os.path.join(
#             raw_path,
#             "segment.npy"
#         )
#
#         pred_path = os.path.join(
#             PRED_DIR,
#             fragment + "_pred.npy"
#         )
#
#         if not os.path.exists(coord_path):
#             raise FileNotFoundError(
#                 coord_path
#             )
#
#         coord = np.load(
#             coord_path
#         )
#
#         gt = np.load(
#             gt_path
#         )
#
#         pred = np.load(
#             pred_path
#         )
#
#         print(
#             "before:",
#             coord.shape,
#             gt.shape,
#             pred.shape
#         )
#
#         # ==========================
#         # 检查预测对应
#         # ==========================
#
#         if (
#                 coord.shape[0]
#                 != pred.shape[0]
#         ):
#             raise RuntimeError(
#                 f"""
# {fragment}
#
# coord:
# {coord.shape}
#
# pred:
# {pred.shape}
#
# 预测没有恢复到原始点数量
# """
#             )
#
#         # ==========================
#         # 关键修改
#         # 恢复完整场景坐标
#         # ==========================
#
#         coord = restore_global_coord(
#             coord,
#             fragment
#         )
#
#         print(
#             "after:",
#             coord.min(axis=0),
#             coord.max(axis=0)
#         )
#
#         coord_all.append(
#             coord
#         )
#
#         gt_all.append(
#             gt
#         )
#
#         pred_all.append(
#             pred
#         )
#
#     # ==========================
#     # 拼接
#     # ==========================
#
#     merged = {
#
#         "coord":
#             np.concatenate(
#                 coord_all,
#                 axis=0
#             ),
#
#         "gt":
#             np.concatenate(
#                 gt_all,
#                 axis=0
#             ),
#
#         "pred":
#             np.concatenate(
#                 pred_all,
#                 axis=0
#             )
#
#     }
#
#     save_path = os.path.join(
#         SAVE_DIR,
#         scene + ".npy"
#     )
#
#     np.save(
#         save_path,
#         merged
#     )
#
#     print(
#         "\nSaved:",
#         save_path
#     )
#
#     print(
#         "Final points:",
#         merged["coord"].shape
#     )
#
#     print(
#         "coord range:"
#     )
#
#     print(
#         merged["coord"].min(axis=0)
#     )
#
#     print(
#         merged["coord"].max(axis=0)
#     )

import os
import numpy as np


# ============================================================
# 1. 路径配置
# ============================================================

PRED_DIR = (
    "/media/cvmaster/P106/"
    "task8/dales-noaug-1-1-test/result"
)

RAW_DIR = (
    "/media/cvmaster/P106/xlh/Data/dales/"
    "dales_noDrop_Normalized/test"
)

SAVE_DIR = "/media/cvmaster/P106/task8/merged_scene"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 2. 配置
# ============================================================

# DALES Ground 类别
GROUND_CLASS = 0

# 是否使用 Ground 对齐各 tile 的 Z
ALIGN_Z_BY_GROUND = True

# 10 × 10
GRID_SIZE = 10

# 每个归一化 block 的 XY 长度
# 当前 coord 的 x/y 范围约为 [-1, 1]
BLOCK_SIZE = 2.0


# ============================================================
# 3. fragment -> scene
# ============================================================

def get_scene_name(name):
    """
    例如：

    5155_54335_new_1-1
        ->
    5155_54335_new
    """

    return name.rsplit("_", 1)[0]


# ============================================================
# 4. 获取 row / col
# ============================================================

def get_row_col(fragment):
    """
    例如：

    5155_54335_new_3-7

    返回：

    row = 3
    col = 7
    """

    rc = fragment.rsplit(
        "_",
        1
    )[-1]

    row, col = rc.split("-")

    return int(row), int(col)


# ============================================================
# 5. fragment 排序
# ============================================================

def fragment_sort_key(name):
    """
    保证排序：

    1-1
    1-2
    ...
    1-10
    2-1
    ...
    10-10
    """

    return get_row_col(name)


# ============================================================
# 6. 恢复 XY 空间位置
# ============================================================

def restore_xy_coord(
    coord,
    fragment
):
    """
    当前每个小块：

    x ∈ [-1, 1]
    y ∈ [-1, 1]

    根据 row-col 恢复其在 10×10 大场景中的 XY 位置。
    """

    row, col = get_row_col(
        fragment
    )

    coord = coord.copy()

    # --------------------------------------------------------
    # 当前假设：
    #
    # row -> X
    # col -> Y
    #
    # 如果后面发现整体转置，只需要交换这两个 offset。
    # --------------------------------------------------------

    x_offset = (
        row - 1
    ) * BLOCK_SIZE

    y_offset = (
        col - 1
    ) * BLOCK_SIZE

    coord[:, 0] += x_offset
    coord[:, 1] += y_offset

    return coord


# ============================================================
# 7. 使用 Ground 对齐 Z
# ============================================================

def align_z_by_ground(
    coord,
    gt,
    fragment
):
    """
    使用 Ground 类点的 Z 中位数作为当前 fragment 的地面基准。

    将主要地面调整到 z = 0。

    注意：
    这是为了恢复大场景可视化连续性，
    并不等价于恢复真正的绝对高程。
    """

    coord = coord.copy()

    gt = np.asarray(
        gt
    ).reshape(-1)

    ground_mask = (
        gt == GROUND_CLASS
    )

    ground_count = int(
        ground_mask.sum()
    )

    if ground_count == 0:

        print(
            f"⚠️ {fragment} 没有 Ground 点，"
            "该块 Z 不进行调整。"
        )

        return coord, 0.0


    ground_z = coord[
        ground_mask,
        2
    ]


    # 中位数比平均值更不容易受到异常点影响
    ground_z_ref = float(
        np.median(
            ground_z
        )
    )


    coord[:, 2] -= ground_z_ref


    return coord, ground_z_ref


# ============================================================
# 8. 收集场景
# ============================================================

scene_dict = {}


for file_name in os.listdir(
    PRED_DIR
):

    if not file_name.endswith(
        "_pred.npy"
    ):
        continue


    fragment = file_name.replace(
        "_pred.npy",
        ""
    )


    scene = get_scene_name(
        fragment
    )


    scene_dict.setdefault(
        scene,
        []
    ).append(
        fragment
    )


print(
    "场景数量:",
    len(scene_dict)
)


# ============================================================
# 9. Merge
# ============================================================

for scene, fragments in scene_dict.items():

    print("\n")
    print("=" * 70)
    print(
        "Processing:",
        scene
    )

    print(
        "fragment数量:",
        len(fragments)
    )


    # --------------------------------------------------------
    # 检查 10×10
    # --------------------------------------------------------

    if len(fragments) != (
        GRID_SIZE * GRID_SIZE
    ):

        print(
            f"⚠️ {scene} fragment数量不是 "
            f"{GRID_SIZE * GRID_SIZE}，"
            f"实际={len(fragments)}"
        )


    # --------------------------------------------------------
    # 正确按照 row-col 排序
    # --------------------------------------------------------

    fragments = sorted(
        fragments,
        key=fragment_sort_key
    )


    coord_all = []
    gt_all = []
    pred_all = []


    # 保存每个 tile 的地面 Z 基准
    tile_ground_z = {}


    # ========================================================
    # 遍历 100 个小块
    # ========================================================

    for fragment in fragments:

        row, col = get_row_col(
            fragment
        )


        print(
            f"\n[{row:02d},{col:02d}] "
            f"{fragment}"
        )


        # ----------------------------------------------------
        # 文件路径
        # ----------------------------------------------------

        raw_path = os.path.join(
            RAW_DIR,
            fragment
        )


        coord_path = os.path.join(
            raw_path,
            "coord.npy"
        )


        gt_path = os.path.join(
            raw_path,
            "segment.npy"
        )


        pred_path = os.path.join(
            PRED_DIR,
            fragment + "_pred.npy"
        )


        # ----------------------------------------------------
        # 检查
        # ----------------------------------------------------

        if not os.path.exists(
            coord_path
        ):
            raise FileNotFoundError(
                coord_path
            )


        if not os.path.exists(
            gt_path
        ):
            raise FileNotFoundError(
                gt_path
            )


        if not os.path.exists(
            pred_path
        ):
            raise FileNotFoundError(
                pred_path
            )


        # ----------------------------------------------------
        # 加载
        # ----------------------------------------------------

        coord = np.load(
            coord_path
        )


        gt = np.load(
            gt_path
        ).reshape(-1)


        pred = np.load(
            pred_path
        ).reshape(-1)


        # ----------------------------------------------------
        # 点数检查
        # ----------------------------------------------------

        if (
            coord.shape[0]
            != gt.shape[0]
        ):

            raise RuntimeError(
                f"{fragment}\n"
                f"coord={coord.shape}\n"
                f"gt={gt.shape}"
            )


        if (
            coord.shape[0]
            != pred.shape[0]
        ):

            raise RuntimeError(
                f"{fragment}\n"
                f"coord={coord.shape}\n"
                f"pred={pred.shape}\n"
                "预测没有恢复到原始点数量。"
            )


        print(
            "points:",
            coord.shape[0]
        )


        print(
            "local xyz range:"
        )

        print(
            " min:",
            coord.min(axis=0)
        )

        print(
            " max:",
            coord.max(axis=0)
        )


        # ====================================================
        # 关键1：Ground 对齐 Z
        # ====================================================

        if ALIGN_Z_BY_GROUND:

            coord, ground_z_ref = (
                align_z_by_ground(
                    coord,
                    gt,
                    fragment
                )
            )

        else:

            ground_z_ref = 0.0


        tile_ground_z[
            (row, col)
        ] = ground_z_ref


        print(
            "ground z reference:",
            ground_z_ref
        )


        # ====================================================
        # 关键2：恢复 XY
        # ====================================================

        coord = restore_xy_coord(
            coord,
            fragment
        )


        print(
            "restored xyz range:"
        )

        print(
            " min:",
            coord.min(axis=0)
        )

        print(
            " max:",
            coord.max(axis=0)
        )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        coord_all.append(
            coord
        )

        gt_all.append(
            gt
        )

        pred_all.append(
            pred
        )


    # ========================================================
    # 10. 拼接完整场景
    # ========================================================

    merged_coord = np.concatenate(
        coord_all,
        axis=0
    )


    merged_gt = np.concatenate(
        gt_all,
        axis=0
    )


    merged_pred = np.concatenate(
        pred_all,
        axis=0
    )


    merged = {

        "coord":
            merged_coord,

        "gt":
            merged_gt,

        "pred":
            merged_pred,

    }


    # ========================================================
    # 11. 保存
    # ========================================================

    save_path = os.path.join(
        SAVE_DIR,
        scene + ".npy"
    )


    np.save(
        save_path,
        merged
    )


    # ========================================================
    # 12. 输出检查
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        "Saved:",
        save_path
    )


    print(
        "Final points:",
        merged_coord.shape
    )


    print(
        "Final coord min:",
        merged_coord.min(
            axis=0
        )
    )


    print(
        "Final coord max:",
        merged_coord.max(
            axis=0
        )
    )


    print(
        "GT classes:",
        np.unique(
            merged_gt
        )
    )


    print(
        "Pred classes:",
        np.unique(
            merged_pred
        )
    )


    # ========================================================
    # 13. 打印 10×10 各块 Z 基准
    # ========================================================

    if ALIGN_Z_BY_GROUND:

        print("\n")
        print(
            "各 fragment Ground Z 中位数："
        )


        for row in range(
            1,
            GRID_SIZE + 1
        ):

            line = []

            for col in range(
                1,
                GRID_SIZE + 1
            ):

                value = tile_ground_z.get(
                    (row, col),
                    np.nan
                )

                line.append(
                    f"{value:+.4f}"
                )


            print(
                f"row {row:02d}:",
                " ".join(line)
            )


    print("=" * 70)