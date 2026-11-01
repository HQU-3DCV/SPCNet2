import os
import numpy as np
import torch


def make_dirs(path):
    os.makedirs(path, exist_ok=True)


class EvalVisSaver:
    """
    用于在 evaluate 过程中，按 epoch 保存指定场景的可视化数据：
      - 基础点级信息（coord / gt / pred / oracle_pred）
      - 超点映射 per_level_mappings
      - 迭代历史 vis_iter_mappings / affiliations_history
      - backbone 特征（用于 PCA）
      - 超点特征（可选）
    """

    def __init__(
        self,
        root_path,
        target_scene_names,
        save_sp_feat=True,
        save_oracle_pred=True,
        save_metadata=True,
    ):
        """
        Args:
            root_path: 总保存根目录，例如:
                /xxx/exp/sonata1/SPGNet-ARConv-v1m1/eval_vis_compare
            target_scene_names: 只保存这些场景名，list / tuple / set 都行
            save_sp_feat: 是否保存超点特征
            save_oracle_pred: 是否保存 oracle 预测
            save_metadata: 是否额外保存 meta 信息
        """
        self.root_path = root_path
        # self.target_scene_names = set(target_scene_names)

        self.target_scene_names = (
            None if target_scene_names is None else set(target_scene_names)
        )

        self.save_sp_feat = save_sp_feat
        self.save_oracle_pred = save_oracle_pred
        self.save_metadata = save_metadata

        make_dirs(self.root_path)

    # def need_save(self, data_name):
    #     return data_name in self.target_scene_names
    def need_save(self, data_name):
        return (
                self.target_scene_names is None
                or data_name in self.target_scene_names
        )

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            return [self._to_numpy(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self._to_numpy(v) for v in x)
        if isinstance(x, dict):
            return {k: self._to_numpy(v) for k, v in x.items()}
        return x

    def _get_scene_dir(self, epoch_id, data_name):
        scene_dir = os.path.join(
            self.root_path,
            f"epoch_{epoch_id:03d}",
            data_name
        )
        make_dirs(scene_dir)
        return scene_dir

    def _ids_to_rgb(self, ids_np):
        """
        将分组 id 映射成稳定伪随机颜色。
        用于 initial_affiliation / per_level_mappings 这种分组可视化。
        """
        ids_np = ids_np.astype(np.int64)

        colors = np.stack(
            [
                (ids_np * 53 + 37) % 256,
                (ids_np * 97 + 17) % 256,
                (ids_np * 193 + 29) % 256,
            ],
            axis=1,
        ).astype(np.uint8)

        return colors

    def _save_xyzrgb_ply(self, save_path, xyz, rgb):
        """
        保存 xyzrgb 格式的 ascii ply 文件。
        可直接用 CloudCompare / MeshLab / Open3D 打开。
        """
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.uint8)

        assert xyz.ndim == 2 and xyz.shape[1] >= 3, \
            f"xyz shape 应为 [N, 3] 或 [N, >=3]，但得到 {xyz.shape}"
        assert rgb.ndim == 2 and rgb.shape[1] == 3, \
            f"rgb shape 应为 [N, 3]，但得到 {rgb.shape}"
        assert xyz.shape[0] == rgb.shape[0], \
            f"xyz 点数 {xyz.shape[0]} 与 rgb 点数 {rgb.shape[0]} 不一致"

        xyz = xyz[:, :3]

        with open(save_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p, c in zip(xyz, rgb):
                f.write(
                    f"{p[0]} {p[1]} {p[2]} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )

    def save_initial_partition_and_seeds(
            self,
            scene_dir,
            data_name,
            coord,
            initial_affiliation,
            init_seed_mask=None,
            init_seed_rank=None,
    ):
        """
        保存初始化分段与初始化种子点可视化。

        输出文件：
            1. init_partition.ply
               初始 Hilbert 分段，每个初始 block 一个颜色。

            2. init_partition_seed_highlight.ply
               在初始分段基础上，将种子点高亮为红色。

            3. init_seed_only.ply
               只保存种子点。

            4. init_seed_rank.ply
               如果 init_seed_rank 存在，则不同 rank 的种子用不同颜色。
        """
        coord = np.asarray(coord, dtype=np.float32)
        initial_affiliation = np.asarray(initial_affiliation).reshape(-1)

        if coord.shape[0] != initial_affiliation.shape[0]:
            print(
                f"[EvalVisSaver Warning] {data_name}: "
                f"coord 点数 {coord.shape[0]} 与 initial_affiliation 点数 "
                f"{initial_affiliation.shape[0]} 不一致，跳过初始化可视化保存。"
            )
            return

        # --------------------------------------------------
        # 1. 初始分段可视化
        # --------------------------------------------------
        base_rgb = self._ids_to_rgb(initial_affiliation)

        self._save_xyzrgb_ply(
            os.path.join(scene_dir, "init_partition.ply"),
            coord,
            base_rgb,
        )

        # --------------------------------------------------
        # 2. 初始分段 + 种子点高亮
        # --------------------------------------------------
        if init_seed_mask is not None:
            init_seed_mask = np.asarray(init_seed_mask).reshape(-1).astype(bool)

            if init_seed_mask.shape[0] != coord.shape[0]:
                print(
                    f"[EvalVisSaver Warning] {data_name}: "
                    f"coord 点数 {coord.shape[0]} 与 init_seed_mask 点数 "
                    f"{init_seed_mask.shape[0]} 不一致，跳过 seed 高亮保存。"
                )
                return

            rgb_seed_highlight = base_rgb.copy()

            # 红色高亮所有种子点
            rgb_seed_highlight[init_seed_mask] = np.array(
                [255, 0, 0],
                dtype=np.uint8,
            )

            self._save_xyzrgb_ply(
                os.path.join(scene_dir, "init_partition_seed_highlight.ply"),
                coord,
                rgb_seed_highlight,
            )

            # --------------------------------------------------
            # 3. 单独保存种子点
            # --------------------------------------------------
            if init_seed_mask.sum() > 0:
                seed_coord = coord[init_seed_mask]
                seed_rgb = np.tile(
                    np.array([[255, 0, 0]], dtype=np.uint8),
                    (seed_coord.shape[0], 1),
                )

                self._save_xyzrgb_ply(
                    os.path.join(scene_dir, "init_seed_only.ply"),
                    seed_coord,
                    seed_rgb,
                )

        # --------------------------------------------------
        # 4. 如果有 seed rank，则保存不同 rank 颜色版本
        # --------------------------------------------------
        if init_seed_rank is not None:
            init_seed_rank = np.asarray(init_seed_rank).reshape(-1)

            if init_seed_rank.shape[0] != coord.shape[0]:
                print(
                    f"[EvalVisSaver Warning] {data_name}: "
                    f"coord 点数 {coord.shape[0]} 与 init_seed_rank 点数 "
                    f"{init_seed_rank.shape[0]} 不一致，跳过 seed rank 保存。"
                )
                return

            rank_rgb = base_rgb.copy()

            seed_rank_mask = init_seed_rank >= 0

            # 给不同 rank 的种子点不同颜色
            # rank=0 红色，rank=1 黄色，rank=2 蓝色，rank=3 紫色，其他绿色
            rank_colors = np.array(
                [
                    [255, 0, 0],  # rank 0
                    [255, 255, 0],  # rank 1
                    [0, 128, 255],  # rank 2
                    [255, 0, 255],  # rank 3
                    [0, 255, 0],  # rank >= 4
                ],
                dtype=np.uint8,
            )

            valid_ranks = init_seed_rank[seed_rank_mask]
            clipped_ranks = np.clip(valid_ranks, 0, len(rank_colors) - 1)

            rank_rgb[seed_rank_mask] = rank_colors[clipped_ranks]

            self._save_xyzrgb_ply(
                os.path.join(scene_dir, "init_partition_seed_rank.ply"),
                coord,
                rank_rgb,
            )

    def save(
        self,
        epoch_id,
        data_name,
        coord,
        segment,
        pred,
        per_level_mappings,
        backbone_feat,
        SPfeat_Pointfeat,
        vis_iter_mappings=None,
        oracle_pred=None,
        sp_feat=None,
        initial_affiliation=None,

        # ==============================
        # 新增：初始化种子点可视化
        # ==============================
        init_seed_mask=None,
        init_seed_rank=None,

        serialize_rank_raw=None,
        serialize_order_raw=None,
        serialize_coord=None,
        extra_data=None,
    ):
        """
        保存一个场景在某个 epoch 的可视化结果
        """
        if not self.need_save(data_name):
            return False

        scene_dir = self._get_scene_dir(epoch_id, data_name)

        coord_npy = self._to_numpy(coord)
        segment_npy = self._to_numpy(segment)
        pred_npy = self._to_numpy(pred)
        per_level_mappings_npy = self._to_numpy(per_level_mappings)
        backbone_feat_npy = self._to_numpy(backbone_feat)
        vis_iter_mappings_npy = self._to_numpy(vis_iter_mappings)
        oracle_pred_npy = self._to_numpy(oracle_pred)
        sp_feat_npy = self._to_numpy(sp_feat)
        initial_affiliation_npy = self._to_numpy(initial_affiliation)
        SPfeat_Pointfeat_npy = self._to_numpy(SPfeat_Pointfeat)
        init_seed_mask_npy = self._to_numpy(init_seed_mask)
        init_seed_rank_npy = self._to_numpy(init_seed_rank)

        #序列化
        serialize_rank_raw_npy = self._to_numpy(serialize_rank_raw)
        serialize_order_raw_npy = self._to_numpy(serialize_order_raw)
        serialize_coord_npy = self._to_numpy(serialize_coord)

        # 1) 基础结果
        basic_data = {
            "coord": coord_npy,
            "segment": segment_npy,
            "pred": pred_npy,
        }
        if self.save_oracle_pred and oracle_pred_npy is not None:
            basic_data["oracle_pred"] = oracle_pred_npy
        if initial_affiliation_npy is not None:
            basic_data["initial_affiliation"] = initial_affiliation_npy
        if init_seed_mask_npy is not None:
            basic_data["init_seed_mask"] = init_seed_mask_npy
        if init_seed_rank_npy is not None:
            basic_data["init_seed_rank"] = init_seed_rank_npy

        #序列化顺序的可视化
        if serialize_rank_raw_npy is not None:
            basic_data["serialize_rank_raw"] = serialize_rank_raw_npy
        if serialize_order_raw_npy is not None:
            basic_data["serialize_order_raw"] = serialize_order_raw_npy
        if serialize_coord_npy is not None:
            basic_data["serialize_coord"] = serialize_coord_npy
        ####################################
        if extra_data is not None:
            basic_data["extra_data"] = self._to_numpy(extra_data)

        np.save(
            os.path.join(scene_dir, "basic.npy"),
            basic_data,
            allow_pickle=True
        )
        # ==============================
        # 新增：保存初始化分段与种子点 PLY 可视化
        # ==============================
        # if initial_affiliation_npy is not None:
        #     self.save_initial_partition_and_seeds(
        #         scene_dir=scene_dir,
        #         data_name=data_name,
        #         coord=coord_npy,
        #         initial_affiliation=initial_affiliation_npy,
        #         init_seed_mask=init_seed_mask_npy,
        #         init_seed_rank=init_seed_rank_npy,
        #     )
        # ==============================

        # 2) 超点映射
        group_data = {
            "per_level_mappings": per_level_mappings_npy,
        }
        np.save(
            os.path.join(scene_dir, "group.npy"),
            group_data,
            allow_pickle=True
        )

        # 3) 迭代历史
        if vis_iter_mappings_npy is not None:
            np.save(
                os.path.join(scene_dir, "iter.npy"),
                vis_iter_mappings_npy,
                allow_pickle=True
            )

        # 4) backbone 特征（PCA 用）
        feat_data = {
            "feat": backbone_feat_npy,
            # "coord": coord_npy,
            # "segment": segment_npy,
            # "pred": pred_npy,
            # "per_level_mappings": per_level_mappings_npy,
        }

        # if self.save_oracle_pred and oracle_pred_npy is not None:
        #     feat_data["oracle_pred"] = oracle_pred_npy
        # if initial_affiliation_npy is not None:
        #     feat_data["initial_affiliation"] = initial_affiliation_npy
        np.save(
            os.path.join(scene_dir, "backbone_feat.npy"),
            feat_data,
            allow_pickle=True
        )
        if SPfeat_Pointfeat_npy is not None:
            sp_point_data = {
                "SPfeat_Pointfeat": SPfeat_Pointfeat_npy
            }
            np.save(
                os.path.join(scene_dir, "sp_point_feat.npy"),
                sp_point_data,
                allow_pickle=True
            )
        # 5) 超点特征（可选）
        if self.save_sp_feat and sp_feat_npy is not None:
            sp_feat_data = {
                "sp_feat": sp_feat_npy,
                "per_level_mappings": per_level_mappings_npy,
            }
            np.save(
                os.path.join(scene_dir, "sp_feat.npy"),
                sp_feat_data,
                allow_pickle=True
            )

        # 6) meta 信息（可选）
        if self.save_metadata:
            meta_data = {
                "epoch": epoch_id,
                "scene_name": data_name,
            }
            np.save(
                os.path.join(scene_dir, "meta.npy"),
                meta_data,
                allow_pickle=True
            )

        return True
