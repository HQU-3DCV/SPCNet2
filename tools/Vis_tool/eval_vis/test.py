import numpy as np
a=np.load(
"/media/cvmaster/P106/xlh/Data/dales/dales_noDrop_Normalized/test/5080_54400_new_1-1/coord.npy"
)
b=np.load(
"/media/cvmaster/P106/xlh/Data/dales/dales_noDrop_Normalized/test/5080_54400_new_1-2/coord.npy"
)
print(a.min(axis=0))
print(a.max(axis=0))
print(b.min(axis=0))
print(b.max(axis=0))


# path="./merged_scene/5080_54470_new.npy"
#
#
# data=np.load(
#     path,
#     allow_pickle=True
# ).item()
#
#
# print(data.keys())
#
#
# coord=data["coord"]
# gt=data["gt"]
# pred=data["pred"]
#
#
# print("coord:",coord.shape)
# print("gt:",gt.shape)
# print("pred:",pred.shape)
#
#
# print("GT classes:")
# print(np.unique(gt))
#
#
# print("Pred classes:")
# print(np.unique(pred))
#
#
# print(
#     "accuracy:",
#     np.mean(gt==pred)
# )