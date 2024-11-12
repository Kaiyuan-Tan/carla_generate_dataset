# import numpy as np
# npy_file = "data/train/intersection_1_2_bbox.npy"
# sequence_labels = np.load(npy_file)

# unique_ts_us = np.unique(np.asarray(sequence_labels['t'], dtype='int64'))
# diff_us = np.diff(unique_ts_us)
# median_diff_us = np.median(diff_us)

# hz = int(np.rint(10 ** 6 / median_diff_us))
# print(hz)

