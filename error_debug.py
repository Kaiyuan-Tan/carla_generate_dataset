import numpy as np
# npy_file = "rvtdataset/test/intersection_1_1/event_representations_v2/stacked_histogram_dt=50_nbins=10/timestamps_us.npy"
npy_file = "train_2_bbox.npy"
sequence_labels = np.load(npy_file)

unique_ts_us = np.unique(np.asarray(sequence_labels['t'], dtype='int64'))
diff_us = np.diff(unique_ts_us)
median_diff_us = np.median(diff_us)

hz = int(np.rint(10 ** 6 / median_diff_us))
print(hz)
# bbox = np.load(npy_file)

print("FINISH")

# import h5py
# h5_file = ""
# h5f = h5py.File(h5_file, 'r')
# height = h5f['events']['height'][()].item()
# width = h5f['events']['width'][()].item()
