import h5py

data = h5py.File("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/mitbih_af_segments.h5", "r")
print(data.keys())