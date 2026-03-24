import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_fc(timeseries):

    n = timeseries.shape[1]

    fc = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            r, _ = pearsonr(timeseries[:, i], timeseries[:, j])
            fc[i, j] = r

    return fc


def load_fmri_timeseries(path):

    ts = np.load(path)

    return ts


def load_sc_matrix(path):

    sc = np.load(path)

    return sc


def load_anatomical_features(path):

    feat = np.load(path)

    return feat


def build_dataset(data_root):

    subjects = sorted(os.listdir(data_root))

    fc_list = []
    sc_list = []
    feat_list = []
    y_list = []

    for sub in subjects:

        sub_dir = os.path.join(data_root, sub)

        fmri_file = os.path.join(sub_dir, "fmri_ts.npy")
        sc_file = os.path.join(sub_dir, "sc.npy")
        anat_file = os.path.join(sub_dir, "anat.npy")
        label_file = os.path.join(sub_dir, "score.txt")

        if not os.path.exists(fmri_file):
            continue

        ts = load_fmri_timeseries(fmri_file)

        fc = compute_fc(ts)

        sc = load_sc_matrix(sc_file)

        feat = load_anatomical_features(anat_file)

        y = float(open(label_file).read().strip())

        fc_list.append(fc)
        sc_list.append(sc)
        feat_list.append(feat)
        y_list.append(y)

    fc_array = np.stack(fc_list)
    sc_array = np.stack(sc_list)
    feat_array = np.stack(feat_list)
    y_array = np.array(y_list)

    return fc_array, sc_array, feat_array, y_array


def main():

    data_root = "raw_hcpd"

    fc, sc, feat, y = build_dataset(data_root)

    os.makedirs("data", exist_ok=True)

    np.savez(
        "data/hcpd_data.npz",
        fc=fc,
        sc=sc,
        feat=feat,
        y=y
    )


if __name__ == "__main__":

    main()