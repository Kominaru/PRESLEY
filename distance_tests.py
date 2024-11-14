import numpy as np
import pandas as pd
import pickle as pkl

train_dev_df = pkl.load(open("data/barcelona/data_10+10/TRAIN_DEV_IMG", "rb"))
test_df = pkl.load(open("data/barcelona/data_10+10/TEST_IMG", "rb"))

train_dev_df = train_dev_df[train_dev_df["take"] == 1]
train_dev_df = train_dev_df.drop_duplicates(keep="first").reset_index(drop=True)

img_embeds = pkl.load(open("data/barcelona/data_10+10/IMG_VEC", "rb"))

user_centroids = np.stack(
    train_dev_df.groupby("id_user")
    .apply(lambda x: img_embeds[x["id_img"]].mean(axis=0))
    .values
)

p90 = train_dev_df.groupby("id_user").apply(
    lambda x: np.percentile(
        np.linalg.norm(img_embeds[x["id_img"]] - user_centroids[x.name], axis=1, ord=2),
        90,
    )
)

imgs_per_user = train_dev_df.groupby("id_user").size().values

test_df = test_df.groupby("id_test").filter(lambda x: len(x) > 10 and imgs_per_user[x["id_user"].values[0]] > 10)

test_pos = test_df[test_df["is_dev"] == 1].sort_values("id_test")

test_pos_dists = np.linalg.norm(
    img_embeds[test_pos["id_img"].values] - user_centroids[test_pos["id_user"].values],
    axis=1,
    ord=2,
)

test_neg_dists = (
    test_df[test_df["is_dev"] == 0]
    .groupby("id_test")
    .apply(
        lambda x: np.mean(
            np.linalg.norm(
                img_embeds[x["id_img"].values] - user_centroids[x["id_user"].values[0]],
                axis=1,
                ord=2,
            )
        )
    )
    .values
)

# Plot the difference between positive and negative distances
import matplotlib.pyplot as plt

diffs = test_neg_dists / test_pos_dists
diffs[diffs > 5] = 5

print("Percentage of negative distances greater than positive distances:", np.mean(diffs > 1))

plt.hist(diffs, bins=50, alpha=0.5, label="Positive")

plt.legend()

plt.show()
