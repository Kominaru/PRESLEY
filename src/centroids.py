# Compute the test set predictions based on the centroids of the training set

import numpy as np
import pandas as pd
import torch

from src.datamodule import ImageAuthorshipDataModule
from tqdm import tqdm


def get_centroid_preds(dm: ImageAuthorshipDataModule):
    def get_centroid(user_df: pd.DataFrame):
        return np.mean(image_embeddings[user_df["id_img"].values], axis=0)

    train_set = dm.train_dataset.dataframe
    train_set = train_set[train_set["take"] == 1].drop_duplicates(subset=["id_img"])

    test_img_ids = dm.test_dataset.dataframe["id_img"].values
    test_user_ids = dm.test_dataset.dataframe["id_user"].values

    image_embeddings = dm.image_embeddings.cpu().numpy()

    train_centroids = train_set.groupby("id_user").apply(get_centroid)

    test_preds = []

    for i in tqdm(range(test_img_ids.shape[0])):
        img_id = test_img_ids[i]
        user_id = test_user_ids[i]

        test_preds.append(1 / (1 + np.linalg.norm(image_embeddings[img_id] - train_centroids[user_id])))

    return np.array(test_preds)


def get_centroid_preds_latent(dm: ImageAuthorshipDataModule, model):
    def get_centroid(user_df: pd.DataFrame):
        return np.mean(image_embeddings[user_df["id_img"].values], axis=0)

    train_set = dm.train_dataset.dataframe
    train_set = train_set[train_set["take"] == 1].drop_duplicates(subset=["id_img"])

    test_img_ids = dm.test_dataset.dataframe["id_img"].values
    test_user_ids = dm.test_dataset.dataframe["id_user"].values

    image_embeddings = dm.image_embeddings.cpu().numpy()

    images_embedding_w = model.embedding_block.img_fc.weight.data.cpu().numpy()
    images_embedding_b = model.embedding_block.img_fc.bias.data.cpu().numpy()

    # In reasonable batches, compute the embeddings of the images (d=64) and replace the original ones (d=1536)
    # with the new ones
    batch_size = 1000
    n_batches = int(np.ceil(image_embeddings.shape[0] / batch_size))

    new_image_embeddings = np.zeros((image_embeddings.shape[0], images_embedding_w.shape[0]))
    for i in tqdm(range(n_batches)):
        batch = image_embeddings[i * batch_size : (i + 1) * batch_size]
        batch = np.dot(batch, images_embedding_w.T) + images_embedding_b
        new_image_embeddings[i * batch_size : (i + 1) * batch_size] = batch

    image_embeddings = new_image_embeddings

    train_centroids = train_set.groupby("id_user").apply(get_centroid)

    test_preds = []

    for i in tqdm(range(test_img_ids.shape[0])):
        img_id = test_img_ids[i]
        user_id = test_user_ids[i]

        test_preds.append(1 / (1 + np.linalg.norm(image_embeddings[img_id] - train_centroids[user_id])))

    return np.array(test_preds)


def get_isle_preds(dm, w, b):
    # Get the embeddings of the images
    image_embeddings = dm.image_embeddings.cpu().numpy()

    # Get the dataframe of the test set
    test_df = dm.test_dataset.dataframe
    train_df = dm.train_val_dataset.dataframe
    aux_df = train_df[train_df["take"] == 1].drop_duplicates(subset=["id_img"])

    # Compute the embeddings of each user
    user_embeddings = np.zeros((dm.nusers, w.shape[0]))
    for user_id, user_df in aux_df.groupby("id_user"):
        user_embeddings[user_id] = np.sum(image_embeddings[user_df["id_img"].values] @ w.T + b, axis=0)

    test_user_ids = test_df["id_user"].values
    test_img_ids = test_df["id_img"].values

    # Batchify the test dataframe and predict
    batch_size = 10000
    n_batches = int(np.ceil(test_df.shape[0] / batch_size))

    preds = []
    for i in tqdm(range(n_batches)):
        batch_user_embeddings = user_embeddings[test_user_ids[i * batch_size : (i + 1) * batch_size]]
        batch_img_embeddings = image_embeddings[test_img_ids[i * batch_size : (i + 1) * batch_size]] @ w.T + b

        preds.append(np.sum(batch_user_embeddings * batch_img_embeddings, axis=1))

    preds = np.concatenate(preds)

    print(preds.shape)

    return preds


def get_inception_sim(dm):
    # Get the embeddings of the images
    image_embeddings = dm.image_embeddings.cpu().numpy()

    # Get the dataframe of the test set
    test_df = dm.test_dataset.dataframe
    train_df = dm.train_val_dataset.dataframe
    aux_df = train_df[train_df["take"] == 1].drop_duplicates(subset=["id_img"])

    # Compute the embeddings of each user
    user_embeddings = np.zeros((dm.nusers, image_embeddings.shape[1]))
    for user_id, user_df in aux_df.groupby("id_user"):
        user_embeddings[user_id] = np.sum(image_embeddings[user_df["id_img"].values], axis=0)

    test_user_ids = test_df["id_user"].values
    test_img_ids = test_df["id_img"].values

    # Batchify the test dataframe and predict
    batch_size = 10000
    n_batches = int(np.ceil(test_df.shape[0] / batch_size))

    preds = []
    for i in tqdm(range(n_batches)):
        batch_user_embeddings = user_embeddings[test_user_ids[i * batch_size : (i + 1) * batch_size]]
        batch_img_embeddings = image_embeddings[test_img_ids[i * batch_size : (i + 1) * batch_size]]
        preds.append(np.sum(batch_user_embeddings * batch_img_embeddings, axis=1))

    preds = np.concatenate(preds)

    print(preds.shape)

    return preds
