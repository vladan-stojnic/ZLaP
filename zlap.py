from argparse import ArgumentParser

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix

from utils import accuracy, dfs_search, get_data, knn2laplacian, search_faiss, voc_mAP


def combine_separate_knns(
    knn_im2im,
    sim_im2im,
    knn_im2text,
    sim_im2text,
    num_classes,
):
    knn_im = knn_im2im + num_classes
    sim_im = sim_im2im

    knn = np.concatenate((knn_im, knn_im2text), axis=1)
    sim = np.concatenate((sim_im, sim_im2text), axis=1)

    return knn, sim


def create_separate_graph(features, clf, k):
    num_classes = clf.shape[0]
    assert k > 0
    k_im2im = min(k, features.shape[0])
    knn_im2im, sim_im2im = search_faiss(
        features, features, k=k_im2im
    )  # image2image search

    k_im2text = min(k, num_classes)
    knn_im2text, sim_im2text = search_faiss(
        clf, features, k=k_im2text
    )  # image2text search

    knn, sim = combine_separate_knns(
        knn_im2im,
        sim_im2im,
        knn_im2text,
        sim_im2text,
        num_classes,
    )

    knn_text = -1 * np.ones((num_classes, knn.shape[1]), dtype=knn.dtype)
    sim_text = np.zeros((num_classes, sim.shape[1]), dtype=sim.dtype)
    knn = np.concatenate((knn_text, knn), axis=0)
    sim = np.concatenate((sim_text, sim), axis=0)

    return knn, sim


def do_transductive_lp(features, clf, k, gamma, alpha, scale_sim=False):
    num_classes = clf.shape[0]
    knn, sim = create_separate_graph(features, clf, k)

    if scale_sim:
        xmin = np.min(sim[knn != -1])
        xmax = np.max(sim[knn != -1])
        sim = (sim - xmin) / (xmax - xmin)
    sim[sim < 0] = 0

    mask_knn = knn < num_classes
    sim[mask_knn] = sim[mask_knn] ** gamma
    L = knn2laplacian(knn, sim, alpha)

    scores = cp.zeros((features.shape[0], num_classes))
    for idx in range(num_classes):
        Y = cp.zeros((L.shape[0],))
        Y[idx] = 1
        out = dfs_search(L, Y, cast_to_numpy=False)
        scores[:, idx] = out[num_classes:]

    return scores.get()


def get_neighbors_for_inductive(
    unlabeled_features,
    clf,
    test_features,
    k,
    gamma,
    scale_sim=False,
    xmin=None,
    xmax=None,
):
    num_classes = clf.shape[0]
    k_im2im = min(k, unlabeled_features.shape[0])
    test_knn, test_sim = search_faiss(
        unlabeled_features, test_features, k=k_im2im
    )  # image2image search
    test_sim[test_sim < 0] = 0
    test_knn += num_classes
    if scale_sim:
        test_sim = (test_sim - xmin) / (xmax - xmin)

    k_im2text = min(k, num_classes)
    test_knn_im2text, test_sim_im2text = search_faiss(
        clf, test_features, k=k_im2text
    )  # image2text search
    test_sim_im2text[test_sim_im2text < 0] = 0
    if scale_sim:
        test_sim_im2text = (test_sim_im2text - xmin) / (xmax - xmin)
    test_sim_im2text = test_sim_im2text**gamma

    test_knn = np.concatenate((test_knn, test_knn_im2text), axis=1)
    test_sim = np.concatenate((test_sim, test_sim_im2text), axis=1)

    return test_knn, test_sim


def do_inductive_lp(
    unlabeled_features,
    clf,
    test_features,
    k,
    gamma,
    alpha,
    scale_sim=False,
):
    num_classes = clf.shape[0]
    knn, sim = create_separate_graph(unlabeled_features, clf, k)

    xmin = None
    xmax = None
    if scale_sim:
        xmin = np.min(sim[knn != -1])
        xmax = np.max(sim[knn != -1])
        sim = (sim - xmin) / (xmax - xmin)
    sim[sim < 0] = 0

    mask_knn = knn < num_classes
    sim[mask_knn] = sim[mask_knn] ** gamma
    L = knn2laplacian(knn, sim, alpha)

    test_knn, test_sim = get_neighbors_for_inductive(
        unlabeled_features,
        clf,
        test_features,
        k,
        gamma,
        scale_sim=scale_sim,
        xmin=xmin,
        xmax=xmax,
    )

    scores = cp.zeros((test_features.shape[0], num_classes))
    for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
        Y = cp.zeros((L.shape[0],))
        Y[k] = s
        out = dfs_search(L, Y, cast_to_numpy=False)
        scores[idx, :] = out[:num_classes]

    return scores.get()


def get_Linv(features, clf, k, gamma, alpha, scale_sim=False):
    num_classes = clf.shape[0]
    knn, sim = create_separate_graph(features, clf, k)

    xmin = None
    xmax = None
    if scale_sim:
        xmin = np.min(sim[knn != -1])
        xmax = np.max(sim[knn != -1])
        sim = (sim - xmin) / (xmax - xmin)
    sim[sim < 0] = 0

    mask_knn = knn < num_classes
    sim[mask_knn] = sim[mask_knn] ** gamma
    L = knn2laplacian(knn, sim, alpha)

    scores = cp.zeros((num_classes + features.shape[0], num_classes))
    for idx in range(num_classes):
        Y = cp.zeros((L.shape[0],))
        Y[idx] = 1
        out = dfs_search(L, Y, cast_to_numpy=False)
        scores[:, idx] = out.copy()

    return scores.get(), xmin, xmax


def do_sparse_inductive_lp(
    unlabeled_features,
    clf,
    test_features,
    k,
    gamma,
    alpha,
    scale_sim=False,
):
    num_classes = clf.shape[0]
    Linv, xmin, xmax = get_Linv(
        unlabeled_features, clf, k, gamma, alpha, scale_sim=scale_sim
    )

    test_knn, test_sim = get_neighbors_for_inductive(
        unlabeled_features,
        clf,
        test_features,
        k,
        gamma,
        scale_sim=scale_sim,
        xmin=xmin,
        xmax=xmax,
    )
    test_knn = cp.array(test_knn)
    test_sim = cp.array(test_sim)

    Linv_sparse = np.zeros_like(Linv)
    top = np.argmax(Linv, axis=1, keepdims=True)
    np.put_along_axis(Linv_sparse, top, np.take_along_axis(Linv, top, axis=1), axis=1)
    Linv_sparse = csr_matrix(cp.array(Linv_sparse))

    scores = cp.zeros((test_features.shape[0], num_classes))
    for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
        Z = (Linv_sparse[k, :]).copy()
        Z.data = Z.data * s.repeat(cp.diff(Z.indptr).get().tolist())
        scores[idx, :] = Z.sum(axis=0)

    return scores.get()


def get_args():
    args = ArgumentParser("ZLaP")
    args.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=[
            "imagenet",
            "dtd",
            "eurosat",
            "fgvca",
            "flowers",
            "food101",
            "pets",
            "sun397",
            "cars",
            "caltech101",
            "cifa10",
            "cifar100",
            "cub",
            "ucf101",
            "coco",
        ],
    )
    args.add_argument(
        "--backbone",
        type=str,
        default="RN50_openai",
        choices=[
            "RN50_openai",
            "ViT-B-16_openai",
            "ViT-B-16_laion2b_s34b_b88k",
            "ViT-H-14_laion2b_s32b_b79k",
            "ViT-L-14-336_openai",
            "ViT-L-14_openai",
            "albef",
            "blip",
            "eva-clip-8b",
            "eva-clip-18b",
        ],
    )
    args.add_argument("--k", type=int, default=5)
    args.add_argument("--gamma", type=float, default=5.0)
    args.add_argument("--alpha", type=float, default=0.3)
    args.add_argument(
        "--setup",
        type=str,
        default="transductive",
        choices=["transductive", "inductive", "sparse-inductive"],
    )
    args.add_argument(
        "--clf_type",
        type=str,
        default="text",
        choices=["text", "proxy", "cupl-text", "cupl-proxy"],
    )

    return args.parse_args()


def main():
    args = get_args()
    (
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        test_targets,
        clf_text,
        clf_image_train,
        clf_image_val,
        clf_image_test,
        clf_cupl_text,
        clf_cupl_image_train,
        clf_cupl_image_val,
        clf_cupl_image_test,
    ) = get_data(args.dataset, args.backbone)

    scale_sim = False
    if "proxy" in args.clf_type:
        scale_sim = True

    if args.clf_type == "text":
        clf_to_use = clf_text
    elif args.clf_type == "cupl-text":
        clf_to_use = clf_cupl_text

    if args.setup == "transductive":
        if args.clf_type == "proxy":
            clf_to_use = clf_image_test
        elif args.clf_type == "cupl-proxy":
            clf_to_use = clf_cupl_image_test

        scores = do_transductive_lp(
            test_features,
            clf_to_use,
            args.k,
            args.gamma,
            args.alpha,
            scale_sim=scale_sim,
        )
    elif args.setup == "inductive":
        if args.clf_type == "proxy":
            clf_to_use = clf_image_train
        elif args.clf_type == "cupl-proxy":
            clf_to_use = clf_cupl_image_train

        scores = do_inductive_lp(
            train_features,
            clf_to_use,
            test_features,
            args.k,
            args.gamma,
            args.alpha,
            scale_sim=scale_sim,
        )
    elif args.setup == "sparse-inductive":
        if args.clf_type == "proxy":
            clf_to_use = clf_image_train
        elif args.clf_type == "cupl-proxy":
            clf_to_use = clf_cupl_image_train

        scores = do_sparse_inductive_lp(
            train_features,
            clf_to_use,
            test_features,
            args.k,
            args.gamma,
            args.alpha,
            scale_sim=scale_sim,
        )

    if args.dataset == "coco":
        mAP = voc_mAP(np.concatenate((scores, test_targets), axis=1), 80)
        print(f"{args.dataset} mAP: {mAP:.2f}")
        return

    acc = accuracy(scores, test_targets)
    print(f"{args.dataset} accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
