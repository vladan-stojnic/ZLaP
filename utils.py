import cupy as cp
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from cupyx.scipy.sparse import csr_matrix, diags, eye
from cupyx.scipy.sparse import linalg as s_linalg


def search_faiss(X, Q, k):
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    res.setTempMemory(0)
    s, knn = faiss.knn_gpu(res, Q, X, k, metric=faiss.METRIC_INNER_PRODUCT)

    return knn, s


def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal(), 0)
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = cp.array(1.0 / cp.sqrt(S))
    D[cp.isnan(D)] = 0
    D[cp.isinf(D)] = 0
    D_mh = diags(D.reshape(-1), 0)
    Wn = D_mh * W * D_mh
    return Wn


def knn2laplacian(knn, s, alpha=0.99):
    N = knn.shape[0]
    k = knn.shape[1]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    knn_flat = knn.flatten("F")
    row_idx_rep_flat = row_idx_rep.flatten("F")
    sim_flat = s.flatten("F")
    valid_knn = np.where(knn_flat != -1)[0]
    knn_flat = cp.array(knn_flat[valid_knn])
    row_idx_rep_flat = cp.array(row_idx_rep_flat[valid_knn])
    sim_flat = cp.array(sim_flat[valid_knn])
    W = csr_matrix(
        (sim_flat, (row_idx_rep_flat, knn_flat)),
        shape=(N, N),
    )
    W = W + W.T
    Wn = normalize_connection_graph(W)
    L = eye(Wn.shape[0]) - alpha * Wn
    return L


def dfs_search(L, Y, tol=1e-6, maxiter=50, cast_to_numpy=True):
    out = s_linalg.cg(L, Y, tol=tol, maxiter=maxiter)[0]
    if cast_to_numpy:
        return cp.asnumpy(out)
    else:
        return out


def normalize(x):
    return F.normalize(torch.tensor(x), p=2, dim=1).cpu().numpy()


def accuracy(scores, labels):
    preds = np.argmax(scores, axis=1)
    acc = np.mean(100.0 * (preds == labels))
    return acc


def get_data(dataset, model="RN50"):
    try:
        train_features = np.load(f"features/{dataset}/{model}_train_feats.npy")
        train_features = normalize(train_features.astype(np.float32))
        train_targets = np.load(f"features/{dataset}/{model}_train_targets.npy")
    except OSError:
        print("No train features! Inductive setting will not be possible!")
        train_features = None
        train_targets = None

    try:
        val_features = np.load(f"features/{dataset}/{model}_val_feats.npy")
        val_features = normalize(val_features.astype(np.float32))
        val_targets = np.load(f"features/{dataset}/{model}_val_targets.npy")
    except OSError:
        print("No val features!!!")
        val_features = None
        val_targets = None

    try:
        test_features = np.load(f"features/{dataset}/{model}_test_feats.npy")
        test_features = normalize(test_features.astype(np.float32))
        test_targets = np.load(f"features/{dataset}/{model}_test_targets.npy")
    except OSError:
        print("No test features! Using val features as test!")
        if val_features is None:
            raise ValueError("No val features either!")

        test_features = val_features
        test_targets = val_targets
        val_features = None
        val_targets = None

    try:
        clf_text = np.load(
            f"features/{dataset}/classifiers/{model}_text_classifier.npy"
        )
        clf_text = normalize(clf_text.T)
    except OSError:
        raise ValueError("No extracted text classifier!")

    try:
        clf_cupl_text = np.load(
            f"features/{dataset}/classifiers/{model}_cupl_text_classifier.npy"
        )
        clf_cupl_text = normalize(clf_cupl_text.T)
    except OSError:
        clf_cupl_text = None

    try:
        clf_image_train = np.load(
            f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_train.npy"
        )
        clf_image_train = normalize(clf_image_train.T)
    except OSError:
        clf_image_train = None

    try:
        clf_cupl_image_train = np.load(
            f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_train.npy"
        )
        clf_cupl_image_train = normalize(clf_cupl_image_train.T)
    except OSError:
        clf_cupl_image_train = None

    try:
        clf_image_val = np.load(
            f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_val.npy"
        )
        clf_image_val = normalize(clf_image_val.T)
    except OSError:
        clf_image_val = None

    try:
        clf_cupl_image_val = np.load(
            f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_val.npy"
        )
        clf_cupl_image_val = normalize(clf_cupl_image_val.T)
    except OSError:
        clf_cupl_image_val = None

    try:
        clf_image_test = np.load(
            f"features/{dataset}/classifiers/{model}_inmap_proxy_classifier_test.npy"
        )
        clf_image_test = normalize(clf_image_test.T)
    except OSError:
        print(
            "No InMaP classifer learned on test, so using the one trained on val instead!"
        )
        clf_image_test = clf_image_val

    try:
        clf_cupl_image_test = np.load(
            f"features/{dataset}/classifiers/{model}_cupl_inmap_proxy_classifier_test.npy"
        )
        clf_cupl_image_test = normalize(clf_cupl_image_test.T)
    except OSError:
        print(
            "No InMaP classifer learned on test, so using the one trained on val instead!"
        )
        clf_cupl_image_test = clf_cupl_image_val

    return (
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
    )


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False):
    seg = imagessetfilelist
    gt_label = seg[:, num:].astype(np.int32)

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = sorted_label[i] > 0
            fp[i] = sorted_label[i] <= 0
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP
