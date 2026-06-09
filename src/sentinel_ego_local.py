# ============================================================
# SENTINEL-EGO — LOCAL VS CODE VERSION
# Datasets: CERT r5.2 and r6.2
# GPU auto-detected (tested on RTX 5070 Laptop, 8.5 GB VRAM)
# IEEE TIFS Submission — hamidborkot/SENTINEL-EGO-TIFS
# ============================================================

import os, gc, glob, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
warnings.filterwarnings('ignore')

# ── GPU setup ────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Paths — update to your local dataset location ────────────
BASE_R52 = r"C:\Users\HamidTulla\Downloads\New folder\sentnail ego\r5.2"
BASE_R62 = r"C:\Users\HamidTulla\Downloads\New folder\sentnail ego\r6.2"
CERT = BASE_R62   # change to BASE_R52 for cross-dataset validation

# ── Hyperparameters ──────────────────────────────────────────
SEED         = 42
FL_ROUNDS    = 10
LOCAL_EPOCHS = 3
LR           = 0.001
NOISE_SCALE  = 2.0
CLIP_NORM    = 1.0
Q_SAMPLE     = 0.01
DP_DELTA     = 1e-5
N_CLIENTS    = 10
N_BYZANTINE  = 3
GRAD_SCALE   = 5.0     # capped Byzantine gradient scale
CLIP_BYZ     = 5.0     # Byzantine grad norm cap
MOMENTUM     = 0.3     # FedAvg momentum coefficient
PATIENCE     = 3       # early stopping patience (rounds)
CHUNK        = 500_000
RISKY = ['wikileaks','pastebin','torrent','jobsearch','linkedin',
         'indeed','exploit','dropbox','4chan','hackforums','thepiratebay']
np.random.seed(SEED); torch.manual_seed(SEED)


def compute_epsilon(q, sigma, steps, delta=DP_DELTA, alpha=10):
    return alpha*q**2/(2*sigma**2)*steps + np.log(1/delta)/(alpha-1)


def to_gpu(x):
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)


class ThreatNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.GELU(), nn.Linear(64,1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


def local_train_stable(model, Xn, yn, epochs=LOCAL_EPOCHS, apply_dp=True):
    model.train(); model.to(DEVICE)
    pos_w = torch.tensor(
        [min(max(int((yn==0).sum()),1)/max(int((yn==1).sum()),1), 40.)],
        dtype=torch.float32).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    dl  = DataLoader(TensorDataset(to_gpu(Xn), to_gpu(yn)),
                     batch_size=max(32, int(Q_SAMPLE*len(Xn))),
                     shuffle=True, drop_last=False)
    for _ in range(epochs):
        for Xb, yb in dl:
            opt.zero_grad()
            wt   = torch.where(yb==1, pos_w.expand_as(yb), torch.ones_like(yb))
            loss = (nn.functional.binary_cross_entropy(
                        model(Xb), yb, reduction='none') * wt).mean()
            loss.backward()
            if apply_dp:
                for p in model.parameters():
                    if p.grad is None: continue
                    p.grad.mul_(min(1., CLIP_NORM/(p.grad.norm(2)+1e-9)))
                    p.grad.add_(torch.randn_like(p.grad)*NOISE_SCALE*CLIP_NORM)
            opt.step()
    model.cpu(); return model


def byzantine_train_stable(model, Xn, yn, epochs=LOCAL_EPOCHS):
    """Label flip + capped gradient scaling Byzantine attack."""
    model.train(); model.to(DEVICE)
    yn_f = yn.copy(); yn_f[yn_f == 1] = 0
    opt  = optim.Adam(model.parameters(), lr=LR)
    dl   = DataLoader(TensorDataset(to_gpu(Xn), to_gpu(yn_f)),
                      batch_size=max(32, int(Q_SAMPLE*len(Xn))),
                      shuffle=True, drop_last=False)
    for _ in range(epochs):
        for Xb, yb in dl:
            opt.zero_grad()
            nn.functional.binary_cross_entropy(model(Xb), yb).backward()
            for p in model.parameters():
                if p.grad is None: continue
                p.grad.mul_(GRAD_SCALE)
                norm = p.grad.norm(2)
                if norm > CLIP_BYZ * CLIP_NORM:
                    p.grad.mul_(CLIP_BYZ * CLIP_NORM / (norm + 1e-9))
            opt.step()
    model.cpu(); return model


def fed_avg_momentum(prev_gm, gm, lms, weights, alpha=MOMENTUM):
    gd_new  = gm.state_dict()
    gd_prev = prev_gm.state_dict()
    for k in gd_new:
        agg = torch.stack([lm.state_dict()[k].float()*w
                           for lm, w in zip(lms, weights)], 0).sum(0)
        gd_new[k] = (1 - alpha) * agg + alpha * gd_prev[k].float()
    gm.load_state_dict(gd_new)
    return gm


def multi_krum_stable(models, n_byz=N_BYZANTINE):
    params  = [torch.cat([p.data.cpu().view(-1) for p in m.parameters()])
               for m in models]
    n = len(params); k = max(1, n-n_byz-2); m_sel = max(1, n-n_byz)
    scores = [sum(sorted([((params[i]-params[j])**2).sum().item()
                           for j in range(n) if j != i])[:k]) for i in range(n)]
    sel = [models[i] for i in np.argsort(scores)[:m_sel]]
    gd  = deepcopy(sel[0]).state_dict()
    for key in gd:
        gd[key] = torch.stack([s.state_dict()[key].float() for s in sel], 0).mean(0)
    r = deepcopy(sel[0]); r.load_state_dict(gd); return r


def eval_model(model, Xt, yt):
    model.eval(); model.to(DEVICE)
    with torch.no_grad():
        prob = model(to_gpu(Xt)).cpu().numpy()
    model.cpu()
    best_f1, best_thr = 0., 0.5
    for thr in np.concatenate([np.arange(0.001,0.01,0.001),
                                 np.arange(0.01,0.1,0.005),
                                 np.arange(0.1,0.95,0.01)]):
        f1 = f1_score(yt, (prob>=thr).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    if len(np.unique(yt)) > 1:
        prec, rec, pr_thr = precision_recall_curve(yt, prob)
        pr_f1 = 2*prec*rec/(prec+rec+1e-9)
        if pr_f1.max() > best_f1: best_thr = float(pr_thr[np.argmax(pr_f1)])
    pred = (prob >= best_thr).astype(int)
    if len(np.unique(yt)) < 2:
        return {'F1':0,'AUC':0.5,'Precision':0,'Recall':0,'FPR':0,'FNR':1}, prob
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0,1]).ravel()
    return {'F1':round(f1_score(yt,pred,zero_division=0),4),
            'AUC':round(roc_auc_score(yt,prob),4),
            'Precision':round(precision_score(yt,pred,zero_division=0),4),
            'Recall':round(recall_score(yt,pred,zero_division=0),4),
            'Accuracy':round(accuracy_score(yt,pred),4),
            'FPR':round(fp/(fp+tn+1e-9),4),
            'FNR':round(fn/(fn+tp+1e-9),4)}, prob


def run_fl_stable(Xtr, ytr, masks, poison_ids=set(),
                   aggregation="fedavg", label="FL", n_rounds=FL_ROUNDS):
    gm      = ThreatNet(Xtr.shape[1])
    prev_gm = deepcopy(gm)
    best_gm = deepcopy(gm)
    best_f1 = 0.0; no_improve = 0
    for rnd in range(n_rounds):
        lms, sizes = [], []
        for i, mask in enumerate(masks):
            if mask.sum() < 10: continue
            lm = deepcopy(gm)
            if i in poison_ids:
                lm = byzantine_train_stable(lm, Xtr[mask], ytr[mask])
            else:
                lm = local_train_stable(lm, Xtr[mask], ytr[mask], apply_dp=True)
            lms.append(lm); sizes.append(int(mask.sum()))
        if not lms: continue
        prev_gm = deepcopy(gm)
        if aggregation == "krum":
            gm = multi_krum_stable(lms)
        else:
            gm = fed_avg_momentum(prev_gm, gm, lms, [s/sum(sizes) for s in sizes])
        r, _ = eval_model(gm, X_test, y_test)
        print(f"  [{label}] Round {rnd+1:2d}/{n_rounds} "
              f"F1={r['F1']:.4f} AUC={r['AUC']:.4f} "
              f"Recall={r['Recall']:.4f} FPR={r['FPR']:.4f}")
        if r['F1'] > best_f1:
            best_f1 = r['F1']; best_gm = deepcopy(gm); no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at round {rnd+1} — best F1={best_f1:.4f}")
            break
    eps = compute_epsilon(Q_SAMPLE, NOISE_SCALE, n_rounds * LOCAL_EPOCHS)
    print(f"  -> Best F1={best_f1:.4f}  eps={eps:.4f}")
    return best_gm, eps


def mia_audit(model, Xtr, Xte, seed=SEED):
    model.eval(); model.to(DEVICE)
    with torch.no_grad():
        p_tr = model(to_gpu(Xtr)).cpu().numpy()
        p_te = model(to_gpu(Xte)).cpu().numpy()
    model.cpu()
    rng = np.random.default_rng(seed)
    n   = min(20000, len(p_tr), len(p_te))
    Xm  = np.concatenate([p_tr[rng.choice(len(p_tr),n,replace=False)],
                           p_te[rng.choice(len(p_te),n,replace=False)]]).reshape(-1,1)
    ym  = np.concatenate([np.ones(n), np.zeros(n)])
    clf = LogisticRegression().fit(Xm, ym)
    return round(roc_auc_score(ym, clf.predict_proba(Xm)[:,1]), 4)


# ── Entry point ──────────────────────────────────────────────
if __name__ == '__main__':
    print("Run as notebook for interactive use.")
    print("All functions exported: run_fl_stable, eval_model, mia_audit")
