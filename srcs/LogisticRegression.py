import numpy as np
from typing import Optional, Union, Tuple, Dict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

def _sigmoid(z):
    out = np.empty_like(z, dtype=z.dtype)
    pos = (z >= 0)
    neg = ~pos
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    expz = np.exp(z[neg])
    out[neg] = expz / (1 + expz)
    return out

def _log_loss(p, y, sample_weight: Optional[np.ndarray] = None) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1-eps)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float64)
    loss = -(sample_weight * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean()
    return float(loss)

def _train_val_split(X: np.ndarray, y:np.ndarray, val_ratio: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_val = int(X.shape[0]*val_ratio)
    val = idx[:n_val]
    train = idx[n_val:]
    return X[train], X[val], y[train], y[val]


class LogisticRegressionScratch:
    def __init__(
            self,
            lr=.1,
            n_iters=1000,
            batch_size: Optional[int]=None,
            fit_intercept=True,
            reg_lambda=0.0,
            eps=1e-10,
            val_ratio=0.0,
            class_weight: Union[str, Dict[int, float], None] = None,
            early_stopping=True,
            patience=20,
            random_state: Optional[int] = 42
    ):
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.fit_intercept = fit_intercept
        self.eps = eps
        self.class_weight = class_weight
        self.val_ratio = val_ratio
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.if_fitted = False
        self.w: Optional[np.ndarray] = None

        self.history_train_loss = []
        self.history_val_loss = []
        

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1),  dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)
    
    def _init_para(self, n_feature):
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(loc=0.0, scale=0.1, size=(n_feature, ))

    def _compute_class_weight(self, y):
        if self.class_weight is None:
            return np.ones(len(y), dtype=np.float32)
        if self.class_weight != "balanced":
            raise ValueError("only support balanced")
        n = len(y)
        pos = y.sum()
        neg = n - pos
        w0 = n / (2 * neg) if neg > 0 else 1
        w1 = n / (2 * pos) if pos > 0 else 1
        return np.where(y==1, w1, w0).astype(np.float32)

    def fit(self, X, y, X_val: Optional[np.ndarray]=None, y_val: Optional[np.ndarray]=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        assert set(np.unique(y)).issubset({0.0, 1.0})

        X_aug = self._add_intercept(X) 
        n_samples, n_features = X_aug.shape
        self._init_para(n_features)
        
        
        if (self.val_ratio > 0) and (X_val is None or y_val is None):
            X_train, X_val_s, y_train, y_val_s = _train_val_split(X, y, val_ratio=self.val_ratio, seed=(self.random_state or 42))
            X_train = self._add_intercept(X_train)
            X_val_s = self._add_intercept(X_val_s)        
        else:
            X_train, y_train = X_aug, y
            X_val_s, y_val_s = self._add_intercept(X_val), y_val

        
        sw_tr = self._compute_class_weight(y_train)

        best_val = np.inf
        best_w = self.w.copy()
        wait = 0

        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_iters):
            if self.batch_size is None or self.batch_size > X_train.shape[0]:
                batches = [(X_train, y_train, sw_tr)]
            else:
                idx = rng.permutation(X_train.shape[0]) 
                batches = []    
                for start in range(0, X_train.shape[0], self.batch_size):
                    sl = idx[start : min(X_train.shape[0], start + self.batch_size)]
                    batches.append((X_train[sl], y_train[sl], sw_tr[sl]))

            for Xb, yb, swb in batches:
                z = Xb @ self.w
                p = _sigmoid(z)

                grad = Xb.T @ ((p - yb) * swb)  / Xb.shape[0]

                
                if self.reg_lambda > 0:
                    reg_vec = self.w.copy()
                    if self.fit_intercept:
                        reg_vec[0] = 0
                    grad += (self.reg_lambda / X_train.shape[0]) * reg_vec
                
                
                self.w -= self.lr * grad

            # Sau khi xong cả epoches cập nhập w hiện tại    
                
            z_train = X_train @ self.w
            p_train = _sigmoid(z_train)
            train_loss = _log_loss(p_train, y_train, sample_weight=sw_tr)
            if self.reg_lambda > 0:
                reg_now = self.w.copy()
                if reg_vec is not None:                                                      #reg_lambda > 0
                    reg_now[0] = 0    
                train_loss += (self.reg_lambda / (2 * X_train.shape[0])) * float(reg_now @ reg_now)
            self.history_train_loss.append(train_loss)


            if X_val_s is not None and y_val_s is not None:
                z_val = X_val_s @ self.w
                p_val = _sigmoid(z_val)
                val_loss = _log_loss(p_val, y_val_s)

                if self.reg_lambda > 0: 
                    val_loss += (self.reg_lambda / (2 * X_val_s.shape[0])) * float(reg_vec @ reg_vec)
                self.history_val_loss.append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val - 1e-10:
                        best_val = val_loss
                        best_w = self.w.copy()
                        wait = 0
                    else:
                        wait += 1
                        if wait > self.patience:
                            self.is_fitted = True
                            self.w = best_w
                            return self
        self.is_fitted = True 
        return self

    def decision_function(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_aug = self._add_intercept(X)
        return X_aug @ self.w

    def predict_proba(self, X):
        z = self.decision_function(X)
        return _sigmoid(z)
    
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(-1, 1)
        X = self._add_intercept(X) if self.fit_intercept else X
        p = _sigmoid(X @ self.w)
        return (p >= 0.5).astype(int)

    def _check_is_fitted(self):
        if not self.is_fitted or self.w is None:
            raise RuntimeError("Not fitted yet!")

                


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    mask = (y==1) | (y==7)
    X, y = X[mask], y[mask]
    y = (y == 1).astype(int)

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)
    #print(X_tr)
    #print(y_tr)
    #print(X_ts)
    #print(y_ts)

    clf = LogisticRegressionScratch(n_iters=3000, lr=0.005, reg_lambda=0.5, batch_size=32, val_ratio=0.2, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    print(clf.w)
    
    def visualize_loss(cl: LogisticRegressionScratch = None):
        plt.plot(cl.history_train_loss, label="train")
        plt.plot(cl.history_val_loss, label="val")
        plt.legend()
        plt.show()

    def metrics(y_true, y_pred):
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return acc, prec, rec, f1

    visualize_loss(clf)
    y_pred = clf.predict(X_ts)
    acc, prec, rec, f1 = metrics(y_ts, y_pred)
    print(f"Test: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")


    print("proba[:20] =", clf.predict_proba(X_ts[:20]))
