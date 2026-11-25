"""Stage 2: the neural vetter (a small 1-D convolutional network, PyTorch).

Owners: Member 2 (network and training) and Member 3 (scalar branch and baselines).
Inputs per KOI:
  global view (1 x 201)                          -> convolutional branch A
  local, odd, even, secondary, centroid (5 x 81) -> convolutional branch B (stacked as channels,
                                                    so the network compares them bin by bin)
  scalar measurements (transit, stellar, diagnostics; never vetting outputs) -> small dense branch
Output: probability that the KOI is a real planet.
"""
import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


def _conv(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.ReLU(), nn.Conv1d(cout, cout, 5, padding=2),
                         nn.ReLU(), nn.MaxPool1d(3, 2))


class Vetter(nn.Module):
    def __init__(self, n_scalar, local_channels=5, use_global=True, use_scalars=True):
        super().__init__()
        self.use_global, self.use_scalars = use_global, use_scalars
        width = 0
        if use_global:
            self.g = nn.Sequential(_conv(1, 16), _conv(16, 32), _conv(32, 64), nn.AdaptiveMaxPool1d(4), nn.Flatten())
            width += 256
        self.l = nn.Sequential(_conv(local_channels, 16), _conv(16, 32), nn.AdaptiveMaxPool1d(8), nn.Flatten())
        width += 256
        if use_scalars:
            self.s = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            width += 64
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, g, l, s):
        parts = [self.l(l)]
        if self.use_global:
            parts.append(self.g(g))
        if self.use_scalars:
            parts.append(self.s(s))
        return self.head(torch.cat(parts, 1)).squeeze(1)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def train_predict(G, L, S, y, tr, va, te, seed, epochs=60, batch=256, patience=12, **kw):
    """Train on rows `tr`, keep the epoch with the lowest log loss on `va`, return probabilities for `te`.
    Runs on the Apple GPU (MPS) when present: ~9x faster than CPU for this model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Vetter(S.shape[1], local_channels=L.shape[1], **kw).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Gt, Lt, St = (torch.from_numpy(a).to(DEVICE) for a in (G, L, S))
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    idx = lambda i: torch.from_numpy(np.asarray(i)).to(DEVICE)
    va_i, te_i = idx(va), idx(te)
    best, best_state, since = np.inf, None, 0
    for ep in range(epochs):
        m.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), batch):
            i = idx(order[b:b + batch])
            g, l = Gt[i], Lt[i]
            if rng.random() < 0.5:  # transits are time-symmetric: mirror augmentation
                g, l = torch.flip(g, [2]), torch.flip(l, [2])
            opt.zero_grad()
            loss = lossf(m(g, l, St[i]), yt[i])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Gt[va_i], Lt[va_i], St[va_i]), yt[va_i]).item()
        if v < best - 1e-4:
            best, best_state, since = v, {k: x.detach().clone() for k, x in m.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Gt[te_i], Lt[te_i], St[te_i])).cpu().numpy()
