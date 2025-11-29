import hflayers as hf
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def fit_ridge_linear_map(
    X_train: torch.Tensor,  # corrupt_train_embeddings (N, d)
    Y_train: torch.Tensor,  # clean_train_embeddings   (N, d)
    lam: float = 1.0,  # ridge strength (on W only, not b)
):
    """
    Closed-form ridge solution for multi-output linear regression with bias.
    We solve for Θ = [W; b] in R^{(d+1) x d} using:
      Θ = (X̃^T X̃ + R)^(-1) X̃^T Y
    where X̃ = [X, 1], and R = diag([lam]*d + [0]) so bias is not regularized.
    """
    assert X_train.shape == Y_train.shape
    N, d = X_train.shape
    device, dtype = X_train.device, X_train.dtype

    # Augment inputs with a column of ones for bias
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    X_aug = torch.cat([X_train, ones], dim=1)  # (N, d+1)

    # Build regularizer (no penalty on bias row)
    R = torch.zeros(d + 1, d + 1, device=device, dtype=dtype)
    R[:d, :d] = lam * torch.eye(d, device=device, dtype=dtype)

    # Normal equations
    A = X_aug.T @ X_aug + R  # (d+1, d+1)
    B = X_aug.T @ Y_train  # (d+1, d)

    # Solve A Θ = B  (use Cholesky if A is PD; fall back to solve)
    try:
        L = torch.linalg.cholesky(A)
        Theta = torch.cholesky_solve(B, L)  # (d+1, d)
    except RuntimeError:
        Theta = torch.linalg.solve(A, B)

    W = Theta[:d, :].T  # (d, d) -> transpose to (d_out=d, d_in=d)
    b = Theta[d, :].view(1, -1)  # (1, d)
    return W, b


@torch.no_grad()
def linear_map_predict(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """Apply y_hat = X @ W^T + b."""
    return X @ W.T + b


@torch.no_grad()
def evaluate_reconstruction(Y_pred: torch.Tensor, Y_true: torch.Tensor):
    """Return MSE, mean L2, and mean cosine similarity."""
    mse = torch.mean((Y_pred - Y_true) ** 2).item()
    l2 = torch.mean(torch.linalg.norm(Y_pred - Y_true, dim=-1)).item()
    cos = (
        torch.nn.functional.cosine_similarity(Y_pred, Y_true, dim=-1, eps=1e-8)
        .mean()
        .item()
    )
    return {"mse": mse, "mean_l2": l2, "mean_cos": cos}


class ResidualMLP(nn.Module):
    """
    MLP with LayerNorm, GELU, and a residual skip from input to output.
    If in_dim != out_dim, the residual path uses a linear projection.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers in (2, 3), "Use 2 or 3 layers."

        self.ln_in = nn.LayerNorm(in_dim)
        layers = []

        # First block
        layers += [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        ]

        if num_layers == 3:
            # Middle block
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ]

        # Head
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        # Residual mapping
        self.use_proj = in_dim != out_dim
        self.proj = nn.Linear(in_dim, out_dim) if self.use_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = self.ln_in(x)
        y = self.net(x)
        return y + self.proj(x0)  # residual skip


class HopfieldLayerDecoder(nn.Module):
    """
    Hopfield layer behaves like a parametric associative memory. For each
    corrupted embedding, it looks up a mixture of internal patterns and returns
    a vector of the same dimension
    """

    def __init__(
        self,
        embeds_dim: int = 768,
        hidden_dim: int = 512,
        quantity: int = 128,
        beta: float = 1.0,
    ):
        super().__init__()
        # project into Hopfield space
        self.in_proj = nn.Sequential(
            nn.LayerNorm(embeds_dim),
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Hopfield layer lives inside `hidden_dim`
        self.hopfield = hf.HopfieldLayer(
            input_size=hidden_dim,  # state pattern R
            hidden_size=hidden_dim,  # associative space W_Q, W_K
            pattern_size=hidden_dim,  # pattern / output space W_V
            quantity=quantity,  # number of memory slots
            scaling=beta,
            stored_pattern_as_static=True,
            state_pattern_as_static=False,
            pattern_projection_as_static=True,
        )
        # project back to embedding space
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, embeds_dim),
            nn.LayerNorm(embeds_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h: (B, embeds_dim)
        h = self.in_proj(x)
        # h_seq: (B, 1, hidden_dim)
        # Hopfield layer expects a length-1 sequence
        h_seq = h.unsqueeze(1)
        # z: (B, 1, hidden_dim)
        z = self.hopfield(h_seq)
        # z: (B, hidden_dim)
        z = z.squeeze(1)
        # out: (B, embeds_dim)
        out = self.out_proj(z)
        out = out + x
        return out


class HopfieldAssociateDecoder(nn.Module):
    """
    Hopfield Associate Decoder retrieves from the clean embeddings, which are
    memory items from the input stored patterns; therefore it is non-parametric.

    In other words, here corrupted batch is used as queries, training clean
    embeddings as stored patterns (possibly projected), no internal memory bank
    needed.
    """

    def __init__(
        self,
        embeds_dim: int = 768,
        hidden_size: int = 512,
        beta: float = 1.0,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(embeds_dim),
            nn.Linear(embeds_dim, embeds_dim),
            nn.ReLU(inplace=True),
        )
        self.hopfield = hf.Hopfield(
            input_size=embeds_dim,  # R (state pattern dimension)
            hidden_size=hidden_size,  # associative (Q/K) space
            stored_pattern_size=embeds_dim,  # Y
            pattern_projection_size=embeds_dim,
            scaling=beta,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embeds_dim, embeds_dim),
            nn.LayerNorm(embeds_dim),
        )

    def forward(self, corrupted_batch, clean_memory):
        B, D = corrupted_batch.shape
        M, _ = clean_memory.shape

        R = self.in_proj(corrupted_batch)
        Y = clean_memory

        # Y: (1, M, D)
        Y_ = Y.unsqueeze(0)
        # R: (1, B, D)
        R_ = R.unsqueeze(0)
        P_ = Y_

        # Z: (1, B, D)
        # hopfield((stored_patterns, state_patterns, pattern_projections))
        # stored_patterns: keys
        # state_patterns: queries
        # pattern_projections: values
        Z_ = self.hopfield((Y_, R_, P_))
        Z = Z_.squeeze(0)

        out = self.out_proj(Z)
        return out
