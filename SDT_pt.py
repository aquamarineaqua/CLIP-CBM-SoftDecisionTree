import torch
import torch.nn as nn
import torch.nn.functional as F


class SDT(nn.Module):
    """Implementation of soft decision tree in PyTorch (PyTorch 2.x ready).

    Parameters
    ----------
    input_dim : int
        The number of input dimensions.
    output_dim : int
        The number of output dimensions (e.g., number of classes).
    depth : int, default=5
        Tree depth. Larger -> exponentially more internal/leaves and cost.
    lamda : float, default=1e-3
        Coefficient of the regularization term.
    use_cuda : bool, default=False
        Whether to use CUDA device if available.
    inv_temp : float, default=1.0
        Inverse temperature for internal node sigmoid; larger -> harder routing.
    hard_leaf_inference : bool, default=False
        If True, at inference choose a single best leaf per sample (hard route)
        instead of mixture over all leaves (original behavior).
    use_penalty_ema : bool, default=False
        Whether to use an EMA-smoothed estimate of branching ratio alpha when
        computing the balance penalty (akin to TF get_penalty with moving avg).
        When enabled, gradients still flow through the current-batch estimate
        mixed with EMA so training remains stable but responsive.
    penalty_ema_beta : float, default=0.9
        Exponential moving-average factor for the alpha statistics. Larger ->
        smoother but slower to react. Effective alpha used is:
            alpha_used = (1-beta) * alpha_batch + beta * alpha_ema
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        depth=5,
        lamda=1e-3,
        use_cuda=False,
        inv_temp=1.0,
        hard_leaf_inference=False,
        use_penalty_ema=False,
        penalty_ema_beta=0.9,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.inv_temp = float(inv_temp)
        self.hard_leaf_inference = bool(hard_leaf_inference)
        self.use_penalty_ema = bool(use_penalty_ema)
        self.penalty_ema_beta = float(penalty_ema_beta)

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-d)) for d in range(0, self.depth)]

        # Internal nodes: linear logits; apply sigmoid with temperature in forward
        self.inner_nodes = nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False)

        # Leaf nodes: opinions over classes, combined by leaf weights
        self.leaf_nodes = nn.Linear(self.leaf_node_num_, self.output_dim, bias=False)

        # Alpha EMA buffer for balance penalty (one per internal node/parent)
        # Initialized to 0.5 (perfectly balanced). Stored as buffer so it is saved/loaded.
        self.register_buffer("alpha_ema", torch.full((self.internal_node_num_,), 0.5))

        # Note: By default, penalty is computed from current batch only. Optionally,
        # an EMA-smoothed alpha can be mixed in by setting use_penalty_ema=True.

    def forward(self, X, is_training_data=False):
        mu, penalty = self._forward(X, is_training_data=is_training_data)

        if not is_training_data and self.hard_leaf_inference:
            # Select a single leaf per sample (closest to TF OutputLayer behavior)
            idx = torch.argmax(mu, dim=1)
            mu_hard = F.one_hot(idx, num_classes=self.leaf_node_num_).to(mu.dtype)
            y_pred = self.leaf_nodes(mu_hard)
        else:
            y_pred = self.leaf_nodes(mu)

        if is_training_data:
            return y_pred, penalty
        else:
            return y_pred

    def _forward(self, X, is_training_data=False):
        """Implementation of data forwarding process."""
        batch_size = X.size(0)
        X = self._data_augment(X)

        # Routing probabilities at internal nodes
        logits_internal = self.inner_nodes(X)
        path_prob = torch.sigmoid(self.inv_temp * logits_internal)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        mu = torch.ones(batch_size, 1, 1, device=X.device, dtype=X.dtype)
        penalty = torch.tensor(0.0, device=X.device)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.depth):
            layer_path_prob = path_prob[:, begin_idx:end_idx, :]

            penalty = penalty + self._cal_penalty(
                layer_idx, begin_idx, mu, layer_path_prob, is_training_data=is_training_data
            )
            mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            mu = mu * layer_path_prob

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = mu.view(batch_size, self.leaf_node_num_)
        return mu, penalty

    def _cal_penalty(self, layer_idx, begin_idx, mu, layer_path_prob, is_training_data=False):
        """Compute balance regularization term for a layer of internal nodes.
        Parameters
        ----------
        layer_idx : int
            Current tree layer (0-based; 0 is root).
        begin_idx : int
            Start index in the flattened internal node indexing for this layer.
        mu : Tensor
            Shape (B, nodes_in_layer, 1). Probability of reaching each parent.
        layer_path_prob : Tensor
            Shape (B, nodes_in_layer, 2). For each parent, prob of going [left, right].
        is_training_data : bool
            If True, may update EMA statistics.

        Returns
        -------
        penalty : Tensor (scalar)
            The layer's balance penalty.
        """
        device = mu.device
        dtype = mu.dtype
        B = mu.size(0)
        parents = 2 ** layer_idx

        # Reshape helpers
        mu_parent = mu.view(B, parents)  # (B, parents)
        left_prob = layer_path_prob[:, :, 0].view(B, parents)  # (B, parents)

        # alpha_batch: expected left-branch probability per parent, weighted by reach prob
        eps = 1e-6
        num = torch.sum(mu_parent * left_prob, dim=0)  # (parents,)
        den = torch.sum(mu_parent, dim=0)              # (parents,)
        alpha_batch = num / torch.clamp(den, min=eps)  # (parents,)
        alpha_batch = torch.clamp(alpha_batch, min=eps, max=1 - eps)

        # Optionally mix with EMA stats (updated only on training batches)
        alpha_used = alpha_batch
        if self.use_penalty_ema:
            sl = slice(begin_idx, begin_idx + parents)
            alpha_old = self.alpha_ema[sl]
            if is_training_data:
                beta = self.penalty_ema_beta
                # Update EMA with detached batch stats to avoid creating a graph
                self.alpha_ema[sl] = beta * alpha_old + (1.0 - beta) * alpha_batch.detach()
                alpha_old = self.alpha_ema[sl]
            # Use a blended alpha to preserve some gradient flow via current batch
            beta = self.penalty_ema_beta
            alpha_used = (1.0 - beta) * alpha_batch + beta * alpha_old.detach()

        coeff = self.penalty_list[layer_idx]
        penalty_terms = torch.log(alpha_used) + torch.log(1.0 - alpha_used)  # (parents,)
        penalty = -coeff * torch.sum(penalty_terms).to(device=device, dtype=dtype)

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size(0)
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1, device=X.device, dtype=X.dtype)
        X = torch.cat((bias, X), dim=1)
        return X

    def _validate_parameters(self):
        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {} "
                   "instead.")
            raise ValueError(msg.format(self.depth))
        if not self.lamda >= 0:
            msg = ("The coefficient of the regularization term should not be "
                   "negative, but got {} instead.")
            raise ValueError(msg.format(self.lamda))
        if not (0.0 <= self.penalty_ema_beta <= 1.0):
            msg = ("penalty_ema_beta must be in [0,1], got {} instead.")
            raise ValueError(msg.format(self.penalty_ema_beta))

    def reset_penalty_ema(self, value: float = 0.5):
        """Reset the alpha EMA buffer to a constant value (default 0.5)."""
        with torch.no_grad():
            self.alpha_ema.fill_(float(value))
