"""
SDT_pt_function.py

Utility functions for the Soft Decision Tree (SDT).
Includes helper functions for model loading, saving, and related operations.

Functions:
    - load_checkpoint_create: Load an SDT model and optimizer from a checkpoint
"""

from typing import Optional, Tuple
import torch
from SDT_pt import SDT


def load_checkpoint_create(
    path: str,
    use_cuda: Optional[bool] = None
) -> Tuple["SDT", torch.optim.Adam, Optional[dict]]:
    """
    Load an SDT model and optimizer from a checkpoint file.

    Args:
        path: path to the checkpoint file
        use_cuda: whether to use CUDA; auto-detected when None

    Returns:
        Tuple[SDT, Adam, Optional[dict]]:
            - tree_new: loaded SDT model
            - optim_new: loaded Adam optimizer
            - extra: additional information stored in the checkpoint (if any)
    """
    use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    ckpt = torch.load(path, map_location=device)
    
    # Reconstruct model from checkpoint metadata
    meta = ckpt.get('meta', {})
    tree_new = SDT(
        input_dim=meta.get('input_dim', 28 * 28),
        output_dim=meta.get('output_dim', 10),
        depth=int(meta.get('depth', 5)),
        lamda=float(meta.get('lamda', 1e-3)),
        use_cuda=use_cuda,
        inv_temp=float(meta.get('inv_temp', 1.0)),
        hard_leaf_inference=bool(meta.get('hard_leaf_inference', False))
    ).to(device)
    
    # Load state in non-strict mode to maintain compatibility with older checkpoints (which may contain EMA buffer keys)
    missing, unexpected = tree_new.load_state_dict(ckpt['model_state'], strict=False)
    if missing or unexpected:
        print(f"load_state_dict warnings - missing: {list(missing)}, unexpected: {list(unexpected)}")
    
    # Reconstruct optimizer
    optim_new = torch.optim.Adam(tree_new.parameters(), lr=1e-3, weight_decay=5e-4)
    if 'optimizer_state' in ckpt:
        optim_new.load_state_dict(ckpt['optimizer_state'])
        # Move optimizer state tensors to the current device
        for state in optim_new.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    print(f"Loaded checkpoint from: {path}")
    return tree_new, optim_new, ckpt.get('extra')
