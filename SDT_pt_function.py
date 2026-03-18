"""
SDT_pt_function.py

SDT（Soft Decision Tree）相关功能函数模块。
包含模型加载、保存等实用工具函数。

Functions:
    - load_checkpoint_create: 从checkpoint加载SDT模型和优化器
"""

from typing import Optional, Tuple
import torch
from SDT_pt import SDT


def load_checkpoint_create(
    path: str,
    use_cuda: Optional[bool] = None
) -> Tuple["SDT", torch.optim.Adam, Optional[dict]]:
    """
    从checkpoint文件加载SDT模型和优化器。

    Args:
        path: checkpoint文件路径
        use_cuda: 是否使用CUDA，None时自动检测

    Returns:
        Tuple[SDT, Adam, Optional[dict]]: 
            - tree_new: 加载的SDT模型
            - optim_new: 加载的Adam优化器
            - extra: checkpoint中的额外信息（如果有）
    """
    use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    ckpt = torch.load(path, map_location=device)
    
    # 从checkpoint元数据重建模型
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
    
    # 使用非严格模式加载状态，兼容旧版checkpoint（可能包含EMA buffer键）
    missing, unexpected = tree_new.load_state_dict(ckpt['model_state'], strict=False)
    if missing or unexpected:
        print(f"load_state_dict warnings - missing: {list(missing)}, unexpected: {list(unexpected)}")
    
    # 重建优化器
    optim_new = torch.optim.Adam(tree_new.parameters(), lr=1e-3, weight_decay=5e-4)
    if 'optimizer_state' in ckpt:
        optim_new.load_state_dict(ckpt['optimizer_state'])
        # 将优化器状态张量移动到当前设备
        for state in optim_new.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    print(f"Loaded checkpoint from: {path}")
    return tree_new, optim_new, ckpt.get('extra')
