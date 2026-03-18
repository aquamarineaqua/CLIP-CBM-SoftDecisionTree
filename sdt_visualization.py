# -*- coding: utf-8 -*-
"""
SDT (Soft Decision Tree) 可视化工具函数

提供以下功能：
1. extract_sdt_parameters: 提取SDT树的参数信息
2. visualize_sdt: 可视化整棵SDT及样本决策路径
3. visualize_internal_node_weight: 可视化内部节点权重向量
4. get_leaf_distribution: 统计叶子节点的样本分布
5. compute_internal_node_counts: 计算内部节点的样本数量
6. compute_node_logits_for_dataset: 计算节点在数据集上的logits
7. visualize_top_k_images_for_node: 可视化节点Top-K激活图像
8. visualize_top_and_bottom_k_images_for_node: 可视化节点高/低激活对比图像
9. analyze_all_nodes_summary: 生成所有节点的激活分析摘要
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


# ===================== 颜色配置 =====================

ESSENTIAL_COLORS = {
    'internal_node': '#1f77b4',  # blue
    'leaf_node': '#ff7f0e',      # orange
    'edge': '#888888',           # default edge
    'best_edge': '#2ca02c',      # green
}


# ===================== 叶子节点与内部节点统计 =====================

def get_leaf_distribution(tree, dataloader, device):
    """
    统计通过SDT决策路径到达各叶子节点的样本分布。
    
    Args:
        tree: 训练好的SDT模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        leaf_counts: 每个叶子节点的样本数量 (numpy array)
        leaf_class_counts: 每个叶子节点中各类别的样本数量 (dict: leaf_idx -> numpy array)
        leaf_predictions: 每个叶子节点预测的类别 (numpy array)
    """
    tree.eval()
    L = tree.leaf_node_num_
    num_classes = tree.output_dim
    
    # 初始化计数器
    leaf_counts = np.zeros(L, dtype=np.int64)
    leaf_class_counts = {l: np.zeros(num_classes, dtype=np.int64) for l in range(L)}
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            # 获取每个样本的叶子节点概率分布 mu
            X = data
            batch_size = X.size(0)
            
            # Data augmentation (prepend constant 1)
            bias = torch.ones(batch_size, 1, device=device, dtype=X.dtype)
            X_aug = torch.cat((bias, X), dim=1)
            
            # 计算路由概率
            logits_internal = tree.inner_nodes(X_aug)
            path_prob = torch.sigmoid(tree.inv_temp * logits_internal)
            path_prob = torch.unsqueeze(path_prob, dim=2)
            path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
            
            # 计算到达各叶子的概率 mu
            mu = torch.ones(batch_size, 1, 1, device=device, dtype=X.dtype)
            begin_idx = 0
            end_idx = 1
            for layer_idx in range(tree.depth):
                layer_path_prob = path_prob[:, begin_idx:end_idx, :]
                mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)
                mu = mu * layer_path_prob
                begin_idx = end_idx
                end_idx = begin_idx + 2 ** (layer_idx + 1)
            
            mu = mu.view(batch_size, L)
            
            # Hard routing: 选择概率最大的叶子节点
            best_leaf = torch.argmax(mu, dim=1).cpu().numpy()
            target_np = target.cpu().numpy()
            
            # 统计
            for i in range(batch_size):
                leaf_idx = best_leaf[i]
                class_idx = target_np[i]
                leaf_counts[leaf_idx] += 1
                leaf_class_counts[leaf_idx][class_idx] += 1
    
    # 获取每个叶子节点预测的类别
    leaf_predictions = np.zeros(L, dtype=np.int64)
    with torch.no_grad():
        W_leaf = tree.leaf_nodes.weight.detach().cpu()  # [C, L]
        for l in range(L):
            leaf_predictions[l] = torch.argmax(W_leaf[:, l]).item()
    
    return leaf_counts, leaf_class_counts, leaf_predictions


def compute_internal_node_counts(leaf_counts, depth):
    """
    通过叶子节点的样本数量，计算每个内部节点的样本数量。
    
    Args:
        leaf_counts: 每个叶子节点的样本数量 (numpy array)
        depth: 树的深度
    
    Returns:
        internal_counts: 每个内部节点的样本数量 (numpy array)
    """
    L = len(leaf_counts)
    internal_node_num = L - 1
    internal_counts = np.zeros(internal_node_num, dtype=np.int64)
    
    # 从叶子节点向上计算
    for l in range(L):
        count = leaf_counts[l]
        node_idx = l + internal_node_num  # 叶子节点在完全二叉树中的索引
        while node_idx > 0:
            parent_idx = (node_idx - 1) // 2
            internal_counts[parent_idx] += count
            node_idx = parent_idx
    
    return internal_counts


# ===================== 参数提取 =====================

def extract_sdt_parameters(tree):
    """
    Extract parameter information of the whole tree for external access:
    - internal_nodes: list[ {index, layer, W (Tensor), b (float)} ]
    - leaves: list[ {index, class_logits (Tensor), class_probs (Tensor)} ]
    
    Note: W corresponds to the weight vector excluding the constant 1 used in 
    data augmentation; b comes from column 0 of inner_nodes.weight.
    
    Args:
        tree: 训练好的SDT模型
    
    Returns:
        dict: 包含 'depth', 'internal_nodes', 'leaves' 的字典
    """
    tree.eval()
    with torch.no_grad():
        D = tree.depth
        Ni = tree.internal_node_num_
        L = tree.leaf_node_num_

        # internal nodes: inner_nodes.weight shape (Ni, input_dim+1)
        W_full = tree.inner_nodes.weight.detach().cpu()  # [Ni, in_dim+1]
        internals = []
        for i in range(Ni):
            wrow = W_full[i]
            b = float(wrow[0].item())
            W = wrow[1:].clone()  # remove the constant term weight
            layer = int(math.floor(math.log2(i + 1)))
            internals.append({
                'index': i,
                'layer': layer,
                'W': W,
                'b': b,
            })

        # leaves: leaf_nodes.weight shape (C, L) -> for leaf l, column weight[:, l] is class logits
        W_leaf = tree.leaf_nodes.weight.detach().cpu()  # [C, L]
        leaves = []
        for l in range(L):
            logits = W_leaf[:, l]
            probs = torch.softmax(logits, dim=0)
            leaves.append({
                'index': l,
                'class_logits': logits,
                'class_probs': probs,
            })

    return {
        'depth': D,
        'internal_nodes': internals,
        'leaves': leaves,
    }


# ===================== 辅助函数 =====================

def _binary_tree_positions(depth, x_span=(0.0, 1.0), y_step=1.0):
    """
    Generate node coordinates for a full binary tree.
    
    Args:
        depth: 树的深度
        x_span: x轴范围 (min, max)
        y_step: 每层y坐标的步进值
    
    Returns:
        pos_internal: dict[i] -> (x, y) for i in [0, Ni)
        pos_leaf: dict[l] -> (x, y) for l in [0, L)
    
    Nodes are arranged level-wise, y decreases downward.
    """
    Ni = 2 ** depth - 1
    L = 2 ** depth
    pos_internal = {}
    pos_leaf = {}

    def layer_y(layer):
        return -layer * y_step

    # compute internal node positions (even spacing within each layer)
    start = 0
    for layer in range(depth):
        count = 2 ** layer
        xs = np.linspace(x_span[0], x_span[1], count + 2)[1:-1]
        for j in range(count):
            idx = start + j
            pos_internal[idx] = (float(xs[j]), layer_y(layer))
        start += count

    # leaves at the last layer: evenly spaced
    xs_leaf = np.linspace(x_span[0], x_span[1], L + 2)[1:-1]
    for l in range(L):
        pos_leaf[l] = (float(xs_leaf[l]), layer_y(depth))

    return pos_internal, pos_leaf


def _best_path_for_sample(tree, x_single):
    """
    Given a single sample, return best path information.
    
    Args:
        tree: SDT模型
        x_single: 单个样本 (Tensor)
    
    Returns:
        path_internal: list of internal node indices (from root to the last internal node)
        lr_choices: list of 0/1, 0 means left, 1 means right
        best_leaf: leaf index
        lr_probs: list of left probabilities at each node
    """
    tree.eval()
    D = tree.depth
    Ni = tree.internal_node_num_

    with torch.no_grad():
        device = next(tree.parameters()).device
        x = x_single.to(device)

        # unify to [batch, dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Data augmentation consistent with the model (prepend constant 1)
        bias = torch.ones(x.size(0), 1, device=device, dtype=x.dtype)
        x_aug = torch.cat((bias, x), dim=1)

        logits_internal = tree.inner_nodes(x_aug)  # [1, Ni]
        path_prob = torch.sigmoid(tree.inv_temp * logits_internal)  # [1, Ni]

        # Traverse the tree, choosing the side with higher probability at each level
        path_internal = []
        lr_choices = []
        lr_probs = []
        curr_idx = 0  # root
        
        for layer in range(D):
            if layer == D:
                break
            if curr_idx >= Ni:
                break
            path_internal.append(curr_idx)
            p = path_prob[0, curr_idx].item()
            go_right = 1 if (1.0 - p) > p else 0
            lr_choices.append(go_right)
            lr_probs.append(p)
            left_child = 2 * curr_idx + 1
            right_child = 2 * curr_idx + 2
            curr_idx = right_child if go_right == 1 else left_child
            if curr_idx >= Ni:
                break

        # compute leaf index when reaching the last internal layer
        last_layer_start = 2 ** (D - 1) - 1
        idx_in_last_layer = path_internal[-1] - last_layer_start
        last_choice = lr_choices[-1] if lr_choices else 0
        best_leaf = 2 * idx_in_last_layer + last_choice

    return path_internal, lr_choices, best_leaf, lr_probs


# ===================== SDT树可视化 =====================

def visualize_sdt(tree, x_single, figsize=(10, 6), title=None):
    """
    Visualize the entire SDT:
    - Draw all nodes and edges;
    - Highlight the best path (relative to the given sample) in green;
    - Annotate left/right probability at internal nodes along the best path.
    
    Requirement: tree.hard_leaf_inference must be True.
    
    Args:
        tree: SDT模型 (hard_leaf_inference=True)
        x_single: 单个样本的特征向量 (Tensor)
        figsize: 图形大小
        title: 图形标题
    
    Returns:
        fig: matplotlib Figure
        ax: matplotlib Axes
        info: dict, 包含提取的SDT参数信息
    """
    if not getattr(tree, 'hard_leaf_inference', False):
        raise ValueError('hard_leaf_inference 必须为 True 才能进行该可视化。'
                         '请在构造 SDT 时设置 hard_leaf_inference=True。')

    D = tree.depth
    Ni = tree.internal_node_num_

    info = extract_sdt_parameters(tree)
    pos_int, pos_leaf = _binary_tree_positions(D)

    # compute the best path for the single sample
    path_internal, lr_choices, best_leaf, lr_probs = _best_path_for_sample(tree, x_single)
    on_path = set(path_internal)
    left_prob_by_idx = {path_internal[k]: lr_probs[k] for k in range(len(path_internal))}

    best_edges = set()
    for k, i in enumerate(path_internal):
        left = 2 * i + 1
        right = 2 * i + 2
        choose_right = (lr_choices[k] == 1) if k < len(lr_choices) else False
        child = right if choose_right else left
        if child < Ni:
            best_edges.add((i, child))
        else:
            last_start = 2 ** (D - 1) - 1
            idx_in_last = i - last_start
            l_left = 2 * idx_in_last
            l_right = l_left + 1
            leaf_child = l_right if choose_right else l_left
            best_edges.add((i, Ni + leaf_child))

    # draw
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # draw edges (all)
    for i in range(Ni):
        xi, yi = pos_int[i]
        left = 2 * i + 1
        right = 2 * i + 2
        # left edge
        if left < Ni:
            xj, yj = pos_int[left]
            col = ESSENTIAL_COLORS['best_edge'] if (i, left) in best_edges else ESSENTIAL_COLORS['edge']
            ax.plot([xi, xj], [yi, yj], color=col, linewidth=2 if col == ESSENTIAL_COLORS['best_edge'] else 1)
        else:
            last_start = 2 ** (D - 1) - 1
            idx_in_last = i - last_start
            l_left = 2 * idx_in_last
            xj, yj = pos_leaf[l_left]
            col = ESSENTIAL_COLORS['best_edge'] if (i, Ni + l_left) in best_edges else ESSENTIAL_COLORS['edge']
            ax.plot([xi, xj], [yi, yj], color=col, linewidth=2 if col == ESSENTIAL_COLORS['best_edge'] else 1)
        # right edge
        if right < Ni:
            xj, yj = pos_int[right]
            col = ESSENTIAL_COLORS['best_edge'] if (i, right) in best_edges else ESSENTIAL_COLORS['edge']
            ax.plot([xi, xj], [yi, yj], color=col, linewidth=2 if col == ESSENTIAL_COLORS['best_edge'] else 1)
        else:
            last_start = 2 ** (D - 1) - 1
            idx_in_last = i - last_start
            l_right = 2 * idx_in_last + 1
            xj, yj = pos_leaf[l_right]
            col = ESSENTIAL_COLORS['best_edge'] if (i, Ni + l_right) in best_edges else ESSENTIAL_COLORS['edge']
            ax.plot([xi, xj], [yi, yj], color=col, linewidth=2 if col == ESSENTIAL_COLORS['best_edge'] else 1)

    # draw internal nodes + annotations
    for i in range(Ni):
        xi, yi = pos_int[i]
        ax.scatter([xi], [yi], s=200, marker='o', 
                   color=ESSENTIAL_COLORS['internal_node'], edgecolors='black', zorder=3)
        ax.text(xi+0.025, yi, f"IN {i}", ha='center', va='bottom', fontsize=9)
        if i in on_path:
            p_left = float(left_prob_by_idx.get(i, 0.0))
            p_right = 1.0 - p_left
            ax.text(xi, yi-0.2, f"L:{p_left:.2f} R:{p_right:.2f}", 
                    color=ESSENTIAL_COLORS['best_edge'], ha='center', va='bottom', fontsize=8)

    # draw leaves
    L = 2 ** D
    for l in range(L):
        xj, yj = pos_leaf[l]
        ax.scatter([xj], [yj], s=180, marker='s', 
                   color=ESSENTIAL_COLORS['leaf_node'], edgecolors='black', zorder=3)
        leaf_logits = info['leaves'][l]['class_logits']
        top1 = int(torch.argmax(leaf_logits).item())
        ax.text(xj, yj-0.1, f"L {l}\nC {top1}", ha='center', va='top', fontsize=7)

    if title is None:
        title = f"Soft Decision Tree (depth={D}) | best leaf={best_leaf}"
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax, info


# ===================== 内部节点权重可视化 =====================

def visualize_internal_node_weight(W, b=None,
                                   mode='heatmap',
                                   image_shape=None,
                                   cmap='gray',
                                   normalize=True,
                                   show_colorbar=True,
                                   figsize=(4, 4),
                                   title=None,
                                   ax=None):
    """
    Visualize an internal node's weight vector W (and optional bias b).
    
    Args:
        W: 1D vector (torch.Tensor or numpy.ndarray), length = input dimension
        b: optional bias (float), will be shown in the title if provided
        mode: 'heatmap' or 'heatvector'
            * heatmap: requires image_shape=(H, W), reshape the vector to HxW heatmap
            * heatvector: show the vector as a 1xN heatmap
        image_shape: tuple (H, W). Required when mode='heatmap';
                     if not provided, will attempt to infer sqrt(N) x sqrt(N) if possible.
        cmap: colormap, default 'gray' (grayscale), or pass 'viridis', etc.
        normalize: whether to set vmin/vmax from current weight range for local contrast
        show_colorbar: whether to show colorbar
        figsize: figure size
        title: title; by default it will include mode and b if provided
        ax: optional Matplotlib Axes; if not provided, a new figure will be created
    
    Returns:
        fig: matplotlib Figure
        ax: matplotlib Axes
    """
    # convert to numpy vector
    if isinstance(W, torch.Tensor):
        w = W.detach().cpu().float().numpy()
    else:
        w = np.asarray(W, dtype=np.float32)

    N = w.size
    # infer/validate shape
    if mode == 'heatmap':
        if image_shape is None:
            r = int(np.sqrt(N))
            if r * r == N:
                image_shape = (r, r)
            else:
                raise ValueError(f"heatmap 需要 image_shape，但 len(W)={N} 不是完美平方，无法自动推断。")
        H, Ww = int(image_shape[0]), int(image_shape[1])
        if H * Ww != N:
            raise ValueError(f"image_shape={image_shape} 与 len(weight)={N} 不匹配。")
        img = w.reshape(H, Ww)
    elif mode == 'heatvector':
        img = w.reshape(1, N)
    else:
        raise ValueError("mode 必须为 'heatmap' 或 'heatvector'")

    # normalization range
    vmin = vmax = None
    if normalize:
        wmin, wmax = np.min(w), np.max(w)
        if wmax > wmin:
            vmin, vmax = wmin, wmax

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(img, cmap=cmap, aspect='auto' if mode == 'heatvector' else 'equal',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is None:
        if b is not None:
            title = f"{mode} | b={b:.4f}"
        else:
            title = f"{mode}"
    ax.set_title(title)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if created_fig:
        plt.tight_layout()
    return fig, ax


# ===================== 样本数据获取与可视化 =====================

# 默认的类别颜色配置
DEFAULT_CATEGORY_COLORS = {
    "nuclear_morphology_nc": "#e41a1c",      # 红色 - 细胞核形态
    "cytoplasmic_tone_texture": "#377eb8",   # 蓝色 - 胞质色调与质地
    "cytoplasmic_granules": "#4daf4a",       # 绿色 - 胞质颗粒与内含物
    "non_leukocyte_elements": "#984ea3",     # 紫色 - 非白细胞血细胞要素
    "artifacts_quality": "#ff7f00",          # 橙色 - 制片与技术伪影
}


def get_sample_data(n, concept_dataset, image_dataset):
    """
    根据样本索引获取concept score vector和原图像。
    
    Args:
        n: 样本索引（在数据集中的编号）
        concept_dataset: concept score vector的Dataset (TensorDataset)
        image_dataset: 原图像数据集的Dataset (如BloodMNIST)
    
    Returns:
        sample: concept score vector (Tensor, shape [num_concepts])
        img: 原图像 (numpy array, shape [H, W, C])
        label: 标签 (int)
    """
    # 从concept dataset获取特征向量
    sample, label_tensor = concept_dataset[n]
    if isinstance(label_tensor, torch.Tensor):
        label = label_tensor.item()
    else:
        label = int(label_tensor)
    
    # 从image dataset获取原图像
    img_tensor, img_label = image_dataset[n]
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    else:
        img = np.array(img_tensor)
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # [C, H, W] format
            img = img.transpose(1, 2, 0)
    
    return sample, img, label


def build_concept_colors(cell_feature_metadata, category_colors=None):
    """
    根据cell_feature_metadata构建每个concept对应的颜色列表。
    
    Args:
        cell_feature_metadata: 细胞特征元数据字典
        category_colors: 类别颜色字典，如果为None则使用默认颜色
    
    Returns:
        concept_colors: 每个concept对应的颜色列表
    """
    if category_colors is None:
        category_colors = DEFAULT_CATEGORY_COLORS
    
    concept_colors = []
    for category_key, category_data in cell_feature_metadata.items():
        color = category_colors.get(category_key, "black")
        for feat in category_data.get("features", []):
            concept_colors.append(color)
    
    return concept_colors


def plot_sample_with_concepts(n, concept_dataset, image_dataset,
                              concept_list, label_dict,
                              cell_feature_metadata=None,
                              category_colors=None,
                              concept_colors=None,
                              figsize=(14, 8),
                              normalize_concepts=True):
    """
    绘制样本的原图像和对应的concept score vector柱状图。
    
    Args:
        n: 样本索引
        concept_dataset: concept score vector的Dataset
        image_dataset: 原图像数据集的Dataset
        concept_list: concept名称列表
        label_dict: 标签字典 {label_id: label_name}
        cell_feature_metadata: 细胞特征元数据（用于构建concept_colors）
        category_colors: 类别颜色字典
        concept_colors: 预先构建的concept颜色列表（优先使用）
        figsize: 图形大小
        normalize_concepts: 是否对concept值进行归一化
    
    Returns:
        sample: concept score vector (Tensor)
        img: 原图像 (numpy array)
        label: 标签 (int)
        fig: matplotlib Figure
        axes: matplotlib Axes数组
    """
    from matplotlib.patches import Patch
    
    # 获取样本数据
    sample, img, label = get_sample_data(n, concept_dataset, image_dataset)
    
    # 归一化concept值用于可视化
    if normalize_concepts:
        sample_viz = sample.clone()
        sample_viz = (sample_viz - sample_viz.min()) / (sample_viz.max() - sample_viz.min() + 1e-8)
    else:
        sample_viz = sample
    
    # 构建concept颜色（如果未提供）
    if concept_colors is None and cell_feature_metadata is not None:
        concept_colors = build_concept_colors(cell_feature_metadata, category_colors)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize, 
                              gridspec_kw={'width_ratios': [1, 3]})
    
    # 左图：原图像
    ax_img = axes[0]
    ax_img.imshow(img)
    ax_img.set_title(f"Label: [{label}] {label_dict.get(label, 'Unknown')}", fontsize=11)
    ax_img.axis('off')
    
    # 右图：concept score柱状图
    ax_bar = axes[1]
    bars = ax_bar.bar(range(len(concept_list)), 
                       sample_viz.squeeze().numpy() if isinstance(sample_viz, torch.Tensor) else sample_viz,
                       color='skyblue')
    ax_bar.set_xticks(range(len(concept_list)))
    ax_bar.set_xticklabels(concept_list, rotation=45, ha='right', fontsize=6)
    
    # 为x轴标签设置颜色
    if concept_colors is not None:
        for ticklabel, color in zip(ax_bar.get_xticklabels(), concept_colors):
            ticklabel.set_color(color)
    
    ax_bar.set_xlabel('Concepts', fontsize=10)
    ax_bar.set_ylabel('Concept Value' + (' (normalized)' if normalize_concepts else ''), fontsize=10)
    ax_bar.set_title(f'Sample Index: {n}, True Label: {label} ({label_dict.get(label, "Unknown")})', fontsize=11)
    
    # 添加图例（如果有category_colors）
    if cell_feature_metadata is not None:
        if category_colors is None:
            category_colors = DEFAULT_CATEGORY_COLORS
        legend_elements = [Patch(facecolor=color, label=f"{cell_feature_metadata[key]['en']}") 
                           for key, color in category_colors.items() if key in cell_feature_metadata]
        ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    plt.tight_layout()
    
    return sample, img, label, fig, axes


# ===================== 节点权重heatvector可视化 =====================

def plot_node_heatvector(node_idx, info, concept_list,
                         cell_feature_metadata=None,
                         category_colors=None,
                         concept_colors=None,
                         cmap='Reds',
                         figsize=(14, 4),
                         show=True,
                         save_path=None):
    """
    绘制指定内部节点的权重heatvector图。
    
    Args:
        node_idx: 内部节点索引
        info: extract_sdt_parameters返回的信息字典
        concept_list: concept名称列表
        cell_feature_metadata: 细胞特征元数据（用于构建图例）
        category_colors: 类别颜色字典
        concept_colors: 预先构建的concept颜色列表
        cmap: colormap名称
        figsize: 图形大小
        show: 是否显示图形
        save_path: 保存路径（如果提供则保存图片）
    
    Returns:
        fig: matplotlib Figure
        ax: matplotlib Axes
    """
    from matplotlib.patches import Patch
    
    # 获取节点信息
    node = info['internal_nodes'][node_idx]
    W_vec = node['W']
    bias = node['b']
    
    # 绘制heatvector
    fig, ax = visualize_internal_node_weight(
        W_vec, b=bias, mode='heatvector', 
        cmap=cmap, figsize=figsize, show_colorbar=True
    )
    
    # 设置x轴标签
    ax.set_xticks(range(len(concept_list)))
    ax.set_xticklabels(concept_list, rotation=45, ha='right', fontsize=6)
    
    # 为x轴标签设置颜色
    if concept_colors is not None:
        for ticklabel, color in zip(ax.get_xticklabels(), concept_colors):
            ticklabel.set_color(color)
    
    ax.set_title(f'Internal Node {node_idx}: Weight Vector (bias={bias:.4f})')
    
    # 添加图例
    if cell_feature_metadata is not None:
        if category_colors is None:
            category_colors = DEFAULT_CATEGORY_COLORS
        legend_elements = [Patch(facecolor=color, label=f"{cell_feature_metadata[key]['en']}") 
                           for key, color in category_colors.items() if key in cell_feature_metadata]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6, ncol=2)
    
    fig.subplots_adjust(bottom=0.45)
    
    # 保存图片
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def export_node_weight_csv(node_idx, info, concept_list, cell_feature_metadata, 
                           output_dir, threshold=0.5):
    """
    导出单个内部节点的权重向量到CSV文件。
    
    Args:
        node_idx: 内部节点索引
        info: extract_sdt_parameters返回的信息字典
        concept_list: concept名称列表（英文）
        cell_feature_metadata: 细胞特征元数据，用于获取中文翻译
        output_dir: 输出目录（节点子目录，如 ./outputs_70/tree_nodes/IN_0）
        threshold: 筛选阈值，用于生成filtered CSV
    
    Returns:
        (csv_path, filtered_csv_path): 完整CSV和筛选后CSV的路径
    """
    import pandas as pd
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # 获取节点权重
    node = info['internal_nodes'][node_idx]
    W_vec = node['W']
    
    # 转换为numpy数组
    if hasattr(W_vec, 'cpu'):
        W_np = W_vec.cpu().numpy()
    else:
        W_np = np.array(W_vec)
    
    # 构建英文到中文的映射
    en_to_zh = {}
    for category in cell_feature_metadata.values():
        for feat in category.get("features", []):
            en_to_zh[feat["en"]] = feat["zh"]
    
    # 构建DataFrame
    data = []
    for i, concept_en in enumerate(concept_list):
        concept_zh = en_to_zh.get(concept_en, "")
        weight_value = W_np[i] if i < len(W_np) else 0.0
        data.append({
            "concept_en": concept_en,
            "concept_zh": concept_zh,
            "weight": float(weight_value)
        })
    
    df = pd.DataFrame(data)
    
    # 保存完整CSV
    node_name = f"IN_{node_idx}"
    csv_path = output_dir / f"weights_{node_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 筛选大于阈值的条目
    df_filtered = df[df['weight'] > threshold].copy()
    df_filtered = df_filtered.sort_values(by='weight', ascending=False)
    filtered_csv_path = output_dir / f"weights_{node_name}_gt{threshold}.csv"
    df_filtered.to_csv(filtered_csv_path, index=False, encoding='utf-8-sig')
    
    return str(csv_path), str(filtered_csv_path)


def batch_export_node_heatvectors(info, concept_list,
                                   output_dir,
                                   cell_feature_metadata=None,
                                   category_colors=None,
                                   concept_colors=None,
                                   cmap='Reds',
                                   figsize=(14, 4),
                                   node_indices=None,
                                   export_csv=True,
                                   csv_threshold=0.5):
    """
    批量导出所有内部节点的heatvector图至指定文件夹。
    
    Args:
        info: extract_sdt_parameters返回的信息字典
        concept_list: concept名称列表
        output_dir: 输出根目录 (如 "./outputs_70/tree_nodes")
        cell_feature_metadata: 细胞特征元数据
        category_colors: 类别颜色字典
        concept_colors: 预先构建的concept颜色列表
        cmap: colormap名称
        figsize: 图形大小
        node_indices: 要导出的节点索引列表，如果为None则导出所有内部节点
        export_csv: 是否同时导出权重CSV文件
        csv_threshold: CSV筛选阈值（导出大于该阈值的条目）
    
    Returns:
        exported_files: 导出的文件路径列表
    """
    import os
    from pathlib import Path
    
    output_dir = Path(output_dir)
    exported_files = []
    
    # 确定要导出的节点
    if node_indices is None:
        node_indices = range(len(info['internal_nodes']))
    
    total = len(node_indices)
    
    for i, node_idx in enumerate(node_indices):
        # 创建节点子目录
        node_name = f"IN_{node_idx}"
        node_dir = output_dir / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存路径
        save_path = node_dir / f"heatvector_{node_name}.png"
        
        # 绘制并保存
        plot_node_heatvector(
            node_idx=node_idx,
            info=info,
            concept_list=concept_list,
            cell_feature_metadata=cell_feature_metadata,
            category_colors=category_colors,
            concept_colors=concept_colors,
            cmap=cmap,
            figsize=figsize,
            show=False,
            save_path=str(save_path)
        )
        
        exported_files.append(str(save_path))
        
        # 导出CSV文件
        csv_info = ""
        if export_csv and cell_feature_metadata is not None:
            csv_path, filtered_csv_path = export_node_weight_csv(
                node_idx=node_idx,
                info=info,
                concept_list=concept_list,
                cell_feature_metadata=cell_feature_metadata,
                output_dir=node_dir,
                threshold=csv_threshold
            )
            exported_files.append(csv_path)
            exported_files.append(filtered_csv_path)
            csv_info = " + CSV"
        
        print(f"[{i+1}/{total}] Exported: {save_path}{csv_info}")
    
    print(f"\n共导出 {len(exported_files)} 个文件至: {output_dir}")
    return exported_files


# ===================== 节点激活分析 =====================

def compute_node_logits_for_dataset(tree, X_data, device):
    """
    计算 SDT 所有内部节点在数据集上的 logits (WX + b)
    
    Args:
        tree: SDT 模型
        X_data: numpy array, shape (N, input_dim), 标准化后的特征
        device: 计算设备
    
    Returns:
        node_logits: numpy array, shape (N, num_internal_nodes)
                     每行是一个样本，每列是一个内部节点的 logits
    """
    tree.eval()
    with torch.no_grad():
        # 转换为 tensor
        X_tensor = torch.from_numpy(X_data).float().to(device)
        
        # 数据增强：在前面加上常数 1（与模型内部一致）
        batch_size = X_tensor.size(0)
        bias_col = torch.ones(batch_size, 1, device=device, dtype=X_tensor.dtype)
        X_aug = torch.cat((bias_col, X_tensor), dim=1)
        
        # 计算所有内部节点的 logits
        logits = tree.inner_nodes(X_aug)  # shape: (N, num_internal_nodes)
        
        node_logits = logits.cpu().numpy()
    
    return node_logits


def visualize_top_k_images_for_node(node_idx, node_logits, ds_images, y_labels, 
                                     label_dict, tree_info, k=20, figsize=(20, 16)):
    """
    可视化某个内部节点激活值最高的 Top-K 图像
    
    Args:
        node_idx: 内部节点索引
        node_logits: shape (N, num_nodes), 所有节点的 logits
        ds_images: 图像数据集 (medmnist dataset 或 ConcatDataset)
        y_labels: 标签数组
        label_dict: 标签索引到名称的映射
        tree_info: extract_sdt_parameters 返回的信息字典
        k: 展示的图像数量
        figsize: 图像大小
    
    Returns:
        fig: matplotlib Figure
        top_k_indices: Top-K 样本索引
        top_k_scores: Top-K 样本分数
    """
    # 获取该节点的所有 logits
    scores = node_logits[:, node_idx]
    
    # 排序，获取 top-k 索引
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_indices]
    
    # 计算布局
    n_cols = 5
    n_rows = (k + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # 获取节点权重信息
    node_info = tree_info['internal_nodes'][node_idx]
    layer = node_info['layer']
    bias = node_info['b']
    
    fig.suptitle(f'内部节点 {node_idx} (Layer {layer}) 激活值 Top-{k} 图像\n'
                 f'bias = {bias:.4f}', fontsize=14, fontweight='bold')
    
    for i, (sample_idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        ax = axes[i]
        
        # 获取原始图像
        img_tensor = ds_images[sample_idx][0]
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.numpy().transpose(1, 2, 0)
        else:
            img = np.array(img_tensor)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
        
        # 获取标签
        label = y_labels[sample_idx]
        label_name = label_dict[label]
        
        ax.imshow(img)
        ax.set_title(f'#{sample_idx}\nScore: {score:.3f}\n{label_name}', fontsize=9)
        ax.axis('off')
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig, top_k_indices, top_k_scores


def visualize_top_and_bottom_k_images_for_node(node_idx, node_logits, ds_images, y_labels, 
                                                label_dict, tree_info, k=10, figsize=(20, 16)):
    """
    同时可视化某个内部节点激活值最高和最低的 Top-K 图像
    用于对比分析该节点学到的判别特征
    
    Args:
        node_idx: 内部节点索引
        node_logits: shape (N, num_nodes), 所有节点的 logits
        ds_images: 图像数据集 (medmnist dataset 或 ConcatDataset)
        y_labels: 标签数组
        label_dict: 标签索引到名称的映射
        tree_info: extract_sdt_parameters 返回的信息字典
        k: 展示的图像数量（高/低各 k 个）
        figsize: 图像大小
    
    Returns:
        fig: matplotlib Figure
        top_k_indices: 高激活样本索引
        bottom_k_indices: 低激活样本索引
    """
    scores = node_logits[:, node_idx]
    sorted_indices = np.argsort(scores)
    
    top_k_indices = sorted_indices[::-1][:k]  # 最高
    bottom_k_indices = sorted_indices[:k]      # 最低
    
    n_cols = 5
    n_rows = (k + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
    
    # 获取节点权重信息
    node_info = tree_info['internal_nodes'][node_idx]
    layer = node_info['layer']
    bias = node_info['b']
    
    fig.suptitle(f'内部节点 {node_idx} (Layer {layer}) | bias = {bias:.4f}\n'
                 f'上半部分: 激活值最高 Top-{k} | 下半部分: 激活值最低 Top-{k}', 
                 fontsize=14, fontweight='bold')
    
    # Top-K (高激活)
    for i, sample_idx in enumerate(top_k_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        img_tensor = ds_images[sample_idx][0]
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.numpy().transpose(1, 2, 0)
        else:
            img = np.array(img_tensor)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
        
        label = y_labels[sample_idx]
        label_name = label_dict[label]
        score = scores[sample_idx]
        
        ax.imshow(img)
        ax.set_title(f'#{sample_idx}\nScore: {score:.3f}\n{label_name}', fontsize=8)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('HIGH', fontsize=10, color='green', fontweight='bold')
    
    # Bottom-K (低激活)
    for i, sample_idx in enumerate(bottom_k_indices):
        row = n_rows + i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        img_tensor = ds_images[sample_idx][0]
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.numpy().transpose(1, 2, 0)
        else:
            img = np.array(img_tensor)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
        
        label = y_labels[sample_idx]
        label_name = label_dict[label]
        score = scores[sample_idx]
        
        ax.imshow(img)
        ax.set_title(f'#{sample_idx}\nScore: {score:.3f}\n{label_name}', fontsize=8)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('LOW', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig, top_k_indices, bottom_k_indices


def analyze_all_nodes_summary(node_logits, y_labels, label_dict, tree_info):
    """
    生成所有节点的激活分析摘要
    
    Args:
        node_logits: shape (N, num_nodes), 所有节点的 logits
        y_labels: 标签数组
        label_dict: 标签索引到名称的映射
        tree_info: extract_sdt_parameters 返回的信息字典
    
    Returns:
        DataFrame: 包含每个节点的统计信息
    """
    import pandas as pd
    from collections import Counter
    
    results = []
    num_nodes = node_logits.shape[1]
    
    for node_idx in range(num_nodes):
        scores = node_logits[:, node_idx]
        top_20_indices = np.argsort(scores)[::-1][:20]
        bottom_20_indices = np.argsort(scores)[:20]
        
        # Top-20 标签分布
        top_labels = [y_labels[i] for i in top_20_indices]
        bottom_labels = [y_labels[i] for i in bottom_20_indices]
        
        top_label_counts = Counter(top_labels)
        bottom_label_counts = Counter(bottom_labels)
        
        # 最常见的标签
        top_dominant_label = top_label_counts.most_common(1)[0] if top_label_counts else (None, 0)
        bottom_dominant_label = bottom_label_counts.most_common(1)[0] if bottom_label_counts else (None, 0)
        
        node_info = tree_info['internal_nodes'][node_idx]
        
        results.append({
            'node_idx': node_idx,
            'layer': node_info['layer'],
            'bias': node_info['b'],
            'score_mean': scores.mean(),
            'score_std': scores.std(),
            'score_max': scores.max(),
            'score_min': scores.min(),
            'top20_dominant_label': label_dict[top_dominant_label[0]] if top_dominant_label[0] is not None else None,
            'top20_dominant_count': top_dominant_label[1],
            'bottom20_dominant_label': label_dict[bottom_dominant_label[0]] if bottom_dominant_label[0] is not None else None,
            'bottom20_dominant_count': bottom_dominant_label[1],
        })
    
    return pd.DataFrame(results)
