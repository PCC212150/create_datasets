"""RSML 根系标注文件解析。

RSML 为 XML：<scene> 下多个 <plant>；plant 下可有 0..N 个 <root>；
每个 <root> 的 <geometry> 内含折线控制点 <point x=".." y=".."/>（本项目为 rootnavspline）。
按项目规范：无几何（无坐标点）的 plant/root 直接忽略。
"""
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass
class Root:
    root_id: str = ""                       # 根 ID，如 "2.1"
    label: str = ""                         # 如 "primary"
    points: list = field(default_factory=list)  # [(x, y), ...] 折线控制点(float)

    @property
    def length(self) -> float:
        """折线逐段欧氏距离求和（像素），与预测侧骨架 8 邻域链码口径一致。"""
        if len(self.points) < 2:
            return 0.0
        pts = np.asarray(self.points, dtype=np.float64)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _collect_points(root_el):
    """收集某个 <root> 直属几何的点；跳过嵌套子 <root>（子根由外层递归另行统计）。"""
    pts = []
    for child in root_el:
        if child.tag == "root":
            continue
        if child.tag == "point":
            x, y = child.get("x"), child.get("y")
            if x is not None and y is not None:
                pts.append((float(x), float(y)))
        else:
            pts.extend(_collect_points(child))
    return pts


def _walk_roots(container, out):
    """深度遍历所有 <root>（含嵌套子根），各自独立收集几何点。

    注意：要递归所有元素（plant 之下才出现 root；root 之下可能出现子 root）。
    """
    for el in container:
        if el.tag == "root":
            pts = _collect_points(el)
            if len(pts) >= 2:  # 少于 2 个点视为无有效几何
                out.append(Root(
                    root_id=el.get("ID", ""),
                    label=el.get("label", ""),
                    points=pts,
                ))
        _walk_roots(el, out)


def parse_rsml(rsml_path) -> list:
    """解析单个 .rsml 文件，返回所有含有效折线（>=2 点）的 Root 列表（文档序）。"""
    tree = ElementTree.parse(rsml_path)
    xml_root = tree.getroot()
    roots = []
    scene = xml_root.find("scene")
    _walk_roots(scene if scene is not None else xml_root, roots)
    return roots


def root_stats(roots) -> tuple:
    """返回 (根数, 逐根长度列表[文档序], 总长度)。"""
    lengths = [r.length for r in roots]
    return len(roots), lengths, float(sum(lengths))
