"""图像读写与尺寸变换辅助。

统一规则：把原图(可能 5472x3648)按长边缩放到 max_side，短边四舍五入到 16 的倍数
(U-Net 4 次下采样要求)。图像与掩码使用同一目标尺寸，保证像素一一对应。
"""
import numpy as np
import torch
from PIL import Image


def load_rgb(path) -> np.ndarray:
    """读图并统一转 RGB，返回 uint8 (h, w, 3)。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def target_size(w0: int, h0: int, max_side: int, stride: int = 16) -> tuple:
    """由原图宽高返回模型输入尺寸 (w1, h1)，均为 stride 的倍数。"""
    scale = max_side / float(max(w0, h0))
    w1 = max(stride, int(round(w0 * scale / stride)) * stride)
    h1 = max(stride, int(round(h0 * scale / stride)) * stride)
    return w1, h1


def _to_pil(arr, mode="RGB"):
    return Image.fromarray(arr, mode=mode)


def resize_rgb(img: np.ndarray, w1: int, h1: int) -> np.ndarray:
    """双线性缩放到 (w1, h1)，返回 uint8 (h1, w1, 3)。"""
    return np.asarray(_to_pil(img).resize((w1, h1), Image.Resampling.BILINEAR),
                      dtype=np.uint8)


def resize_bool_mask(mask: np.ndarray, w1: int, h1: int) -> np.ndarray:
    """最近邻缩放二值掩码到 (w1, h1)，返回 bool (h1, w1)。"""
    arr = np.where(mask, 255, 0).astype(np.uint8)
    small = np.asarray(_to_pil(arr, "L").resize((w1, h1), Image.Resampling.NEAREST),
                       dtype=np.uint8)
    return small > 127


def to_model_input(img: np.ndarray) -> torch.Tensor:
    """uint8 (h, w, 3) -> float32 [1, 3, h, w]，值域 0~1。"""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)  # (h,w,3)
    return t.permute(2, 0, 1).unsqueeze(0)


def prob_to_orig_mask(prob: torch.Tensor, w0: int, h0: int,
                      threshold: float = 0.5, channel: int = 0) -> np.ndarray:
    """模型概率图 [1,C,h1,w1] -> 取第 channel 通道、双线性放大回原图尺寸 -> 阈值二值化。

    返回 bool (h0, w0)；prob 在 cuda 上亦可，输出为 numpy bool（CPU）。
    channel 必须显式给定（多通道模型下默认取根系通道）。
    """
    h1, w1 = prob.shape[-2], prob.shape[-1]
    # 只上采样要用的那个通道：全分辨率的插值很贵（5472x3648 上三通道 89ms、单通道 37ms），
    # 而双线性插值逐通道独立，结果与「整幅上采样后取第 channel 个」**逐位相同**。
    one = prob[:, channel:channel + 1]
    if (w1, h1) != (w0, h0):
        up = torch.nn.functional.interpolate(one.cpu().float(), size=(h0, w0),
                                             mode="bilinear", align_corners=False)
    else:
        up = one.cpu().float()
    return (up[0, 0] > threshold).numpy()


def prob_to_orig_mask_hysteresis(prob: torch.Tensor, w0: int, h0: int,
                                 high: float = 0.5,
                                 low: float = 0.15,
                                 channel: int = 0) -> np.ndarray:
    """滞回阈值二值化（推荐用于"统计根数/长度"），只适用于根系这类细线结构。

    强阈值(>high)确定可靠根段；弱阈值(>low)把细弱处断续的段接回，
    避免一根根被断开后统计成多根。返回 bool (h0, w0)。
    """
    from skimage.measure import label  # 延迟导入，避免拖慢模块加载
    h1, w1 = prob.shape[-2], prob.shape[-1]
    one = prob[:, channel:channel + 1]      # 同上：只上采样要用的那一个通道
    if (w1, h1) != (w0, h0):
        up = torch.nn.functional.interpolate(one.cpu().float(), size=(h0, w0),
                                             mode="bilinear", align_corners=False)
    else:
        up = one.cpu().float()
    p = up[0, 0].numpy()
    weak = p > low
    strong = p > high
    if not weak.any():
        return np.zeros((h0, w0), dtype=bool)
    lab = label(weak, connectivity=2)
    keep = np.unique(lab[strong])
    # 查表代替 np.isin：全分辨率 2000 万像素上，isin 139ms、查表 52ms，结果逐位相同
    # （isin 内部要排序/二分，查表是一次 O(N) 索引）
    table = np.zeros(int(lab.max()) + 1, dtype=bool)
    table[keep[keep > 0]] = True
    return table[lab]
