import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # optional dependency
    skimage_ssim = None


def masked_mae(pred, gt, mask):
    diff = np.abs(pred - gt)
    return float(diff[mask > 0].mean())


def masked_mse(pred, gt, mask):
    diff = (pred - gt) ** 2
    return float(diff[mask > 0].mean())


def masked_psnr(pred, gt, mask, data_range=None):
    mse = masked_mse(pred, gt, mask)
    if data_range is None:
        gt_vals = gt[mask > 0]
        data_range = float(gt_vals.max() - gt_vals.min())
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def masked_ssim(pred, gt, mask):
    if skimage_ssim is None:
        return None
    gt_vals = gt[mask > 0]
    data_range = float(gt_vals.max() - gt_vals.min())
    return float(skimage_ssim(gt, pred, data_range=data_range))


def gradient_magnitude(vol):
    gx = np.gradient(vol, axis=0)
    gy = np.gradient(vol, axis=1)
    gz = np.gradient(vol, axis=2)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def edge_ncc(edge_a, edge_b, mask):
    a = edge_a[mask > 0].reshape(-1)
    b = edge_b[mask > 0].reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def boundary_band(mask, vox=2):
    dil = binary_dilation(mask > 0, iterations=vox)
    ero = binary_erosion(mask > 0, iterations=vox)
    return (dil.astype(np.uint8) - ero.astype(np.uint8)).astype(np.uint8)


def z_gradient_energy(vol, mask):
    gz = np.abs(np.gradient(vol, axis=2))
    return float(gz[mask > 0].mean())


def summarize_metrics(pred, gt, t1, mask):
    out = {
        "mae": masked_mae(pred, gt, mask),
        "psnr": masked_psnr(pred, gt, mask),
    }
    ssim_val = masked_ssim(pred, gt, mask)
    if ssim_val is not None:
        out["ssim"] = ssim_val

    edge_t1 = gradient_magnitude(t1)
    edge_pred = gradient_magnitude(pred)
    out["edge_ncc"] = edge_ncc(edge_t1, edge_pred, mask)

    band = boundary_band(mask)
    out["band_mae"] = masked_mae(pred, gt, band)
    out["band_psnr"] = masked_psnr(pred, gt, band)

    out["z_grad_energy"] = z_gradient_energy(pred, mask)
    return out
