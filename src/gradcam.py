"""Reusable Grad-CAM explainability for evaluation, notebooks, and UI apps."""
import keras
import numpy as np
import tensorflow as tf


def _layer_output_rank(layer) -> int | None:
    """Safely extract output tensor rank across Keras 3 layers and nested models."""
    try:
        shape = layer.output.shape
    except Exception:
        shape = getattr(layer, "output_shape", None)
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    try:
        return len(shape)
    except TypeError:
        return None


def find_last_spatial_layer_index(model) -> int:
    """Find the top-level layer index producing the final 4D spatial feature map."""
    layers = model.layers
    for idx in range(len(layers) - 1, -1, -1):
        layer = layers[idx]
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if isinstance(layer, keras.Model) and _layer_output_rank(layer) == 4:
            return idx

    last_conv_idx = None
    for idx in range(len(layers) - 1, -1, -1):
        if isinstance(layers[idx], keras.layers.Conv2D):
            last_conv_idx = idx
            break

    if last_conv_idx is not None:
        target_idx = last_conv_idx
        for idx in range(last_conv_idx + 1, len(layers)):
            layer = layers[idx]
            if isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Activation)):
                if _layer_output_rank(layer) == 4:
                    target_idx = idx
            else:
                break
        return target_idx

    for idx in range(len(layers) - 1, -1, -1):
        layer = layers[idx]
        if not isinstance(layer, keras.layers.InputLayer) and _layer_output_rank(layer) == 4:
            return idx

    raise ValueError("Model tidak memiliki layer spasial 4D untuk Grad-CAM.")


def compute_gradcam(
    model,
    img_batch,
    class_index: int | None = None,
    spatial_layer_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute normalized 2D Grad-CAM heatmap, predicted probs, and target class index."""
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    split_idx = (
        spatial_layer_index
        if spatial_layer_index is not None
        else find_last_spatial_layer_index(model)
    )
    layers = model.layers

    for idx in range(split_idx + 1):
        layer = layers[idx]
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x, training=False)
    conv_outputs = x

    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        feat = conv_outputs
        for layer in layers[split_idx + 1 : -1]:
            feat = layer(feat, training=False)

        last_layer = layers[-1]
        is_softmax_dense = (
            isinstance(last_layer, keras.layers.Dense)
            and getattr(last_layer.activation, "__name__", "") == "softmax"
        )
        if is_softmax_dense:
            feat_f32 = tf.cast(feat, tf.float32)
            kernel_f32 = tf.cast(last_layer.kernel, tf.float32)
            logits = tf.matmul(feat_f32, kernel_f32)
            if last_layer.use_bias:
                logits = logits + tf.cast(last_layer.bias, tf.float32)
            probs = tf.nn.softmax(logits, axis=-1)
            target_idx = int(tf.argmax(probs[0])) if class_index is None else int(class_index)
            score = logits[:, target_idx]
        else:
            probs = tf.cast(last_layer(feat, training=False), tf.float32)
            target_idx = int(tf.argmax(probs[0])) if class_index is None else int(class_index)
            score = probs[:, target_idx]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gagal menghitung gradien terhadap layer spasial.")

    pooled_grads = tf.cast(tf.reduce_mean(grads, axis=(0, 1, 2)), tf.float32)
    conv_f32 = tf.cast(conv_outputs[0], tf.float32)
    heatmap = tf.reduce_sum(conv_f32 * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)
    max_val = float(heatmap.max())
    if max_val > 1e-12:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    return heatmap, probs[0].numpy().astype(np.float32), target_idx


def overlay_heatmap(
    rgb_uint8: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Resize 2D heatmap to image size and alpha-blend onto an RGB uint8 image."""
    from matplotlib import colormaps
    from PIL import Image

    height, width = rgb_uint8.shape[:2]
    hm = np.clip(np.asarray(heatmap_2d, dtype=np.float32), 0.0, 1.0)
    hm_img = Image.fromarray((hm * 255.0).astype(np.uint8), mode="L")
    hm_resized = np.asarray(
        hm_img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0

    cmap = colormaps[colormap]
    colored = (cmap(hm_resized)[..., :3] * 255.0).astype(np.float32)
    blended = (1.0 - float(alpha)) * rgb_uint8.astype(np.float32) + float(alpha) * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def explain_image(
    model,
    image_or_path,
    cfg: dict,
    class_names: list[str],
    class_index: int | None = None,
    top_k: int = 3,
    alpha: float | None = None,
    colormap: str = "jet",
) -> dict:
    """Run preprocessing, prediction, and Grad-CAM overlay for one image."""
    from src.data import preprocess_image

    display_rgb, batch = preprocess_image(
        image_or_path, cfg["data"], cfg["model"]["arch"]
    )
    heatmap, probs, target_idx = compute_gradcam(model, batch, class_index=class_index)
    blend_alpha = cfg.get("eval", {}).get("gradcam_alpha", 0.4) if alpha is None else alpha
    overlay_rgb = overlay_heatmap(display_rgb, heatmap, alpha=blend_alpha, colormap=colormap)

    pred_idx = int(np.argmax(probs))
    k = max(1, min(int(top_k), len(class_names)))
    order = np.argsort(probs)[::-1][:k]
    top_k_preds = [
        {"class": class_names[int(i)], "prob": float(probs[int(i)])}
        for i in order
    ]
    return {
        "pred_index": pred_idx,
        "pred_class": class_names[pred_idx],
        "pred_prob": float(probs[pred_idx]),
        "target_index": target_idx,
        "target_class": class_names[target_idx],
        "top_k": top_k_preds,
        "original_rgb": display_rgb,
        "heatmap": heatmap,
        "overlay_rgb": overlay_rgb,
    }
