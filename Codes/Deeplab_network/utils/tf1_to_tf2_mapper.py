import argparse
import os, sys, datetime, json, pathlib, tempfile, shutil
import re
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_tf1_tensors(ckpt_prefix: str):
    """Return dict: {name: (ndarray, shape)} for all tensors in a TF1 checkpoint."""
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt_prefix)
    ckpt_var_to_shape = reader.get_variable_to_shape_map()
    tensors = {}
    for k, shape in ckpt_var_to_shape.items():
        tensors[k] = (reader.get_tensor(k), tuple(shape))
    return tensors

def build_deeplabv2_tf2_model(num_classes=4, input_shape=(513, 513, 3)):
    """
    It must produce variables whose names already include the 'deeplab_v2/...' scopes.
    """
    # Minimal dummy with same variable scopes is impossible without your real model code.
    # Here, we raise to make it clear you need to plug in your model.
    from models.deeplab import DeepLabV2
    model = DeepLabV2(num_classes=num_classes, freeze_bn=True)
    model.build(input_shape=(None, *input_shape))
    return model

def get_model_variables(model):
    """
    Returns list of (name, shape_tuple, var) for all TF2 variables, sorted by name.
    .name includes ':0' suffix; we strip that in the normalizer.
    """
    out = []
    for v in sorted(model.variables, key=lambda x: x.name):
        out.append((v.name, tuple(v.shape.as_list()), v))
    return out

_BR_RE = re.compile(r'^(res[0-9][a-z0-9]*)_branch(1|2[abc])$')  # e.g. res4b1_branch2a
_BN_RE = re.compile(r'^bn([0-9][a-z0-9]*)_branch(1|2[abc])$')    # e.g. bn4b1_branch2a
_ASPP_RE = re.compile(r'^fc1_voc12_c([0-3])$')                   # fc1_voc12_c0..3

def _strip_tf_suffixes(name: str) -> str:
    # Remove TF tensor suffix and optimizer slot names we don't want
    if name.endswith(':0'):
        name = name[:-2]
    return name

def should_skip_tf1_var(name: str) -> bool:
    """
    Skip optimizer slot variables (e.g. Momentum) and anything else you don't want to restore.
    """
    return ('/Momentum' in name) or name.endswith('/Momentum')

def normalize_tf1_name(name: str) -> str:
    """
    Convert TF1 variable names to a canonical path comparable with TF2 names.
    Rules implemented from your dumps:
      - conv1/* and bn_conv1/* keep their simple scopes
      - resX..._branchY/* => resX... / resX..._branchY / resX..._branchY / <tensor>
      - bnX..._branchY/*  => resX... / resX..._branchY / bnX..._branchY / <tensor>
      - fc1_voc12_cN/*    => aspp / fc1_voc12_cN / <tensor>
    """
    name = _strip_tf_suffixes(name)

    # ASPP heads
    # ex: fc1_voc12_c0/weights -> aspp/fc1_voc12_c0/weights
    head, sep, tail = name.partition('/')
    m_aspp = _ASPP_RE.match(head)
    if m_aspp and sep:
        return f"aspp/{head}/{tail}"

    # Root conv / bn (names already match TF2)
    if name.startswith('conv1/'):
        return name
    if name.startswith('bn_conv1/'):
        return name

    # BatchNorm inside residual blocks: bn4b1_branch2a/gamma -> res4b1/res4b1_branch2a/bn4b1_branch2a/gamma
    m_bn = _BN_RE.match(head)
    if m_bn and sep:
        block_token = f"res{m_bn.group(1)}"       # e.g. 'res4b1'
        branch = f"{block_token}_branch{m_bn.group(2)}"  # e.g. 'res4b1_branch2a'
        return f"{block_token}/{branch}/{head}/{tail}"

    # Residual branch conv weights: res4b1_branch2a/weights -> res4b1/res4b1_branch2a/res4b1_branch2a/weights
    m_br = _BR_RE.match(head)
    if m_br and sep:
        block_token = m_br.group(1)               # e.g. 'res4b1'
        branch = f"{block_token}_branch{m_br.group(2)}"
        return f"{block_token}/{branch}/{branch}/{tail}"

    # As a conservative default, just return as-is (this covers resX_branch1 weights too).
    return name


# ---------------------------
# Name normalization (TF2)
# ---------------------------
def normalize_tf2_name(name: str) -> str:
    """
    Canonicalize TF2 variable names by stripping ':0'. We keep scopes intact because
    TF2 already uses the nested form we map TF1 to.
    """
    return _strip_tf_suffixes(name)


# ---------------------------
# Mapping and load logic
# ---------------------------
def build_normalized_maps(tf1_ckpt_map, tf2_var_list):
    """
    Build normalized dicts for TF1 and TF2:
      tf1_norm: {norm_name: (np_array, shape_tuple, original_name)}
      tf2_norm: {norm_name: (shape_tuple, var_obj, original_name)}
    If duplicates in TF1 (rare), last one wins (you can change to assert if desired).
    """
    tf1_norm = {}
    for k, (arr, shape) in tf1_ckpt_map.items():
        if should_skip_tf1_var(k):
            continue
        norm = normalize_tf1_name(k)
        tf1_norm[norm] = (arr, shape, k)

    tf2_norm = {}
    for name, shape, var in tf2_var_list:
        norm = normalize_tf2_name(name)
        tf2_norm[norm] = (shape, var, name)

    return tf1_norm, tf2_norm


def match_and_assign(tf1_norm, tf2_norm, strict_shape=True, assign=True):
    """
    Attempt to match by normalized name and (optionally) shape.
    Returns diagnostics and (optionally) assigns arrays to TF2 variables.
    """
    matched = []
    tf1_only = []
    tf2_only = []

    # First pass: exact name matches
    for n, (shape2, var2, orig2) in tf2_norm.items():
        if n in tf1_norm:
            arr1, shape1, orig1 = tf1_norm[n]
            shapes_ok = (tuple(shape1) == tuple(shape2))
            if not shapes_ok and strict_shape:
                tf2_only.append((orig2, shape2, "shape_mismatch_with_tf1", tf1_norm[n][1]))
                continue
            # Assign
            if assign and shapes_ok:
                var2.assign(arr1)
            matched.append((orig1, shape1, orig2, shape2))
        else:
            tf2_only.append((orig2, shape2, "no_tf1_match", None))

    # TF1 leftovers
    tf2_norm_names = set(tf2_norm.keys())
    for n, (arr1, shape1, orig1) in tf1_norm.items():
        if n not in tf2_norm_names:
            tf1_only.append((orig1, shape1, "no_tf2_match"))

    return matched, tf1_only, tf2_only


def pretty_report(matched, tf1_only, tf2_only, max_list=40):
    def _fmt_pairs(pairs, n=10):
        return "\n".join([f"  - {p}" for p in pairs[:n]]) + ("" if len(pairs) <= n else f"\n  ... and {len(pairs)-n} more")

    print("\n===== MAPPING REPORT =====")
    print(f"Matched assignments: {len(matched)}")
    if matched:
        print(_fmt_pairs([(m[0] + "  ->  " + m[2]) for m in matched], max_list))

    print(f"\nTF1-only (no TF2 match): {len(tf1_only)}")
    if tf1_only:
        print(_fmt_pairs([f"{n}  {s}  ({why})" for (n, s, why) in tf1_only], max_list))

    print(f"\nTF2-only (no TF1 match OR shape mismatch): {len(tf2_only)}")
    if tf2_only:
        def fmt(t):
            name2, shape2, why, extra = t
            if why == "shape_mismatch_with_tf1":
                return f"{name2}  {shape2}  (shape mismatch vs TF1 {extra})"
            return f"{name2}  {shape2}  ({why})"
        print(_fmt_pairs([fmt(t) for t in tf2_only], max_list))
    print("==========================\n")

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _safe_mkdir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_model(
    model: tf.keras.Model,
    output_dir: str = "out",
    model_name: str = "deeplab_v2",
    save_format: str = "savedmodel",     # "savedmodel" | "weights" | "ckpt"
    include_optimizer: bool = False,
    input_spec: tf.TensorSpec = tf.TensorSpec([None, None, None, 3], tf.float32, name="images"),
    metadata: dict | None = None,
):
    """
    Save a TF2 model three different ways with clean names:
      - savedmodel: portable, recommended for inference
      - weights:    Keras .h5 weights only (good for reloading the exact code)
      - ckpt:       TensorFlow checkpoint (subclass-friendly)

    Returns the final path written.
    """
    assert save_format in {"savedmodel", "weights", "ckpt"}
    stamp = _timestamp()
    base = f"{model_name}"
    root = _safe_mkdir(output_dir)

    # Save a tiny metadata.json next to the artifact for reproducibility
    meta = {
        "model_name": model_name,
        "saved_at": stamp,
        "save_format": save_format,
        "tensorflow": tf.__version__,
    }
    
    if metadata:
        meta.update(metadata)

    if save_format == "savedmodel":
        export_dir = os.path.join(root, base)  # directory
        tmpdir = tempfile.mkdtemp(prefix=base + "_tmp_", dir=root)
        # Trace a stable serving function
        @tf.function(input_signature=[input_spec])
        def serving_fn(x):
            y = model(x, training=False)
            return {"logits": y}
        tf.saved_model.save(model, tmpdir, signatures={"serving_default": serving_fn})
        # Atomically move into place
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        shutil.move(tmpdir, export_dir)
        with open(os.path.join(export_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"SavedModel exported to: {export_dir}")
        return export_dir

    if save_format == "weights":
        # Keras H5 weights file
        filepath = os.path.join(root, base + ".weights.h5")
        model.save_weights(filepath)
        with open(os.path.join(root, base + ".weights.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Weights saved to: {filepath}")
        return filepath

    if save_format == "ckpt":
        # Subclass-safe TF checkpoint (optionally include optimizer)
        ckpt_dir = os.path.join(root, base + "_ckpt")
        _safe_mkdir(ckpt_dir)
        objects = {"model": model}
        if include_optimizer and getattr(model, "optimizer", None) is not None:
            objects["optimizer"] = model.optimizer
        ckpt = tf.train.Checkpoint(**objects)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
        ckpt_path = manager.save()
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Checkpoint saved to: {ckpt_path}")
        return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="TF1 checkpoint prefix (no .index/.data suffix)")
    parser.add_argument("--save_to", default=None, help="Optional path to save TF2 weights (e.g., .h5)")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--height", type=int, default=513)
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--no_assign", action="store_true", help="Dry-run: do not assign weights")
    parser.add_argument("--non_strict_shapes", action="store_true", help="Allow shape mismatch (will NOT assign)")
    args = parser.parse_args()

    print("[1/5] Reading TF1 checkpoint...")
    tf1_tensors = load_tf1_tensors(args.ckpt)
    print(f"     Loaded {len(tf1_tensors)} tensors from TF1 ckpt")

    print("[2/5] Building TF2 model...")
    model = build_deeplabv2_tf2_model(
        num_classes=args.num_classes,
        input_shape=(args.height, args.width, 3)
    )

    print(f"[3/5] Extracting layer names from TF2 model...")
    tf2_vars = get_model_variables(model)
    print(f"     Found {len(tf2_vars)} variables in TF2 model")

    print("[4/5] Normalizing names...")
    tf1_norm, tf2_norm = build_normalized_maps(tf1_tensors, tf2_vars)
    print(f"      TF1 normalized: {len(tf1_norm)}  |  TF2 normalized: {len(tf2_norm)}")

    print("[5/5] Matching and (optionally) assigning...")
    matched, tf1_only, tf2_only = match_and_assign(
        tf1_norm,
        tf2_norm,
        strict_shape=not args.non_strict_shapes,
        assign=not args.no_assign
    )

    pretty_report(matched, tf1_only, tf2_only)

    if args.save_to and not args.no_assign:
        path = save_model(
                model,
                output_dir=args.save_to,              # e.g. "out/exports"
                model_name="deeplab_v2",      # nice, descriptive name
                save_format="ckpt",             # or "weights" or "ckpt"
                include_optimizer=False,              # True only if you need to resume training
                input_spec=tf.TensorSpec([None, 513, 513, 3], tf.float32, name="images"),
                metadata={
                    "notes": "",
                    "num_classes": 4,
                    "output_stride": 8
                }
            )
        print("Saved at:", path)


if __name__ == "__main__":
    main()

# ---------------------------

# python scripts/tf1_to_tf2_mapper.py --ckpt /home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/MODELS/TF1/HR/model.ckpt-1 --save_to /home/anish.tatke/blue-group/anish.tatke/IFTA_segmentation/MODELS/TF2/DeepLabV2-1/