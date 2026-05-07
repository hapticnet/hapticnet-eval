#!/usr/bin/env python3
"""
hpbr_to_usd.py
==============
Convert a .hpbr (haptic-PBR, new format) file to a USD stage.

USD structure
-------------
/HapticMaterial_{stem}          ← defaultPrim; overall haptic stats as custom attrs
  haptic:source_file        ← path to the source .hpbr file
  haptic:youngs_modulus     ← spatial mean across the whole surface
  haptic:poissons_ratio
  ...                       ← one attr per haptic map

  /Material                     ← UsdPreviewSurface with all PBR tile textures wired
  /PreviewMesh                  ← unit-plane mesh for in-viewer preview

  /Segments                     ← present only when a segmentation map is embedded
    haptic:classes              ← comma-separated list of all class names
    /{ClassName}                ← one prim per unique material class
      haptic:class_name
      haptic:pixel_fraction  ← fraction of surface covered by this class
      haptic:youngs_modulus  ← spatial mean for THIS class only
      ...

Textures are written to <output_dir>/textures/ and referenced with relative paths,
so the .usd file and its textures folder can be moved together freely.

Requirements
------------
  pxr  — install with:  pip install usd-core
         or use the NVIDIA OpenUSD distribution's bundled Python.
  No other external libraries required (struct, array, re, zlib are stdlib).

Usage
-----
  After pip install usd-core:
    python hpbr_to_usd.py --input_file file.hpbr --output_file file.usda

  With the NVIDIA OpenUSD distribution's bundled Python:
    Windows (Command Prompt):
      C:\path\to\OpenUSD\python\python.exe hpbr_to_usd.py --input_file file.hpbr --output_file file.usda

    Windows (PowerShell):
      & "C:\path\to\OpenUSD\python\python.exe" hpbr_to_usd.py --input_file file.hpbr --output_file file.usda

    Linux / macOS:
      /path/to/OpenUSD/python/python3 hpbr_to_usd.py --input_file file.hpbr --output_file file.usda
"""

import argparse
import array as _array
import os
import re
import struct
import sys
import zlib
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  USD environment bootstrap
#  Self-configures sys.path and (on Windows) the DLL search directories so
#  that `pxr` can be imported without any wrapper script or manual env setup.
#
#  Two discovery strategies are tried in order:
#   1. Script directory   — works when the script is placed inside the
#                           OpenUSD distribution root (classic usage).
#   2. Python executable  — works when the script lives anywhere but is
#                           invoked with the distribution's bundled Python
#                           (e.g. ..\..\OpenUSD\python\python.exe script.py).
#                           sys.executable is  <usd_root>/python/python.exe,
#                           so its grandparent is the distribution root.
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_usd_env() -> None:
    """Add OpenUSD lib/python and native DLL dirs to the search paths."""
    candidates = [
        Path(__file__).resolve().parent,             # script's own folder
        Path(sys.executable).resolve().parent.parent, # <usd_root>/python/python.exe
    ]

    for usd_root in candidates:
        # Detect a distribution root by the presence of lib/python (contains pxr)
        lib_python = usd_root / "lib" / "python"
        if not lib_python.is_dir():
            continue

        # Make `import pxr` work
        for p in (lib_python, usd_root / "pip-packages"):
            s = str(p)
            if p.is_dir() and s not in sys.path:
                sys.path.insert(0, s)

        # On Windows, register native DLL directories so the pxr extension
        # modules can locate their dependencies when first imported (Python 3.8+).
        if hasattr(os, "add_dll_directory"):
            for sub in ("lib", "bin", str(Path("plugin") / "usd")):
                d = usd_root / sub
                if d.is_dir():
                    os.add_dll_directory(str(d))

        return  # Found and configured a valid USD root — done


_bootstrap_usd_env()

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  hPBR constants
# ─────────────────────────────────────────────────────────────────────────────

_MAGIC   = bytes([0x68, 0x50, 0x42, 0x52])   # b"hPBR"
_VERSION = 2

# Standard haptic properties — displayed in this order in usdview
_PROPERTY_DEFS = [
    ("youngs_modulus",       "Young's Modulus [MPa]"),
    ("poissons_ratio",       "Poisson's Ratio"),
    ("roughness_average_ra", "Roughness Average (Ra) [um]"),
    ("static_friction",      "Static Friction Coefficient"),
    ("kinetic_friction",     "Kinetic Friction Coefficient"),
    ("thermal_conductivity", "Thermal Conductivity [W/(mK)]"),
    ("thermal_effusivity",   "Thermal Effusivity [Ws^0.5/(m^2K)]"),
    ("haptic_tensor",               "Haptic Tensor"),
]
_PROP_LABEL  = {s: l for s, l in _PROPERTY_DEFS}
_PROP_ORDER  = [s for s, _ in _PROPERTY_DEFS]

# PBR tile stem → (UsdPreviewSurface input, colorspace, is_normal_map)
_TILE_CONFIG = {
    "basecolor":    ("diffuseColor",  "sRGB", False),
    "albedo":       ("diffuseColor",  "sRGB", False),
    "diffuse":      ("diffuseColor",  "sRGB", False),
    "roughness":    ("roughness",     "raw",  False),
    "metallic":     ("metallic",      "raw",  False),
    "metalness":    ("metallic",      "raw",  False),
    "normal":       ("normal",        "raw",  True),
    "displacement": ("displacement",  "raw",  False),
    "height":       ("displacement",  "raw",  False),
    "opacity":      ("opacity",       "raw",  False),
    "alpha":        ("opacity",       "raw",  False),
    "specular":     ("specularColor", "raw",  False),
    "emission":     ("emissiveColor", "sRGB", False),
    "occlusion":    ("occlusion",     "raw",  False),
    "ao":           ("occlusion",     "raw",  False),
}

_TILE_PRIORITY = [
    "basecolor", "albedo", "diffuse",
    "roughness", "metallic", "metalness",
    "normal", "displacement", "height",
    "opacity", "alpha", "specular",
    "emission", "occlusion", "ao",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Minimal .npy parser  (stdlib only — no numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _npy_header(data: bytes):
    """
    Parse the .npy magic + version + header dict.
    Returns (header_str, data_offset).
    """
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Not a .npy file (bad magic)")
    major = data[6]
    if major == 1:
        hlen, hstart = int.from_bytes(data[8:10], "little"), 10
    elif major == 2:
        hlen, hstart = int.from_bytes(data[8:12], "little"), 12
    else:
        raise ValueError(f"Unsupported .npy version {major}")
    return data[hstart:hstart + hlen].decode("ascii").strip(), hstart + hlen


def _parse_npy_numeric(data: bytes):
    """
    Parse a numeric .npy file without numpy.
    Returns (rows, cols, stride, flat_list) where stride = values-per-pixel.
    Note: assumes little-endian byte order (standard for hPBR files on x86).
    """
    header, offset = _npy_header(data)  # raises ValueError if bad magic

    # Shape
    m = re.search(r"'shape'\s*:\s*\(([^)]*)\)", header)
    if not m:
        raise ValueError("Cannot parse .npy shape")
    dims = [int(p.strip()) for p in m.group(1).split(",") if p.strip()]
    rows   = dims[0] if dims else 1
    cols   = dims[1] if len(dims) > 1 else 1
    stride = 1
    for d in dims[2:]:
        stride *= d

    # dtype
    dm = re.search(r"'descr'\s*:\s*'([^']+)'", header)
    if not dm:
        raise ValueError("Cannot parse .npy descr")
    descr = dm.group(1).lstrip("<>=|").lower()

    total = rows * cols * stride
    raw   = data[offset:]

    dtype_map = {
        "f4": ("f", 4), "f8": ("d", 8),
        "i1": ("b", 1), "u1": ("B", 1),
        "i2": ("h", 2), "u2": ("H", 2),
        "i4": ("i", 4), "u4": ("I", 4),
        "i8": ("q", 8), "u8": ("Q", 8),
    }
    if descr not in dtype_map:
        raise ValueError(f"Unsupported .npy dtype '{descr}'")

    typecode, itemsize = dtype_map[descr]
    a = _array.array(typecode)
    a.frombytes(raw[:total * itemsize])
    return rows, cols, stride, list(a)


def _parse_narr_old(payload: bytes):
    """
    Parse an OLD-format NARR payload (pre-new-format hPBR).
    Layout: [rows:4B LE][cols:4B LE][dtype_code:1B][raw C-order bytes]
    dtype codes: 1=float32, 2=float64, 3=int32, 4=int64
    Returns (rows, cols, stride=1, flat_list).
    """
    rows, cols, code = struct.unpack("<IIb", payload[:9])
    dtype_map_old = {1: ("f", 4), 2: ("d", 8), 3: ("i", 4), 4: ("q", 8)}
    if code not in dtype_map_old:
        raise ValueError(f"Unknown old-format NARR dtype code {code}")
    typecode, itemsize = dtype_map_old[code]
    total = rows * cols
    a = _array.array(typecode)
    a.frombytes(payload[9:9 + total * itemsize])
    return rows, cols, 1, list(a)


# ── Pickle stream parser for numpy object arrays (segmentation) ───────────────

# Strings that are numpy / pickle internals, not material class names.
_SEG_SKIP = frozenset({
    "numpy", "ndarray", "dtype", "reconstruct", "multiarray",
    "b", "r", "c", "f", "version", "typecode", "shape", "rawdata",
    "data", "strides", "descr", "order", "is_f_order", "allow_pickle",
    "umath", "core",
    "O8", "O4", "U", "S", "V",
    "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
    "f4", "f8", "f2", "c8", "c16", "m8", "M8",
})


def _is_class_name(s) -> bool:
    if not s or len(s) < 2:
        return False
    if "." in s or s.startswith("_"):
        return False
    return s.lower() not in {x.lower() for x in _SEG_SKIP}


def _parse_pickle_strings(data: bytes, start: int, total: int) -> list:
    """
    Extract up to `total` material class-name strings from a pickle stream.
    Ported directly from HapticNetSegReader.cs — handles SHORT_BINUNICODE,
    BINUNICODE, MEMOIZE, BINPUT/GET, LONG_BINPUT/GET and all framing opcodes.
    """
    MEMO_SIZE = 1024
    memo       = [None] * MEMO_SIZE
    memo_ctr   = 0
    last_str   = None
    last_is_str = False
    flat       = []

    i, end = start, len(data)

    while i < end and len(flat) < total:
        op = data[i]; i += 1

        if op == 0x80:    # PROTO — skip version byte
            i += 1
        elif op == 0x95:  # FRAME — skip 8-byte length
            i += 8

        # ── String literals ───────────────────────────────────────────────
        elif op == 0x8C:  # SHORT_BINUNICODE (1-byte length)
            n = data[i]; i += 1
            last_str = data[i:i+n].decode("utf-8", errors="replace"); i += n
            last_is_str = True
        elif op == 0x58:  # BINUNICODE (4-byte length)
            n = int.from_bytes(data[i:i+4], "little"); i += 4
            if i + n > end: break
            last_str = data[i:i+n].decode("utf-8", errors="replace"); i += n
            last_is_str = True
        elif op == 0x55:  # SHORT_BINSTRING (latin-1, protocol 2)
            n = data[i]; i += 1
            last_str = data[i:i+n].decode("latin-1"); i += n
            last_is_str = True

        # ── Memo store ────────────────────────────────────────────────────
        elif op == 0x94:  # MEMOIZE — sequential id
            v = last_str if last_is_str and _is_class_name(last_str) else None
            if memo_ctr < MEMO_SIZE: memo[memo_ctr] = v
            memo_ctr += 1
            if v: flat.append(v)
            last_is_str = False
        elif op == 0x71:  # BINPUT — 1-byte id
            id_ = data[i]; i += 1
            v = last_str if last_is_str and _is_class_name(last_str) else None
            if id_ < MEMO_SIZE: memo[id_] = v
            if v: flat.append(v)
            last_is_str = False
        elif op == 0x72:  # LONG_BINPUT — 4-byte id
            id_ = int.from_bytes(data[i:i+4], "little"); i += 4
            v = last_str if last_is_str and _is_class_name(last_str) else None
            if 0 <= id_ < MEMO_SIZE: memo[id_] = v
            if v: flat.append(v)
            last_is_str = False

        # ── Memo recall ───────────────────────────────────────────────────
        elif op == 0x68:  # BINGET — 1-byte id
            id_ = data[i]; i += 1
            v = memo[id_] if id_ < MEMO_SIZE else None
            if v: flat.append(v)
            last_str = v; last_is_str = bool(v)
        elif op == 0x6A:  # LONG_BINGET — 4-byte id
            id_ = int.from_bytes(data[i:i+4], "little"); i += 4
            v = memo[id_] if 0 <= id_ < MEMO_SIZE else None
            if v: flat.append(v)
            last_str = v; last_is_str = bool(v)

        elif op == 0x2E:  # STOP
            break

        # ── Stack-changing ops — clear lastStr flag ───────────────────────
        elif op in (0x93, 0x81, 0x52, 0x62, 0x85, 0x86, 0x87, 0x74,
                    0x65, 0x75, 0x7D, 0x64, 0x28, 0x61, 0x92):
            last_is_str = False

        # ── Skip fixed-width payloads ─────────────────────────────────────
        elif op == 0x4B:  # BININT1
            i += 1;     last_is_str = False
        elif op == 0x4D:  # BININT2
            i += 2;     last_is_str = False
        elif op == 0x4A:  # BININT
            i += 4;     last_is_str = False
        elif op == 0x43:  # SHORT_BINBYTES
            n = data[i]; i += 1 + n; last_is_str = False
        elif op == 0x42:  # BINBYTES
            n = int.from_bytes(data[i:i+4], "little"); i += 4 + n; last_is_str = False
        elif op == 0x63:  # GLOBAL — two newline-terminated strings
            while i < end and data[i] != ord("\n"): i += 1; i += 1
            while i < end and data[i] != ord("\n"): i += 1; i += 1
            last_is_str = False
        elif op == 0x8A:  # LONG1
            n = data[i]; i += 1 + n; last_is_str = False
        elif op == 0x4C:  # INT (text)
            while i < end and data[i] != ord("\n"): i += 1; i += 1
            last_is_str = False
        else:
            last_is_str = False

    return flat


def _parse_npy_object(data: bytes):
    """
    Parse a .npy file containing an object array of strings (pickle stream).
    Returns (rows, cols, flat_strings).
    """
    header, offset = _npy_header(data)

    m = re.search(r"'shape'\s*:\s*\(([^)]*)\)", header)
    if not m:
        raise ValueError("Cannot parse .npy shape")
    dims = [int(p.strip()) for p in m.group(1).split(",") if p.strip()]
    rows, cols = (dims[0] if dims else 1), (dims[1] if len(dims) > 1 else 1)

    flat = _parse_pickle_strings(data, offset, rows * cols)
    if len(flat) < rows * cols:
        print(f"  [warn] Segmentation: got {len(flat)} class entries, "
              f"expected {rows * cols} — partial map loaded")
    return rows, cols, flat


# ─────────────────────────────────────────────────────────────────────────────
#  hPBR v2 (new format) chunk reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_hpbr(path: str) -> dict:
    """
    Parse a .hpbr v2 (new format) file.

    Returns:
        {
          "material_props": {stem: (rows, cols, stride, flat_floats)},
          "pbr_tiles":      {tile_name: bytes (raw PNG)},
          "seg_data":       (rows, cols, flat_strings) | None,
        }
    """
    result = {"material_props": {}, "pbr_tiles": {}, "seg_data": None}
    fsize = os.path.getsize(path)

    with open(path, "rb") as fh:
        if fh.read(4) != _MAGIC or fh.read(1) != bytes([_VERSION]):
            raise ValueError(f"Not a valid hPBR v2 file: {path}")

        for tag, name, payload in _iter_chunks(fh, fsize):
            if tag == b"IEND":
                break

            if tag == b"NARR":
                if len(payload) < 9:
                    print(f"  [warn] NARR '{name}': payload too short, skipping")
                    continue
                _rows, _cols, pflag = struct.unpack("<IIb", payload[:9])
                npy_bytes = payload[9:]
                # Detect format: new format starts with \x93NUMPY magic;
                # old format has raw C-order bytes at offset 9.
                is_new_fmt = npy_bytes[:6] == b"\x93NUMPY"
                try:
                    if is_new_fmt:
                        if pflag:
                            result["seg_data"] = _parse_npy_object(npy_bytes)
                        else:
                            result["material_props"][name] = _parse_npy_numeric(npy_bytes)
                    else:
                        # Old format: dtype code (1–4) + raw C-order bytes
                        result["material_props"][name] = _parse_narr_old(payload)
                except Exception as exc:
                    print(f"  [warn] NARR '{name}' skipped: {exc}")

            elif tag == b"IMAG":
                if payload[0] == 1:   # PNG
                    result["pbr_tiles"][name] = bytes(payload[1:])
                else:
                    print(f"  [warn] IMAG '{name}': unknown format {payload[0]}, skipping")

    return result


def _iter_chunks(fh, fsize):
    while fh.tell() < fsize:
        raw_tag = fh.read(4)
        if len(raw_tag) < 4:
            break
        chunk_len     = struct.unpack("<I", fh.read(4))[0]
        name_len_byte = fh.read(1)
        if not name_len_byte:
            break
        name_len   = struct.unpack("B", name_len_byte)[0]
        name_bytes = fh.read(name_len)
        name       = name_bytes.decode("utf-8")
        payload    = fh.read(chunk_len)
        crc_stored = struct.unpack("<I", fh.read(4))[0]
        crc_data   = raw_tag + name_len_byte + name_bytes + payload
        if (zlib.crc32(crc_data) & 0xFFFFFFFF) != crc_stored:
            raise ValueError(f"CRC mismatch for chunk '{name}'")
        yield raw_tag, name, payload


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics  (pure Python, no numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _finite_mean(vals) -> float:
    """NaN-safe mean of a sequence of floats."""
    finite = [v for v in vals if v == v]   # NaN != NaN
    return sum(finite) / len(finite) if finite else float("nan")


def _build_class_index(seg_flat: list) -> dict:
    """
    Return {class_name: [pixel_indices]} from a flat segmentation list.
    Built once and reused for every property × class combination.
    """
    idx: dict = {}
    for i, cls in enumerate(seg_flat):
        if cls and isinstance(cls, str):
            idx.setdefault(cls, []).append(i)
    return idx


def _prop_mean(prop_tuple, indices=None) -> float:
    """
    Compute the spatial mean of a haptic property array.
    If *indices* (pixel indices list) is given, restrict to those pixels.
    Handles multi-channel properties (stride > 1) by averaging all channels.
    """
    rows, cols, stride, flat = prop_tuple
    n_pixels = rows * cols

    if indices is not None:
        vals = [flat[i * stride + k]
                for i in indices if i < n_pixels
                for k in range(stride)]
    else:
        vals = list(flat)

    return _finite_mean(vals)


# ─────────────────────────────────────────────────────────────────────────────
#  USD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prim_name(s: str) -> str:
    """Convert an arbitrary string to a valid USD prim name."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s or "unnamed"


def _set_haptic_attrs(prim: Usd.Prim, props: dict, indices=None):
    """
    Write haptic property means as custom ``haptic:`` attributes on *prim*.
    Known properties are written in _PROPERTY_DEFS order; extras follow.
    *indices* restricts sampling to the given pixel indices (per-class stats).
    """
    written = set()

    def _write(stem, prop_tuple):
        val = _prop_mean(prop_tuple, indices)
        if val != val:    # NaN — nothing meaningful to show
            return
        attr_name = stem
        label = _PROP_LABEL.get(stem, attr_name)
        qualifier = "" if indices is None else "  (class mean)"
        attr = prim.CreateAttribute(
            f"haptic:{attr_name}", Sdf.ValueTypeNames.Float, custom=True
        )
        attr.Set(float(val))
        attr.SetDocumentation(f"{label}{qualifier}")
        written.add(stem)

    # Known properties in defined order
    for stem in _PROP_ORDER:
        if stem in props:
            _write(stem, props[stem])
    # Any extra properties not in the standard list
    for stem, pt in props.items():
        if stem not in written:
            _write(stem, pt)


def _build_material(stage: Usd.Stage, mat_path: Sdf.Path,
                    pbr_tiles: dict, tex_rel_dir: str) -> UsdShade.Material:
    """Define a UsdPreviewSurface material network at *mat_path*."""
    material = UsdShade.Material.Define(stage, mat_path)

    # UV primvar reader
    uv = UsdShade.Shader.Define(stage, mat_path.AppendChild("TexCoordReader"))
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # Surface shader
    sh = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(
        sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    _COLOR3F = {"diffuseColor", "emissiveColor", "specularColor"}
    filled   = set()

    def _sort_key(k):
        try:    return _TILE_PRIORITY.index(k.lower())
        except: return len(_TILE_PRIORITY)

    for tile_name in sorted(pbr_tiles, key=_sort_key):
        cfg = _TILE_CONFIG.get(tile_name.lower())
        if not cfg:
            print(f"  [info] Tile '{tile_name}' not in config — included as texture asset only")
            continue
        usd_input, colorspace, is_normal = cfg
        if usd_input in filled:
            continue

        node_name = re.sub(r"[^a-zA-Z0-9_]", "_", tile_name) + "Tex"
        tex = UsdShade.Shader.Define(stage, mat_path.AppendChild(node_name))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"{tex_rel_dir}/{tile_name}.png")
        )
        tex.CreateInput("wrapS",            Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT",            Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        tex.CreateInput("st",               Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)

        if is_normal:
            # Remap stored [0,1] to expected [-1,1] tangent-space range
            tex.CreateInput("bias",  Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f( 2,  2,  2, 1))
            sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            )
        elif usd_input in _COLOR3F:
            sh.CreateInput(usd_input, Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            )
        else:
            sh.CreateInput(usd_input, Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
            )

        filled.add(usd_input)

    return material


def _build_preview_mesh(stage: Usd.Stage, mesh_path: Sdf.Path,
                        material: UsdShade.Material) -> UsdGeom.Mesh:
    """Define a unit-plane mesh at *mesh_path* and bind the material to it.

    The quad stands upright in the XY plane facing +Z so it is front-facing
    to usdview's default camera (which looks along -Z).
    """
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-0.5, -0.5, 0), Gf.Vec3f(0.5, -0.5, 0),
        Gf.Vec3f(0.5,   0.5, 0), Gf.Vec3f(-0.5, 0.5, 0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation("vertex")

    # CreatePrimvar was moved from UsdGeom.Gprim to UsdGeom.PrimvarsAPI in
    # newer USD distributions; use the API object for forward compatibility.
    pvars = UsdGeom.PrimvarsAPI(mesh)
    st = pvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    # Apply() registers the MaterialBindingAPI schema on the prim
    # (adds it to apiSchemas); without this, Storm ignores the binding.
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
#  Main converter
# ─────────────────────────────────────────────────────────────────────────────

def convert(input_file: str, output_file: str):
    in_path  = Path(input_file).resolve()
    out_path = Path(output_file).resolve()
    stem     = in_path.stem

    # ── Read source file ──────────────────────────────────────────────────────
    print(f"[hpbr->usd] Reading  {in_path.name} ...")
    hpbr     = _read_hpbr(str(in_path))
    props    = hpbr["material_props"]
    tiles    = hpbr["pbr_tiles"]
    seg_data = hpbr["seg_data"]   # (rows, cols, flat_strings) | None

    n_classes = len(set(seg_data[2])) if seg_data else 0
    print(f"  PBR tiles   : {list(tiles) or '(none)'}")
    print(f"  Haptic maps : {list(props) or '(none)'}")
    print(f"  Segmentation: "
          + (f"{n_classes} unique class(es)" if seg_data else "not present"))

    # ── Write textures ────────────────────────────────────────────────────────
    tex_dir = out_path.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    for tile_name, png_bytes in tiles.items():
        (tex_dir / f"{tile_name}.png").write_bytes(png_bytes)
    if tiles:
        print(f"  Textures    : written to {tex_dir}")

    # ── Build USD stage ───────────────────────────────────────────────────────
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    root_path  = Sdf.Path(f"/HapticMaterial_{_prim_name(stem)}")
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root_xform.GetPrim())
    root_prim = root_xform.GetPrim()

    # ── Root prim: metadata + overall haptic stats ────────────────────────────
    root_prim.SetDocumentation(
        f"Haptic PBR material converted from {in_path.name}.\n"
        f"Tiles: {', '.join(tiles) or 'none'}.\n"
        f"Haptic maps: {', '.join(props) or 'none'}."
    )
    root_prim.CreateAttribute(
        "haptic:source_file", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(in_path))

    if props:
        _set_haptic_attrs(root_prim, props, indices=None)

    # ── Material ──────────────────────────────────────────────────────────────
    material = _build_material(stage, root_path.AppendChild("Material"),
                               tiles, "./textures")

    # ── Preview mesh ──────────────────────────────────────────────────────────
    _build_preview_mesh(stage, root_path.AppendChild("PreviewMesh"), material)

    # ── Segmentation class prims ──────────────────────────────────────────────
    if seg_data is not None:
        seg_rows, seg_cols, seg_flat = seg_data
        unique_classes = sorted(
            cls for cls in set(seg_flat) if cls and isinstance(cls, str)
        )

        if unique_classes:
            # Pre-build index dict once — reused for every class × property
            class_index = _build_class_index(seg_flat)
            total_pixels = seg_rows * seg_cols

            seg_scope_path = root_path.AppendChild("Segments")
            seg_scope = stage.DefinePrim(seg_scope_path, "Scope")
            seg_scope.SetDocumentation(
                f"{len(unique_classes)} segmentation class(es): "
                + ", ".join(unique_classes)
            )
            seg_scope.CreateAttribute(
                "haptic:classes", Sdf.ValueTypeNames.String, custom=True
            ).Set(", ".join(unique_classes))

            # Warn once if shapes differ (per-class stats will be approximate)
            if props:
                first_prop = next(iter(props.values()))
                prop_pixels = first_prop[0] * first_prop[1]
                if prop_pixels != total_pixels:
                    print(f"  [warn] Segmentation shape ({seg_rows}×{seg_cols}) differs "
                          f"from property shape — per-class stats may be approximate")

            for cls in unique_classes:
                cls_path = seg_scope_path.AppendChild(_prim_name(cls))
                cls_prim = stage.DefinePrim(cls_path, "Scope")

                indices        = class_index.get(cls, [])
                pixel_fraction = len(indices) / total_pixels if total_pixels else 0.0

                cls_prim.SetDocumentation(
                    f"Class '{cls}'  —  {pixel_fraction * 100:.1f}% of surface"
                )
                cls_prim.CreateAttribute(
                    "haptic:class_name", Sdf.ValueTypeNames.String, custom=True
                ).Set(cls)
                cls_prim.CreateAttribute(
                    "haptic:pixel_fraction", Sdf.ValueTypeNames.Float, custom=True
                ).Set(float(pixel_fraction))

                if props:
                    _set_haptic_attrs(cls_prim, props, indices=indices)

    # ── Save ──────────────────────────────────────────────────────────────────
    stage.Save()
    print(f"[hpbr->usd] Written  -> {out_path}")
    if seg_data and unique_classes:
        print(f"           Segments -> {root_path}/Segments/{{class_name}}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a .hpbr (haptic-PBR) file to a USD stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",  required=True, metavar="FILE",
        help="Path to the source .hpbr file.",
    )
    parser.add_argument(
        "--output_file", required=True, metavar="FILE",
        help="Path for the output .usd file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")

    convert(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
