bl_info = {
    "name":        "hPBR Importer",
    "version":     (2, 0),
    "blender":     (4, 0, 0),
    "category":    "Material",
    "description": (
        "Import .hpbr haptic-PBR files (new format): auto-wires all PBR textures to "
        "Principled BSDF and caches haptic property maps + segmentation on the active object"
    ),
}

import bpy
import struct
import zlib
import numpy as np
import tempfile
import os
from io import BytesIO
from bpy.props import BoolProperty, IntProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

# ══════════════════════════════════════════════════════════════════════════════
#  Public haptic-data cache
#  {object_name: {"props": {prop_name: np.ndarray}, "seg": np.ndarray | None}}
#
#  hover_tooltip.py (or any other script) can read this after loading:
#      import hpbr_importer
#      props = hpbr_importer.get_haptic_data("MyObject")  # -> dict | None
#      seg   = hpbr_importer.get_seg_data("MyObject")     # -> ndarray | None
# ══════════════════════════════════════════════════════════════════════════════
_haptic_cache: dict = {}


def get_haptic_data(obj_name: str):
    """Return haptic property dict for *obj_name*, or None if not yet loaded."""
    entry = _haptic_cache.get(obj_name)
    return entry["props"] if entry else None


def get_seg_data(obj_name: str):
    """Return the HapticNetSegmentation array for *obj_name*, or None."""
    entry = _haptic_cache.get(obj_name)
    return entry["seg"] if entry else None


# ══════════════════════════════════════════════════════════════════════════════
#  hPBR v2 reader  (new NARR format — numpy .npy bytes + pickle_flag)
# ══════════════════════════════════════════════════════════════════════════════
_MAGIC   = bytes([0x68, 0x50, 0x42, 0x52])   # b"hPBR"
_VERSION = 2


class _HpbrReader:
    """Parse a .hpbr v2 (new format) binary file into numpy arrays and raw PNG bytes."""

    def __init__(self):
        self.material_props: dict      = {}    # prop_name → np.ndarray  (haptic maps)
        self.pbr_tiles:      dict      = {}    # tile_name → bytes (raw PNG)
        self.seg_arr:        object    = None  # (H, W) object array of class-name strings

    # ── Public entry-point ────────────────────────────────────────────────────
    def read(self, filepath: str):
        filesize = os.path.getsize(filepath)
        with open(filepath, "rb") as fh:
            magic = fh.read(4)
            ver   = fh.read(1)
            if magic != _MAGIC or ver != bytes([_VERSION]):
                raise ValueError(f"Not a valid hPBR v2 file: {filepath}")

            for tag, name, payload in self._iter_chunks(fh, filesize):
                if tag == b"IEND":
                    break
                elif tag == b"NARR":
                    try:
                        arr = _parse_narr(payload)
                    except Exception as exc:
                        print(f"[hPBR] Could not decode NARR '{name}': {exc}")
                        continue
                    if name.endswith("_HapticNetSegmentation"):
                        self.seg_arr = arr
                    else:
                        self.material_props[name] = arr
                elif tag == b"IMAG":
                    fmt = payload[0]
                    if fmt == 1:   # PNG
                        self.pbr_tiles[name] = bytes(payload[1:])
                    else:
                        print(f"[hPBR] Unknown image format {fmt} for '{name}', skipping")

    # ── Chunk iterator with CRC verification ─────────────────────────────────
    @staticmethod
    def _iter_chunks(fh, filesize):
        while fh.tell() < filesize:
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

            crc_computed = (
                zlib.crc32(raw_tag + name_len_byte + name_bytes + payload) & 0xFFFFFFFF
            )
            if crc_stored != crc_computed:
                raise ValueError(
                    f"CRC mismatch for chunk '{name}': "
                    f"expected {crc_stored:#010x}, got {crc_computed:#010x}"
                )
            yield (raw_tag, name, payload)


def _parse_narr(payload: bytes) -> np.ndarray:
    """
    Decode a NARR payload (new format) back into a numpy array.

    Header: rows(4B) + cols(4B) + pickle_flag(1B)
    Body:   numpy .npy format bytes (via np.save / allow_pickle)

    pickle_flag 0 = numeric array, 1 = object array (e.g. string labels).
    """
    _rows, _cols, pickle_flag = struct.unpack("<IIb", payload[:9])
    return np.load(BytesIO(payload[9:]), allow_pickle=bool(pickle_flag))


# ══════════════════════════════════════════════════════════════════════════════
#  Blender image creation from raw PNG bytes
# ══════════════════════════════════════════════════════════════════════════════
def _load_png_bytes_as_image(png_bytes: bytes, img_name: str):
    """
    Write *png_bytes* to a temp file, load it as a Blender Image, pack the
    data into memory (so the temp file can be removed), and return the Image.
    """
    existing = bpy.data.images.get(img_name)
    if existing is not None:
        bpy.data.images.remove(existing)

    tmp_path = os.path.join(tempfile.gettempdir(), img_name + ".png")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(png_bytes)
        img = bpy.data.images.load(tmp_path, check_existing=False)
        img.name     = img_name
        img.pack()
        img.filepath = ""
        return img
    except Exception as exc:
        print(f"[hPBR] Could not create Blender image '{img_name}': {exc}")
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Node-graph / material builder
# ══════════════════════════════════════════════════════════════════════════════

_TILE_CONFIG = {
    "basecolor":    ("Base Color",          "sRGB"),
    "albedo":       ("Base Color",          "sRGB"),
    "diffuse":      ("Base Color",          "sRGB"),
    "roughness":    ("Roughness",           "Non-Color"),
    "metallic":     ("Metallic",            "Non-Color"),
    "metalness":    ("Metallic",            "Non-Color"),
    "normal":       ("NORMAL_MAP",          "Non-Color"),
    "displacement": ("DISPLACEMENT",        "Non-Color"),
    "height":       ("DISPLACEMENT",        "Non-Color"),
    "opacity":      ("Alpha",               "Non-Color"),
    "alpha":        ("Alpha",               "Non-Color"),
    "specular":     ("Specular IOR Level",  "Non-Color"),
    "emission":     ("Emission Color",      "sRGB"),
}

_TILE_PRIORITY = [
    "basecolor", "albedo", "diffuse",
    "roughness",
    "metallic", "metalness",
    "normal",
    "displacement", "height",
    "opacity", "alpha",
    "specular",
    "emission",
]

_X_TC   = -1400
_X_MAP  = -1100
_X_TEX  =  -700
_X_MID  =  -300
_X_BSDF =   200
_X_OUT  =   600

_Y_START =  400
_Y_STEP  = -300


def _apply_material(
    obj,
    reader: _HpbrReader,
    hpbr_path: str,
    add_subdivision: bool   = True,
    subdivision_levels: int  = 4,
    displacement_scale: float = 0.05,
):
    """Build a Principled-BSDF node graph from *reader* and assign it to *obj*."""
    stem     = os.path.splitext(os.path.basename(hpbr_path))[0]
    mat_name = f"hPBR.{stem}"
    img_pfx  = f"hpbr.{stem}"

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tc_node   = nodes.new("ShaderNodeTexCoord")
    map_node  = nodes.new("ShaderNodeMapping")
    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    out_node  = nodes.new("ShaderNodeOutputMaterial")

    tc_node.location   = (_X_TC,   300)
    map_node.location  = (_X_MAP,  300)
    bsdf_node.location = (_X_BSDF, 300)
    out_node.location  = (_X_OUT,  300)

    links.new(tc_node.outputs["UV"],      map_node.inputs["Vector"])
    links.new(bsdf_node.outputs["BSDF"],  out_node.inputs["Surface"])

    def _sort_key(name):
        try:
            return _TILE_PRIORITY.index(name.lower())
        except ValueError:
            return len(_TILE_PRIORITY)

    filled        = set()
    y             = _Y_START
    disp_tex_node = None
    disp_tex_y    = 0

    for tile_name in sorted(reader.pbr_tiles.keys(), key=_sort_key):
        cfg = _TILE_CONFIG.get(tile_name.lower())
        if cfg is None:
            print(f"[hPBR] Unrecognised tile '{tile_name}', skipping")
            continue
        target, colorspace = cfg

        if target in filled:
            continue

        img_name = f"{img_pfx}.{tile_name}"
        img = _load_png_bytes_as_image(reader.pbr_tiles[tile_name], img_name)
        if img is None:
            continue
        try:
            img.colorspace_settings.name = colorspace
        except TypeError:
            pass

        tex          = nodes.new("ShaderNodeTexImage")
        tex.image    = img
        tex.label    = tile_name
        tex.location = (_X_TEX, y)
        links.new(map_node.outputs["Vector"], tex.inputs["Vector"])

        if target == "NORMAL_MAP":
            nm          = nodes.new("ShaderNodeNormalMap")
            nm.location = (_X_MID, y)
            links.new(tex.outputs["Color"], nm.inputs["Color"])
            if "Normal" in bsdf_node.inputs:
                links.new(nm.outputs["Normal"], bsdf_node.inputs["Normal"])

        elif target == "DISPLACEMENT":
            disp_tex_node = tex
            disp_tex_y    = y

        else:
            if target in bsdf_node.inputs:
                links.new(tex.outputs["Color"], bsdf_node.inputs[target])
            else:
                print(
                    f"[hPBR] Principled BSDF has no input '{target}' "
                    f"(tile '{tile_name}'). Blender version mismatch?"
                )

        filled.add(target)
        y += _Y_STEP

    if disp_tex_node is not None:
        disp_node          = nodes.new("ShaderNodeDisplacement")
        disp_node.location = (_X_MID, disp_tex_y)
        disp_node.inputs["Scale"].default_value = displacement_scale
        links.new(disp_tex_node.outputs["Color"],    disp_node.inputs["Height"])
        links.new(disp_node.outputs["Displacement"], out_node.inputs["Displacement"])
        try:
            mat.cycles.displacement_method = "BOTH"
        except AttributeError:
            pass

    if not obj.material_slots:
        obj.data.materials.append(mat)
    else:
        obj.material_slots[0].material = mat

    if add_subdivision and disp_tex_node is not None:
        if not any(m.type == "SUBSURF" for m in obj.modifiers):
            sub               = obj.modifiers.new(name="hPBR_Subdivision", type="SUBSURF")
            sub.levels        = subdivision_levels
            sub.render_levels = subdivision_levels

    obj["hpbr_path"] = hpbr_path

    return mat


# ══════════════════════════════════════════════════════════════════════════════
#  Operators
# ══════════════════════════════════════════════════════════════════════════════

class HPBR_OT_import(bpy.types.Operator, ImportHelper):
    """Open a .hpbr file and apply its PBR textures + haptic maps to the active object"""
    bl_idname  = "hpbr.import_material"
    bl_label   = "Import hPBR …"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".hpbr"
    filter_glob: StringProperty(default="*.hpbr;*.HPBR", options={"HIDDEN"})

    add_subdivision: BoolProperty(
        name        = "Add Subdivision",
        description = "Add a Subdivision Surface modifier so the displacement map creates real geometry",
        default     = True,
    )
    subdivision_levels: IntProperty(
        name        = "Subdivision Levels",
        description = "Viewport subdivision level (higher = smoother displacement, slower)",
        default     = 4, min = 1, max = 8,
    )
    displacement_scale: FloatProperty(
        name        = "Displacement Scale",
        description = "Multiplier applied to the height / displacement map",
        default     = 0.05, min = 0.0, soft_max = 1.0, step = 1,
    )

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type in {"MESH", "CURVE", "SURFACE"}

    def execute(self, context):
        obj  = context.object
        path = self.filepath

        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        reader = _HpbrReader()
        try:
            reader.read(path)
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to read hPBR file: {exc}")
            return {"CANCELLED"}

        _haptic_cache[obj.name] = {
            "props": reader.material_props,
            "seg":   reader.seg_arr,
        }

        seg_info = " + segmentation" if reader.seg_arr is not None else ""
        self.report(
            {"INFO"},
            f"Parsed '{os.path.basename(path)}': "
            f"{len(reader.pbr_tiles)} texture(s), "
            f"{len(reader.material_props)} haptic map(s){seg_info}",
        )

        has_disp = any(
            _TILE_CONFIG.get(n.lower(), (None,))[0] == "DISPLACEMENT"
            for n in reader.pbr_tiles
        )
        if has_disp and context.scene.render.engine != "CYCLES":
            self.report(
                {"WARNING"},
                "Displacement maps require Cycles. "
                "Switch to Cycles (Properties → Render → Render Engine) to see geometry detail.",
            )

        try:
            _apply_material(
                obj, reader, path,
                add_subdivision    = self.add_subdivision,
                subdivision_levels = self.subdivision_levels,
                displacement_scale = self.displacement_scale,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Material setup failed: {exc}")
            import traceback; traceback.print_exc()
            return {"CANCELLED"}

        self.report({"INFO"}, f"hPBR material applied to '{obj.name}'")
        return {"FINISHED"}


class HPBR_OT_reload(bpy.types.Operator):
    """Re-read the .hpbr file stored on this object and rebuild its material"""
    bl_idname  = "hpbr.reload_material"
    bl_label   = "Reload hPBR"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and "hpbr_path" in obj

    def execute(self, context):
        obj  = context.object
        path = obj.get("hpbr_path", "")
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"hPBR file not found: {path}")
            return {"CANCELLED"}

        reader = _HpbrReader()
        try:
            reader.read(path)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not read file: {exc}")
            return {"CANCELLED"}

        _haptic_cache[obj.name] = {
            "props": reader.material_props,
            "seg":   reader.seg_arr,
        }
        try:
            _apply_material(obj, reader, path)
        except Exception as exc:
            self.report({"ERROR"}, f"Material rebuild failed: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, "hPBR material reloaded")
        return {"FINISHED"}


# ══════════════════════════════════════════════════════════════════════════════
#  Shared panel drawing
# ══════════════════════════════════════════════════════════════════════════════
def _draw_panel(layout, context):
    obj = context.object

    layout.operator("hpbr.import_material", icon="MATERIAL")

    if obj is None:
        return

    if "hpbr_path" in obj:
        box = layout.box()
        box.label(text=os.path.basename(obj["hpbr_path"]), icon="FILE_TICK")
        box.operator("hpbr.reload_material", text="Reload", icon="FILE_REFRESH")
        entry = _haptic_cache.get(obj.name)
        if entry:
            n_props  = len(entry.get("props", {}))
            has_seg  = entry.get("seg") is not None
            seg_text = " + segmentation" if has_seg else ""
            box.label(
                text=f"{n_props} haptic map(s) cached{seg_text}",
                icon="FORCE_HARMONIC",
            )
        else:
            box.label(text="Haptic maps not yet loaded", icon="INFO")
    else:
        layout.label(text="No hPBR assigned to this object", icon="INFO")


# ══════════════════════════════════════════════════════════════════════════════
#  Panels
# ══════════════════════════════════════════════════════════════════════════════

class HPBR_PT_material_props(bpy.types.Panel):
    """hPBR block in Properties → Material Properties"""
    bl_label       = "hPBR"
    bl_idname      = "HPBR_PT_material_props"
    bl_space_type  = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context     = "material"
    bl_options     = {"DEFAULT_CLOSED"}

    def draw(self, context):
        _draw_panel(self.layout, context)


class HPBR_PT_shader_editor(bpy.types.Panel):
    """hPBR tab in Shader Editor → N-panel"""
    bl_label       = "hPBR"
    bl_idname      = "HPBR_PT_shader_editor"
    bl_space_type  = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category    = "hPBR"

    @classmethod
    def poll(cls, context):
        snode = context.space_data
        return snode is not None and snode.tree_type == "ShaderNodeTree"

    def draw(self, context):
        _draw_panel(self.layout, context)


# ══════════════════════════════════════════════════════════════════════════════
#  Registration
# ══════════════════════════════════════════════════════════════════════════════
_CLASSES = (
    HPBR_OT_import,
    HPBR_OT_reload,
    HPBR_PT_material_props,
    HPBR_PT_shader_editor,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
