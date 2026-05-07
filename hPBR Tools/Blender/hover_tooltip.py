bl_info = {
    "name": "Surface Hover Info",
    "version": (2, 0),
    "blender": (4, 0, 0),
    "category": "3D View",
    "description": "Hover over a surface to read per-pixel haptic property values from an hPBR (new format) file",
}

import bpy
import blf
import gpu
import sys
import numpy as np
import os
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
import mathutils

# ── Haptic property definitions: (cache key / npy stem, display label) ───────
# Displayed in this order in the tooltip.
_PROPERTY_DEFS = [
    ("youngs_modulus",       "Young's Modulus [MPa]"),
    ("poissons_ratio",       "Poisson's Ratio"),
    ("roughness_average_ra", "Roughness Average (Ra) [µm]"),
    ("static_friction",      "Static Friction Coefficient"),
    ("kinetic_friction",     "Kinetic Friction Coefficient"),
    ("thermal_conductivity", "Thermal Conductivity [W/(mK)]"),
    ("thermal_effusivity",   "Thermal Effusivity [Ws^0.5/(m^2K)]"),
    ("haptic_tensor",               "Haptic Tensor"),
]

_state = {
    "mouse_x": 0,
    "mouse_y": 0,
    "info":    ["Hover over a surface"],
}


# ── Importer access ────────────────────────────────────────────────────────────
def _get_importer():
    """Return the hpbr_importer module if it is loaded, else None."""
    return sys.modules.get("hpbr_importer")


# ── Sampling helpers ───────────────────────────────────────────────────────────
def _sample_npy(arr, u, v):
    """Return a display string for the value at UV coordinate (u, v)."""
    h, w = arr.shape[0], arr.shape[1]
    x    = int((u % 1.0) * w) % w
    y    = int((v % 1.0) * h) % h
    val  = arr[y, x]
    if isinstance(val, np.ndarray):
        parts = [
            "N/A" if (isinstance(c, float) and np.isnan(c)) else f"{c:.3f}"
            for c in val.flat
        ]
        return "[" + ", ".join(parts) + "]"
    if isinstance(val, (np.floating, float)):
        return "N/A" if np.isnan(float(val)) else f"{float(val):.4f}"
    return str(val)


def _get_material_class(seg_arr, u, v):
    """
    Sample the HapticNetSegmentation array at (u, v).
    Elements are Python strings in the new format (object dtype from np.save/load).
    Returns "Unknown" if the segmentation array is absent.
    """
    if seg_arr is None:
        return "Unknown"
    h, w = seg_arr.shape[0], seg_arr.shape[1]
    x    = int((u % 1.0) * w) % w
    y    = int((v % 1.0) * h) % h
    try:
        val  = seg_arr[y, x]
        name = val.decode("utf-8") if isinstance(val, bytes) else str(val)
        return name.replace("_", " ").capitalize()
    except (IndexError, ValueError):
        return "Unknown"


# ── UV from evaluated (subdivided) mesh ───────────────────────────────────────
def _get_uv(eval_mesh, face_index, hit_local):
    if not eval_mesh.uv_layers.active:
        return None
    uv_data = eval_mesh.uv_layers.active.data
    face    = eval_mesh.polygons[face_index]
    loops   = list(face.loop_indices)
    verts   = [eval_mesh.vertices[eval_mesh.loops[li].vertex_index].co.copy()
               for li in loops]
    uvs     = [uv_data[li].uv.copy() for li in loops]
    w = mathutils.geometry.barycentric_transform(
        hit_local, verts[0], verts[1], verts[2],
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, 0, 1)),
    )
    u = uvs[0].x * w.x + uvs[1].x * w.y + uvs[2].x * w.z
    v = uvs[0].y * w.x + uvs[1].y * w.y + uvs[2].y * w.z
    return u, v


# ── Raycast + build tooltip lines ─────────────────────────────────────────────
def do_raycast(context, mx, my):
    region, rv3d = context.region, context.region_data
    if not region or not rv3d:
        return ["No 3D view"]

    ray_o     = view3d_utils.region_2d_to_origin_3d(region, rv3d, (mx, my))
    ray_d     = view3d_utils.region_2d_to_vector_3d(region, rv3d, (mx, my))
    depsgraph = context.evaluated_depsgraph_get()
    hit, loc, _normal, face_i, obj, _ = context.scene.ray_cast(depsgraph, ray_o, ray_d)

    if not hit:
        return ["No surface under cursor"]

    # Check that the importer module is loaded
    importer = _get_importer()
    if importer is None:
        return [
            f"Object: {obj.name}",
            "hpbr_importer is not loaded",
            "Enable it in Preferences > Add-ons first",
        ]

    haptic_props = importer.get_haptic_data(obj.name)
    seg_arr      = importer.get_seg_data(obj.name)

    if haptic_props is None and seg_arr is None:
        return [
            f"Object: {obj.name}",
            "No hPBR data — import a .hpbr file onto this object first",
        ]

    # Get UV at hit point
    eval_obj  = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    hit_local = obj.matrix_world.inverted() @ loc
    uv        = _get_uv(eval_mesh, face_i, hit_local)
    eval_obj.to_mesh_clear()

    if uv is None:
        return ["No UV map on surface"]

    u, v  = uv
    lines = []

    # 1 ── Per-pixel material class from embedded segmentation map
    lines.append(f"Haptic PBR Material: {_get_material_class(seg_arr, u, v)}")

    # 2 ── Haptic property values in defined order
    if haptic_props:
        for stem, label in _PROPERTY_DEFS:
            arr = haptic_props.get(stem)
            if arr is None:
                continue
            lines.append(f"{label}: {_sample_npy(arr, u, v)}")

    # 3 ── Object name and source file
    lines.append(f"Object: {obj.name}")
    hpbr_path = obj.get("hpbr_path", "")
    if hpbr_path:
        lines.append(f"File: {os.path.basename(hpbr_path)}")

    return lines


# ── Draw callback ──────────────────────────────────────────────────────────────
def draw_callback():
    font_id, padding = 0, 8
    x, y  = _state["mouse_x"] + 20, _state["mouse_y"] + 10
    info  = [l for l in _state["info"] if l]
    if not info:
        return
    blf.size(font_id, 16)
    max_w  = max(blf.dimensions(font_id, l)[0] for l in info)
    line_h = blf.dimensions(font_id, "A")[1] + 6
    bw, bh = max_w + padding * 2, line_h * len(info) + padding * 2

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    verts  = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", (0.05, 0.05, 0.05, 0.88))
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')

    for i, line in enumerate(reversed(info)):
        blf.position(font_id, x + padding, y + padding + i * line_h, 0)
        blf.color(font_id, 1, 1, 1, 1)
        blf.draw(font_id, line)


# ── Operator ───────────────────────────────────────────────────────────────────
class OBJECT_OT_hover_info(bpy.types.Operator):
    bl_idname      = "object.hover_info"
    bl_label       = "Surface Hover Info"
    bl_description = "Show per-pixel surface properties under the cursor"

    _handle      = None
    _cancel_flag = False

    def modal(self, context, event):
        if self.__class__._cancel_flag:
            self.__class__._cancel_flag = False
            context.area.tag_redraw()
            return {'CANCELLED'}

        context.area.tag_redraw()
        if event.type == 'MOUSEMOVE':
            _state["mouse_x"] = event.mouse_region_x
            _state["mouse_y"] = event.mouse_region_y
            _state["info"]    = do_raycast(context, _state["mouse_x"], _state["mouse_y"])
        if event.type == 'ESC':
            self._remove_handler()
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        cls = self.__class__

        if cls._handle is not None:
            self._remove_handler()
            cls._cancel_flag = True
            context.area.tag_redraw()
            return {'FINISHED'}

        cls._cancel_flag = False
        cls._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback, (), 'WINDOW', 'POST_PIXEL'
        )
        _state["mouse_x"] = event.mouse_region_x
        _state["mouse_y"] = event.mouse_region_y
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    @classmethod
    def _remove_handler(cls):
        if cls._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(cls._handle, 'WINDOW')
            cls._handle = None


# ── N-panel ────────────────────────────────────────────────────────────────────
class VIEW3D_PT_hover_info(bpy.types.Panel):
    bl_label       = "Surface Hover Info"
    bl_idname      = "VIEW3D_PT_hover_info"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Hover Info'

    def draw(self, context):
        running = OBJECT_OT_hover_info._handle is not None
        label   = "Stop  (Alt+Shift+H)" if running else "Start  (Alt+Shift+H)"
        icon    = 'HIDE_OFF' if running else 'HIDE_ON'
        self.layout.operator("object.hover_info", text=label, icon=icon)


# ── Keymap ─────────────────────────────────────────────────────────────────────
_keymaps = []

def _register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:
        return 0.5
    km  = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
    kmi = km.keymap_items.new("object.hover_info", type='H', value='PRESS',
                              alt=True, shift=True)
    _keymaps.append((km, kmi))
    return None

def register():
    bpy.utils.register_class(OBJECT_OT_hover_info)
    bpy.utils.register_class(VIEW3D_PT_hover_info)
    bpy.app.timers.register(_register_keymaps, first_interval=0.1)

def unregister():
    for km, kmi in _keymaps:
        km.keymap_items.remove(kmi)
    _keymaps.clear()
    OBJECT_OT_hover_info._remove_handler()
    bpy.utils.unregister_class(VIEW3D_PT_hover_info)
    bpy.utils.unregister_class(OBJECT_OT_hover_info)

if __name__ == "__main__":
    register()
