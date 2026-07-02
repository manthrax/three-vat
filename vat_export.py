bl_info = {
    "name": "Thraxes VAT Exporter (VAT 3.0 style)",
    "author": "thrax & Google DeepMind team",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Thrax VAT Exporter",
    "description": "Exports fully compliant Houdini-style VAT3 assets for particles, rigidBody, softBody, dynamicMesh",
    "category": "Import-Export",
}

import bpy
import numpy as np
import json
import os
import math
import mathutils

_vat_sync_in_progress = False
_vat_export_in_progress = False


def normalize_export_path(path_value):
    if not path_value:
        raise ValueError("Export path is empty.")

    raw_path = bpy.path.abspath(path_value).strip()

    # Blender/clipboard input on Windows can sometimes arrive with escaped
    # control characters already interpreted, e.g. "C:\tmp" -> "C:<tab>mp".
    sanitized = (
        raw_path
        .replace('\t', '\\t')
        .replace('\n', '\\n')
        .replace('\r', '\\r')
    )

    normalized = os.path.normpath(sanitized)

    if normalized.endswith(':'):
        raise ValueError(f"Invalid export path: {normalized}")

    return normalized


def sanitize_asset_name(name_value):
    name = (name_value or "").strip().replace('.', '_')
    if not name:
        raise ValueError("Asset name is empty.")

    sanitized = []
    for char in name:
        if char.isalnum() or char in ('_', '-'):
            sanitized.append(char)
        else:
            sanitized.append('_')

    result = ''.join(sanitized).strip('_')
    if not result:
        raise ValueError("Asset name contains no usable characters.")

    return result


def pack_lookup_component(value):
    value = min(max(float(value), 0.0), 1.0 - (1.0 / 65535.0))
    scaled = value * 255.0
    major = int(np.floor(scaled))
    minor = int(np.floor((scaled - major) * 255.0 + 0.5))
    major = min(max(major, 0), 255)
    minor = min(max(minor, 0), 255)
    return major / 255.0, minor / 255.0


def pack_lookup_uv(u_value, v_value):
    u_major, u_minor = pack_lookup_component(u_value)
    v_major, v_minor = pack_lookup_component(v_value)
    return [u_major, u_minor, v_major, v_minor]


def texel_center(index, size):
    if size <= 0:
        return 0.5
    return (index + 0.5) / size


def normalize_to_bounds(data, min_vec, max_vec):
    min_arr = np.asarray(min_vec, dtype=np.float32)
    max_arr = np.asarray(max_vec, dtype=np.float32)
    range_arr = max_arr - min_arr
    range_arr[range_arr == 0.0] = 1.0
    if len(data.shape) > 2 and data.shape[2] >= 3:
        normalized = np.array(data, copy=True)
        normalized[:, :, 0:3] = (data[:, :, 0:3] - min_arr) / range_arr
        return np.clip(normalized, 0.0, 1.0)
    normalized = (data - min_arr) / range_arr
    return np.clip(normalized, 0.0, 1.0)


def normalize_signed(data):
    return np.clip(data * 0.5 + 0.5, 0.0, 1.0)


def choose_texture_width(vertex_count, max_width):
    width = min(max(vertex_count, 3), max(max_width, 3))
    width -= (width % 3)
    return max(width, 3)


def pack_rigid_quaternion(quat):
    q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)
    dominant_index = int(np.argmax(np.abs(q)))

    # Flip the quaternion so the omitted dominant component is non-negative.
    # q and -q represent the same rotation, and this matches the shader's
    # reconstruction path that always solves the missing component as positive.
    if q[dominant_index] < 0.0:
        q = -q

    packed = np.zeros(4, dtype=np.float32)
    packed[3] = dominant_index / 4.0

    # Shader mapping:
    # 0 => omitted w, 1 => omitted x, 2 => omitted y, 3 => omitted z
    if dominant_index == 3:
        packed[3] = 0.0
        packed[0:3] = [q[0], q[1], q[2]]
    elif dominant_index == 0:
        packed[3] = 0.25
        packed[0:3] = [q[1], q[2], q[3]]
    elif dominant_index == 1:
        packed[3] = 0.5
        packed[0:3] = [q[0], q[2], q[3]]
    else:
        packed[3] = 0.75
        packed[0:3] = [q[0], q[1], q[3]]

    return packed


def convert_blender_vec_to_y_up(vec3):
    return np.array([vec3[0], vec3[2], -vec3[1]], dtype=np.float32)


def convert_blender_vectors_to_y_up(vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return np.array(arr, copy=True)
    return np.stack((arr[:, 0], arr[:, 2], -arr[:, 1]), axis=1).astype(np.float32, copy=False)


def convert_blender_quat_to_y_up(quat):
    conversion = mathutils.Euler((-math.pi * 0.5, 0.0, 0.0), 'XYZ').to_quaternion()
    return conversion @ quat @ conversion.inverted()


def normalize_quaternion(quat):
    result = quat.copy()
    result.normalize()
    return result


def pack_compressed_normals(normals):
    arr = np.asarray(normals, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)

    lengths = np.linalg.norm(arr, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    arr = arr / lengths

    y_term = np.clip((1.0 + arr[:, 1]) * 0.5, 1.0e-8, 1.0)
    scale = np.sqrt(y_term)
    angle_x = arr[:, 0] / scale
    angle_y = arr[:, 2] / scale

    packed_x = np.clip(np.floor(((angle_x + 2.0) * 0.25) * 31.5 + 0.5), 0.0, 31.0).astype(np.int32)
    packed_y = np.clip(np.floor(((angle_y + 2.0) * 0.25) * 31.5 + 0.5), 0.0, 31.0).astype(np.int32)

    return ((packed_x * 32 + packed_y) / 1024.0).astype(np.float32)


def transform_positions_with_matrix(vectors, matrix):
    if matrix is None:
        return np.asarray(vectors, dtype=np.float32)
    return np.array(
        [[*(matrix @ mathutils.Vector(co))] for co in vectors],
        dtype=np.float32
    )


def transform_normals_with_matrix(vectors, matrix):
    arr = np.asarray(vectors, dtype=np.float32)
    if matrix is None:
        return arr

    normal_matrix = matrix.to_3x3().inverted().transposed()
    transformed = np.array(
        [[*(normal_matrix @ mathutils.Vector(no))] for no in arr],
        dtype=np.float32
    )

    lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return transformed / lengths


def get_scene_fps(scene):
    if not scene:
        return 24.0
    fps = float(getattr(scene.render, "fps", 24))
    fps_base = float(getattr(scene.render, "fps_base", 1.0))
    if fps_base == 0.0:
        fps_base = 1.0
    return fps / fps_base


def infer_asset_name_from_object(obj):
    if not obj:
        return "vat_export"
    return sanitize_asset_name(obj.name)


def infer_vat_mode_from_object(obj):
    if not obj:
        return 'SOFT_BODY'

    for modifier in getattr(obj, "modifiers", []):
        if modifier.type == 'FLUID':
            return 'FLUID'

    if any(psys.settings.type in {'EMITTER', 'HAIR'} for psys in getattr(obj, "particle_systems", [])):
        return 'PARTICLES'

    for modifier in getattr(obj, "modifiers", []):
        if modifier.type in {'SOFT_BODY', 'CLOTH'}:
            return 'SOFT_BODY'

    for child in gather_mesh_descendants(obj):
        for modifier in getattr(child, "modifiers", []):
            if modifier.type in {'SOFT_BODY', 'CLOTH'}:
                return 'SOFT_BODY'

    if getattr(obj, "rigid_body", None) is not None:
        return 'RIGID'

    if any(child.type == 'MESH' for child in getattr(obj, "children", [])):
        return 'RIGID'

    return 'SOFT_BODY'


def gather_mesh_descendants(obj):
    if not obj:
        return []
    found = []
    stack = list(getattr(obj, "children", []))
    while stack:
        child = stack.pop()
        stack.extend(getattr(child, "children", []))
        if child.type == 'MESH':
            found.append(child)
    return found


def filter_deepest_meshes(meshes):
    mesh_set = set(meshes)
    filtered = []
    for mesh in meshes:
        has_mesh_descendant_in_set = False
        stack = list(getattr(mesh, "children", []))
        while stack:
            child = stack.pop()
            stack.extend(getattr(child, "children", []))
            if child in mesh_set:
                has_mesh_descendant_in_set = True
                break
        if not has_mesh_descendant_in_set:
            filtered.append(mesh)
    return filtered


def has_soft_body_modifier(obj):
    return any(modifier.type in {'SOFT_BODY', 'CLOTH'} for modifier in getattr(obj, "modifiers", []))


def gather_soft_body_sources(obj):
    sources = []
    if obj and obj.type == 'MESH' and has_soft_body_modifier(obj):
        sources.append(obj)
    for child in gather_mesh_descendants(obj):
        if has_soft_body_modifier(child):
            sources.append(child)
    return filter_deepest_meshes(sources)


def get_relative_root_object(context, source_obj, mode):
    if not context or len(context.selected_objects) != 1:
        return None

    root = context.selected_objects[0]
    if mode in {'RIGID', 'SOFT_BODY'}:
        return root

    return None


def on_asset_name_update(self, context):
    self.asset_name_customized = bool(self.asset_name and self.asset_name != self.last_inferred_asset_name)


def on_vat_mode_update(self, context):
    self.vat_mode_customized = self.vat_mode != self.last_inferred_vat_mode


def sync_props_from_selection(context=None, scene=None, obj=None):
    global _vat_sync_in_progress

    if _vat_sync_in_progress:
        return

    if context is None:
        context = bpy.context

    scene = scene or (context.scene if context else None)
    props = getattr(scene, "vat_props", None) if scene else None
    obj = obj or (context.active_object if context else None)
    if not props or not obj:
        return

    _vat_sync_in_progress = True
    try:
        inferred_name = infer_asset_name_from_object(obj)
        inferred_mode = infer_vat_mode_from_object(obj)
        object_key = obj.name_full

        object_changed = props.last_inferred_object != object_key
        props.last_inferred_object = object_key
        props.last_inferred_asset_name = inferred_name
        props.last_inferred_vat_mode = inferred_mode

        if object_changed:
            if not props.asset_name_customized or not props.asset_name:
                props.asset_name = inferred_name
            if not props.vat_mode_customized:
                props.vat_mode = inferred_mode
        else:
            if not props.asset_name_customized and props.asset_name != inferred_name:
                props.asset_name = inferred_name
            if not props.vat_mode_customized and props.vat_mode != inferred_mode:
                props.vat_mode = inferred_mode
    finally:
        _vat_sync_in_progress = False


def vat_selection_sync_handler(scene, depsgraph):
    global _vat_export_in_progress

    if _vat_export_in_progress:
        return

    context = bpy.context
    active_scene = context.scene if context else scene
    active_object = context.active_object if context else None

    if not active_scene or not active_object:
        return

    try:
        sync_props_from_selection(context=context, scene=active_scene, obj=active_object)
    except Exception:
        # Avoid breaking Blender's handler loop if context is temporarily invalid.
        pass

class OBJECT_OT_export_vat(bpy.types.Operator):
    bl_idname = "object.export_vat"
    bl_label = "Export VAT3"
    bl_options = {'REGISTER', 'UNDO'}

    def _progress_start(self, total_steps):
        wm = bpy.context.window_manager
        wm.progress_begin(0, max(total_steps, 1))
        self._progress_total = max(total_steps, 1)
        self._progress_step = 0

    def _progress_advance(self, message=None):
        wm = bpy.context.window_manager
        self._progress_step += 1
        wm.progress_update(min(self._progress_step, self._progress_total))
        if message:
            self.report({'INFO'}, message)

    def _progress_end(self):
        bpy.context.window_manager.progress_end()

    def _build_export_summary(self, asset_name, out_dir, mode, props):
        texture_ext = "exr" if props.export_hdr_textures else "png"
        expected_files = [
            f"{asset_name}_data.json",
            f"{asset_name}_mesh.glb" if mode in {'FLUID', 'SOFT_BODY', 'RIGID', 'PARTICLES'} else None,
            f"{asset_name}_pos.{texture_ext}",
            f"{asset_name}_rot.{texture_ext}" if mode in {'FLUID', 'SOFT_BODY', 'RIGID'} else None,
        ]
        files = [name for name in expected_files if name]
        existing = [name for name in files if os.path.exists(os.path.join(out_dir, name))]
        missing = [name for name in files if name not in existing]
        return {
            "asset_name": asset_name,
            "out_dir": out_dir,
            "mode": mode,
            "files": existing,
            "missing": missing,
        }

    def execute(self, context):
        global _vat_export_in_progress

        scene = context.scene
        obj = context.active_object
            
        props = scene.vat_props
        start_frame = props.frame_start
        end_frame = props.frame_end
        total_frames = (end_frame - start_frame) + 1
        mode = props.vat_mode
        export_obj = obj
        relative_root = None
        soft_body_sources = None
        original_frame = scene.frame_current
        original_subframe = getattr(scene, "frame_subframe", 0.0)
        original_active = context.view_layer.objects.active
        original_selection = list(context.selected_objects)

        if not obj:
            self.report({'ERROR'}, "Select an object to export.")
            return {'CANCELLED'}

        if mode == 'SOFT_BODY':
            soft_body_sources = gather_soft_body_sources(obj)
            if not soft_body_sources:
                self.report({'ERROR'}, "Select a soft body mesh or a parent containing one.")
                return {'CANCELLED'}

        if mode in {'FLUID', 'PARTICLES'} and obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a Mesh for this VAT mode.")
            return {'CANCELLED'}

        relative_root = get_relative_root_object(context, export_obj, mode)
        
        depsgraph = context.evaluated_depsgraph_get()

        try:
            export_root = normalize_export_path(props.export_path)
            asset_name = sanitize_asset_name(props.asset_name or obj.name)
            out_dir = os.path.join(export_root, asset_name)
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Invalid export path: {exc}")
            return {'CANCELLED'}

        total_steps = 5
        if mode == 'FLUID':
            total_steps = 7
        elif mode in {'SOFT_BODY', 'RIGID'}:
            total_steps = 5
        elif mode == 'PARTICLES':
            total_steps = 4

        self._progress_start(total_steps)
        _vat_export_in_progress = True

        try:
            self._progress_advance(f"VAT export started: {asset_name} ({mode})")

            # --- EXECUTE CORRESPONDING VAT TYPE exporter ---
            if mode == 'FLUID':
                self.export_dynamic_mesh(context, export_obj, start_frame, end_frame, total_frames, depsgraph, out_dir, props, asset_name)
            elif mode == 'SOFT_BODY':
                self.export_soft_body(context, export_obj, soft_body_sources, start_frame, end_frame, total_frames, depsgraph, out_dir, props, asset_name, relative_root)
            elif mode == 'RIGID':
                self.export_rigid_body(context, export_obj, start_frame, end_frame, total_frames, depsgraph, out_dir, props, asset_name, relative_root)
            elif mode == 'PARTICLES':
                self.export_particles(context, export_obj, start_frame, end_frame, total_frames, depsgraph, out_dir, props, asset_name)

            summary = self._build_export_summary(asset_name, out_dir, mode, props)
            self._progress_advance("Validating exported files...")

            if summary["missing"]:
                self.report(
                    {'WARNING'},
                    f"VAT export finished with missing files: {', '.join(summary['missing'])}"
                )
            else:
                self.report(
                    {'INFO'},
                    f"VAT export complete: {asset_name} -> {out_dir} ({len(summary['files'])} files)"
                )
        except Exception as exc:
            self.report({'ERROR'}, f"VAT export failed for {asset_name}: {exc}")
            return {'CANCELLED'}
        finally:
            _vat_export_in_progress = False
            scene.frame_set(original_frame, subframe=original_subframe)
            try:
                bpy.ops.object.select_all(action='DESELECT')
                for selected_obj in original_selection:
                    if selected_obj.name in bpy.data.objects:
                        selected_obj.select_set(True)
                if original_active and original_active.name in bpy.data.objects:
                    context.view_layer.objects.active = original_active
            except Exception:
                pass
            try:
                sync_props_from_selection(context=context, scene=scene, obj=original_active or obj)
            except Exception:
                pass
            self._progress_end()

        return {'FINISHED'}

    def export_dynamic_mesh(self, context, obj, start, end, total, depsgraph, out_dir, props, asset_name):
        self.report({'INFO'}, "Extracting Dynamic Mesh (Fluid) topology...")
        self._progress_advance("Scanning dynamic mesh topology...")
        max_triangles = 0
        for f in range(start, end + 1):
            context.scene.frame_set(f)
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            eval_mesh.calc_loop_triangles()
            tri_count = len(eval_mesh.loop_triangles)
            if tri_count > max_triangles:
                max_triangles = tri_count
            eval_obj.to_mesh_clear()
             
        max_vertices = max_triangles * 3
        frame_vertex_counts = [0] * total
        initial_positions = np.zeros((max_vertices, 3), dtype=np.float32)
        texture_width = choose_texture_width(max_vertices, props.max_texture_width)
        rows_per_frame = max(1, int(np.ceil(max_vertices / texture_width)))
        texture_height = total * rows_per_frame
        self._progress_advance(f"Packing dynamic mesh atlas {texture_width}x{texture_height}...")

        pos_archive = np.zeros((texture_height, texture_width, 4), dtype=np.float32)
        rot_archive = np.zeros((texture_height, texture_width, 3), dtype=np.float32)

        # Boundaries
        global_min = np.array([1e10, 1e10, 1e10])
        global_max = np.array([-1e10, -1e10, -1e10])

        for idx, f in enumerate(range(start, end + 1)):
            context.scene.frame_set(f)
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            eval_mesh.calc_loop_triangles()
            
            tri_loops = np.zeros(len(eval_mesh.loop_triangles) * 3, dtype=np.int32)
            eval_mesh.loop_triangles.foreach_get("vertices", tri_loops)
            
            vertex_cos = np.zeros(len(eval_mesh.vertices) * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", vertex_cos)
            vertex_cos = vertex_cos.reshape(-1, 3)
            
            vertex_norms = np.zeros(len(eval_mesh.vertices) * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("normal", vertex_norms)
            vertex_norms = vertex_norms.reshape(-1, 3)

            tri_indices = tri_loops.reshape(-1, 3)
            tri_positions = vertex_cos[tri_indices]
            tri_normals = vertex_norms[tri_indices]

            # Blender's evaluated dynamic topology can reorder triangles between
            # frames. Sorting by centroid gives us a much more stable slot order
            # for the VAT textures than raw loop-triangle iteration order.
            tri_centroids = np.mean(tri_positions, axis=1)
            tri_order = np.lexsort((tri_centroids[:, 2], tri_centroids[:, 1], tri_centroids[:, 0]))
            tri_positions = tri_positions[tri_order]
            tri_normals = tri_normals[tri_order]

            unwelded_pos = tri_positions.reshape(-1, 3)
            unwelded_norm = tri_normals.reshape(-1, 3)
            frame_vertex_counts[idx] = len(unwelded_pos)

            if unwelded_pos.size > 0:
                global_min = np.minimum(global_min, np.amin(unwelded_pos, axis=0))
                global_max = np.maximum(global_max, np.amax(unwelded_pos, axis=0))

            if idx == 0 and len(unwelded_pos) > 0:
                initial_positions[0:len(unwelded_pos)] = unwelded_pos

            frame_row_start = idx * rows_per_frame
            for i in range(len(unwelded_pos)):
                row = frame_row_start + (i // texture_width)
                col = i % texture_width
                pos_archive[row, col, 0:3] = unwelded_pos[i]
                pos_archive[row, col, 3] = 1.0
                rot_archive[row, col] = unwelded_norm[i]

            eval_obj.to_mesh_clear()

        # Write textures
        self._progress_advance("Writing dynamic mesh textures...")
        self.save_texture(pos_archive, "pos", out_dir, asset_name, props, global_min, global_max)
        self.save_texture(rot_archive, "rot", out_dir, asset_name, props)
        self._progress_advance("Writing dynamic mesh carrier mesh...")
        self.export_dynamic_mesh_glb(context, initial_positions, max_triangles, out_dir, asset_name, texture_width, texture_height, rows_per_frame)
        self._progress_advance("Writing dynamic mesh metadata...")
        self.save_json(
            "DynamicMesh",
            total,
            max_vertices,
            global_min,
            global_max,
            out_dir,
            asset_name,
            context.scene,
            props,
            extra_fields={"Frame Vertex Counts": frame_vertex_counts}
        )

    def export_soft_body(self, context, obj, source_objects, start, end, total, depsgraph, out_dir, props, asset_name, relative_root=None):
        self.report({'INFO'}, "Extracting Soft Body morph data...")
        self._progress_advance("Sampling soft body frames...")
        if not source_objects:
            raise ValueError("No soft body source meshes found in the selected export scope.")

        root_inv = relative_root.matrix_world.inverted() if relative_root else None

        context.scene.frame_set(start)
        base_position_parts = []
        vertex_counts = []

        for source_obj in source_objects:
            eval_obj = source_obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            vertex_count = len(eval_mesh.vertices)
            vertex_counts.append(vertex_count)

            base_positions = np.zeros(vertex_count * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", base_positions)
            base_positions = base_positions.reshape(-1, 3)

            if root_inv:
                rel_matrix = root_inv @ eval_obj.matrix_world
                base_positions = transform_positions_with_matrix(base_positions, rel_matrix)

            base_position_parts.append(base_positions)
            eval_obj.to_mesh_clear()

        vertex_count = sum(vertex_counts)
        if vertex_count <= 0:
            raise ValueError("Soft body export scope contains no evaluated vertices.")

        base_positions = np.concatenate(base_position_parts, axis=0)

        pos_archive = np.zeros((total, vertex_count, 4), dtype=np.float32)
        rot_archive = np.zeros((total, vertex_count, 3), dtype=np.float32)

        global_min = np.array([1e10, 1e10, 1e10])
        global_max = np.array([-1e10, -1e10, -1e10])

        for idx, f in enumerate(range(start, end + 1)):
            context.scene.frame_set(f)
            vertex_cos_parts = []
            vertex_norm_parts = []

            for source_obj, source_vertex_count in zip(source_objects, vertex_counts):
                eval_obj = source_obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.to_mesh()

                if len(eval_mesh.vertices) != source_vertex_count:
                    eval_obj.to_mesh_clear()
                    raise ValueError(
                        f"Soft body source '{source_obj.name}' changed vertex count from {source_vertex_count} to {len(eval_mesh.vertices)} at frame {f}."
                    )

                vertex_cos = np.zeros(source_vertex_count * 3, dtype=np.float32)
                eval_mesh.vertices.foreach_get("co", vertex_cos)
                vertex_cos = vertex_cos.reshape(-1, 3)
                rel_matrix = None
                if root_inv:
                    rel_matrix = root_inv @ eval_obj.matrix_world
                    vertex_cos = transform_positions_with_matrix(vertex_cos, rel_matrix)

                vertex_norms = np.zeros(source_vertex_count * 3, dtype=np.float32)
                eval_mesh.vertices.foreach_get("normal", vertex_norms)
                vertex_norms = vertex_norms.reshape(-1, 3)
                if rel_matrix is not None:
                    vertex_norms = transform_normals_with_matrix(vertex_norms, rel_matrix)

                vertex_cos_parts.append(vertex_cos)
                vertex_norm_parts.append(vertex_norms)
                eval_obj.to_mesh_clear()

            vertex_cos = np.concatenate(vertex_cos_parts, axis=0)
            vertex_norms = np.concatenate(vertex_norm_parts, axis=0)

            offsets = vertex_cos - base_positions
            offsets = convert_blender_vectors_to_y_up(offsets)
            vertex_norms = convert_blender_vectors_to_y_up(vertex_norms)

            global_min = np.minimum(global_min, np.amin(offsets, axis=0))
            global_max = np.maximum(global_max, np.amax(offsets, axis=0))

            pos_archive[idx, :, 0:3] = offsets
            pos_archive[idx, :, 3] = pack_compressed_normals(vertex_norms)

        self._progress_advance("Writing soft body textures...")
        self.save_texture(pos_archive, "pos", out_dir, asset_name, props, global_min, global_max)
        self.save_texture(rot_archive, "rot", out_dir, asset_name, props)
        self._progress_advance("Writing soft body carrier mesh...")
        self.export_soft_body_glb(context, source_objects, start, depsgraph, out_dir, asset_name, vertex_counts, total, relative_root)
        self._progress_advance("Writing soft body metadata...")
        self.save_json(
            "Softbody",
            total,
            vertex_count,
            global_min,
            global_max,
            out_dir,
            asset_name,
            context.scene,
            props,
            extra_fields={"Use Compressed Normals": True}
        )

    def export_rigid_body(self, context, obj, start, end, total, depsgraph, out_dir, props, asset_name, relative_root=None):
        self.report({'INFO'}, "Extracting Rigid Body chunks...")
        self._progress_advance("Sampling rigid body chunks...")
        # Rigid body exports expect linked chunk duplicates or separate mesh pieces
        # We group children or parent collections to isolate pivot arrays
        chunks = filter_deepest_meshes(gather_mesh_descendants(obj))
        if not chunks:
            chunks = [obj] # Fallback to active object as single chunk

        chunk_count = len(chunks)
        pos_archive = np.zeros((total, chunk_count, 4), dtype=np.float32)
        rot_archive = np.zeros((total, chunk_count, 4), dtype=np.float32)
        bind_matrices = []
        bind_pivots_y_up = []
        bind_rotations_y_up = []

        global_min = np.array([1e10, 1e10, 1e10])
        global_max = np.array([-1e10, -1e10, -1e10])

        context.scene.frame_set(start)
        for chunk in chunks:
            eval_chunk = chunk.evaluated_get(depsgraph)
            bind_matrix = eval_chunk.matrix_world.copy()
            if relative_root:
                bind_matrix = relative_root.matrix_world.inverted() @ bind_matrix
            bind_matrices.append(bind_matrix.copy())

            bind_pos = bind_matrix.to_translation()
            bind_pivots_y_up.append(
                convert_blender_vec_to_y_up((bind_pos.x, bind_pos.y, bind_pos.z))
            )

            bind_rot_y_up = normalize_quaternion(
                convert_blender_quat_to_y_up(bind_matrix.to_quaternion())
            )
            bind_rotations_y_up.append(bind_rot_y_up)

        for idx, f in enumerate(range(start, end + 1)):
            context.scene.frame_set(f)
            for c_idx, chunk in enumerate(chunks):
                eval_chunk = chunk.evaluated_get(depsgraph)
                matrix = eval_chunk.matrix_world
                if relative_root:
                    matrix = relative_root.matrix_world.inverted() @ matrix
                pos = matrix.to_translation()
                rot = normalize_quaternion(matrix.to_quaternion())
                pos_y_up = convert_blender_vec_to_y_up((pos.x, pos.y, pos.z))
                rot_y_up = normalize_quaternion(convert_blender_quat_to_y_up(rot))
                bind_rot_y_up = bind_rotations_y_up[c_idx]
                rot_delta_y_up = normalize_quaternion(rot_y_up @ bind_rot_y_up.inverted())
                
                packed_rot = pack_rigid_quaternion(rot_delta_y_up)
                pos_archive[idx, c_idx, 0:3] = pos_y_up
                pos_archive[idx, c_idx, 3] = packed_rot[3]
                rot_archive[idx, c_idx, 0:3] = packed_rot[0:3]
                rot_archive[idx, c_idx, 3] = 0.0

                global_min = np.minimum(global_min, pos_y_up)
                global_max = np.maximum(global_max, pos_y_up)

        self._progress_advance("Writing rigid body textures...")
        self.save_texture(pos_archive, "pos", out_dir, asset_name, props, global_min, global_max)
        self.save_texture(rot_archive, "rot", out_dir, asset_name, props)
        self._progress_advance("Writing rigid body carrier mesh...")
        self.export_rigid_body_glb(context, chunks, start, depsgraph, out_dir, asset_name, total, relative_root, bind_matrices, bind_pivots_y_up)
        self._progress_advance("Writing rigid body metadata...")
        self.save_json("Rigidbody", total, chunk_count, global_min, global_max, out_dir, asset_name, context.scene, props)

    def export_particles(self, context, obj, start, end, total, depsgraph, out_dir, props, asset_name):
        self.report({'INFO'}, "Extracting Particle billboard paths...")
        self._progress_advance("Sampling particle frames...")
        if not obj.particle_systems:
            self.report({'ERROR'}, "No active particle system found on object.")
            raise ValueError("No particle systems found on the selected object.")

        source_psys = obj.particle_systems.active or obj.particle_systems[0]
        if not source_psys:
            self.report({'ERROR'}, "No active particle system found on object.")
            raise ValueError("No active particle system found on the selected object.")

        particle_count = 0
        saw_live_particle = False
        for f in range(start, end + 1):
            context.scene.frame_set(f)
            eval_obj = obj.evaluated_get(depsgraph)
            eval_psys = eval_obj.particle_systems.get(source_psys.name)
            if not eval_psys:
                continue

            particle_count = max(particle_count, len(eval_psys.particles))
            if any(getattr(part, "alive_state", 'UNBORN') == 'ALIVE' for part in eval_psys.particles):
                saw_live_particle = True

        if particle_count <= 0:
            raise ValueError("Particle system contains no evaluated particles in the requested frame range.")

        if not saw_live_particle:
            raise ValueError("Particle system has no live particles in the requested frame range.")

        pos_archive = np.zeros((total, particle_count, 4), dtype=np.float32)

        global_min = np.array([1e10, 1e10, 1e10])
        global_max = np.array([-1e10, -1e10, -1e10])

        for idx, f in enumerate(range(start, end + 1)):
            context.scene.frame_set(f)
            eval_obj = obj.evaluated_get(depsgraph)
            eval_psys = eval_obj.particle_systems.get(source_psys.name)
            if not eval_psys:
                continue

            for p_idx, part in enumerate(eval_psys.particles):
                if getattr(part, "alive_state", 'UNBORN') != 'ALIVE':
                    continue
                pos = part.location
                pos_archive[idx, p_idx, 0:3] = [pos.x, pos.y, pos.z]
                pos_archive[idx, p_idx, 3] = 1.0
                global_min = np.minimum(global_min, pos)
                global_max = np.maximum(global_max, pos)

        self._progress_advance("Writing particle textures...")
        self.save_texture(pos_archive, "pos", out_dir, asset_name, props, global_min, global_max)
        self._progress_advance("Writing particle carrier mesh...")
        self.export_particles_glb(context, particle_count, total, out_dir, asset_name)
        self._progress_advance("Writing particle metadata...")
        self.save_json("Particles", total, particle_count, global_min, global_max, out_dir, asset_name, context.scene, props)

    def export_particles_glb(self, context, particle_count, total_frames, out_dir, asset_name):
        if particle_count <= 0:
            raise ValueError("Particle export has no particles to export.")

        mesh_name = f"{asset_name}_mesh"
        temp_mesh = bpy.data.meshes.new(mesh_name)

        vertices = []
        faces = []
        for i in range(particle_count):
            base = len(vertices)
            vertices.extend([
                (-0.5, 0.0, -0.5),
                (0.5, 0.0, -0.5),
                (0.5, 0.0, 0.5),
                (-0.5, 0.0, 0.5),
            ])
            faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

        temp_mesh.from_pydata(vertices, [], faces)
        temp_mesh.update()
        temp_mesh.normals_split_custom_set_from_vertices([(0.0, 1.0, 0.0)] * len(vertices))

        uv_layer = temp_mesh.uv_layers.new(name="UVMap")
        vat_uv1 = temp_mesh.uv_layers.new(name="VAT_UV1")
        quad_uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for poly in temp_mesh.polygons:
            particle_index = poly.vertices[0] // 4
            slot_uv = (texel_center(particle_index, particle_count), texel_center(0, total_frames))
            for loop_offset, loop_index in enumerate(poly.loop_indices):
                vertex_index = poly.vertices[loop_offset]
                uv_layer.data[loop_index].uv = quad_uvs[vertex_index % 4]
                vat_uv1.data[loop_index].uv = slot_uv

        self.export_temp_mesh_glb(context, temp_mesh, out_dir, asset_name)

    def export_soft_body_glb(self, context, source_objects, frame, depsgraph, out_dir, asset_name, vertex_counts, total_frames, relative_root=None):
        context.scene.frame_set(frame)
        mesh_name = f"{asset_name}_mesh"
        temp_mesh = bpy.data.meshes.new(mesh_name)

        vertices = []
        faces = []
        normals = []
        display_uvs = []
        vat_vertex_indices = []
        root_inv = relative_root.matrix_world.inverted() if relative_root else None
        source_vertex_offset = 0

        for source_obj in source_objects:
            eval_obj = source_obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            rel_matrix = None
            if root_inv:
                rel_matrix = root_inv @ eval_obj.matrix_world
                eval_mesh.transform(rel_matrix)
            if len(eval_mesh.loop_triangles) == 0:
                eval_mesh.calc_loop_triangles()

            loop_normals = np.zeros(len(eval_mesh.loops) * 3, dtype=np.float32)
            eval_mesh.loops.foreach_get("normal", loop_normals)
            loop_normals = loop_normals.reshape(-1, 3)
            if rel_matrix is not None:
                loop_normals = transform_normals_with_matrix(loop_normals, rel_matrix)
            source_uv_layer = eval_mesh.uv_layers.active.data if eval_mesh.uv_layers.active else None

            for tri in eval_mesh.loop_triangles:
                face = []
                for corner_index, original_vertex_index in enumerate(tri.vertices):
                    loop_index = tri.loops[corner_index]
                    output_vertex_index = len(vertices)
                    vertices.append(tuple(eval_mesh.vertices[original_vertex_index].co))
                    normals.append(tuple(loop_normals[loop_index]))
                    if source_uv_layer is not None:
                        display_uvs.append(tuple(source_uv_layer[loop_index].uv))
                    else:
                        display_uvs.append((0.0, 0.0))
                    vat_vertex_indices.append(source_vertex_offset + original_vertex_index)
                    face.append(output_vertex_index)
                faces.append(tuple(face))

            source_vertex_offset += len(eval_mesh.vertices)
            eval_obj.to_mesh_clear()

        temp_mesh.from_pydata(vertices, [], faces)
        temp_mesh.update()
        for poly in temp_mesh.polygons:
            poly.use_smooth = True
        if normals:
            temp_mesh.normals_split_custom_set_from_vertices(normals)

        if temp_mesh.uv_layers.get("UVMap"):
            temp_mesh.uv_layers.remove(temp_mesh.uv_layers["UVMap"])
        uv_layer = temp_mesh.uv_layers.new(name="UVMap")
        if temp_mesh.uv_layers.get("VAT_UV1"):
            temp_mesh.uv_layers.remove(temp_mesh.uv_layers["VAT_UV1"])
        vat_uv1 = temp_mesh.uv_layers.new(name="VAT_UV1")
        vertex_count = sum(vertex_counts)
        for poly in temp_mesh.polygons:
            for loop_offset, loop_index in enumerate(poly.loop_indices):
                vertex_index = poly.vertices[loop_offset]
                uv_layer.data[loop_index].uv = display_uvs[vertex_index]
                vat_uv1.data[loop_index].uv = (
                    texel_center(vat_vertex_indices[vertex_index], vertex_count),
                    texel_center(0, total_frames)
                )

        self.export_temp_mesh_glb(context, temp_mesh, out_dir, asset_name)

    def export_rigid_body_glb(self, context, chunks, frame, depsgraph, out_dir, asset_name, total_frames, relative_root=None, bind_matrices=None, bind_pivots_y_up=None):
        context.scene.frame_set(frame)
        mesh_name = f"{asset_name}_mesh"
        temp_mesh = bpy.data.meshes.new(mesh_name)

        vertices = []
        faces = []
        normals = []
        display_uvs = []
        piece_slots = []
        piece_pivots = []

        if bind_matrices is None:
            bind_matrices = [None] * len(chunks)
        if bind_pivots_y_up is None:
            bind_pivots_y_up = [None] * len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            eval_chunk = chunk.evaluated_get(depsgraph)
            eval_mesh = eval_chunk.to_mesh()
            bind_matrix = bind_matrices[chunk_index]
            if bind_matrix is None:
                bind_matrix = eval_chunk.matrix_world
                if relative_root:
                    bind_matrix = relative_root.matrix_world.inverted() @ bind_matrix
            pivot_y_up = bind_pivots_y_up[chunk_index]
            if pivot_y_up is None:
                bind_pos = bind_matrix.to_translation()
                pivot_y_up = convert_blender_vec_to_y_up((bind_pos.x, bind_pos.y, bind_pos.z))

            if len(eval_mesh.loop_triangles) == 0:
                eval_mesh.calc_loop_triangles()
            loop_normals = np.zeros(len(eval_mesh.loops) * 3, dtype=np.float32)
            eval_mesh.loops.foreach_get("normal", loop_normals)
            loop_normals = loop_normals.reshape(-1, 3)
            loop_normals = transform_normals_with_matrix(loop_normals, bind_matrix)
            source_uv_layer = eval_mesh.uv_layers.active.data if eval_mesh.uv_layers.active else None
            chunk_vertices = [tuple(bind_matrix @ v.co) for v in eval_mesh.vertices]
            for tri in eval_mesh.loop_triangles:
                face = []
                for corner_index, original_vertex_index in enumerate(tri.vertices):
                    loop_index = tri.loops[corner_index]
                    output_vertex_index = len(vertices)
                    vertices.append(chunk_vertices[original_vertex_index])
                    normals.append(tuple(loop_normals[loop_index]))
                    if source_uv_layer is not None:
                        display_uvs.append(tuple(source_uv_layer[loop_index].uv))
                    else:
                        display_uvs.append((0.0, 0.0))
                    piece_slots.append(chunk_index)
                    piece_pivots.append((pivot_y_up[0], pivot_y_up[1], pivot_y_up[2]))
                    face.append(output_vertex_index)
                faces.append(tuple(face))
            eval_chunk.to_mesh_clear()

        temp_mesh.from_pydata(vertices, [], faces)
        temp_mesh.update()
        for poly in temp_mesh.polygons:
            poly.use_smooth = True
        if normals:
            temp_mesh.normals_split_custom_set_from_vertices(normals)

        uv_layer = temp_mesh.uv_layers.new(name="UVMap")
        vat_uv1 = temp_mesh.uv_layers.new(name="VAT_UV1")
        vat_uv2 = temp_mesh.uv_layers.new(name="VAT_UV2")
        vat_uv3 = temp_mesh.uv_layers.new(name="VAT_UV3")

        for poly in temp_mesh.polygons:
            for loop_offset, loop_index in enumerate(poly.loop_indices):
                vertex_index = poly.vertices[loop_offset]
                chunk_index = piece_slots[vertex_index]
                pivot = piece_pivots[vertex_index]
                slot_uv = (texel_center(chunk_index, len(chunks)), texel_center(0, total_frames))
                uv_layer.data[loop_index].uv = display_uvs[vertex_index]
                vat_uv1.data[loop_index].uv = slot_uv
                vat_uv2.data[loop_index].uv = (pivot[0], 0.0)
                vat_uv3.data[loop_index].uv = (pivot[1], pivot[2])

        self.export_temp_mesh_glb(context, temp_mesh, out_dir, asset_name)

    def export_temp_mesh_glb(self, context, temp_mesh, out_dir, asset_name):
        mesh_name = temp_mesh.name
        temp_obj = bpy.data.objects.new(mesh_name, temp_mesh)
        temp_collection = bpy.data.collections.new(f"{mesh_name}_export")
        context.scene.collection.children.link(temp_collection)
        temp_collection.objects.link(temp_obj)

        previous_active = context.view_layer.objects.active
        previous_selection = list(context.selected_objects)

        try:
            bpy.ops.object.select_all(action='DESELECT')
            temp_obj.select_set(True)
            context.view_layer.objects.active = temp_obj

            bpy.ops.export_scene.gltf(
                filepath=os.path.join(out_dir, f"{asset_name}_mesh.glb"),
                export_format='GLB',
                use_selection=True,
                export_apply=False,
                export_animations=False,
                export_texcoords=True,
                export_normals=True,
                export_tangents=False,
                export_materials='NONE'
            )
        finally:
            bpy.ops.object.select_all(action='DESELECT')
            for selected_obj in previous_selection:
                if selected_obj.name in bpy.data.objects:
                    selected_obj.select_set(True)
            if previous_active and previous_active.name in bpy.data.objects:
                context.view_layer.objects.active = previous_active

            temp_collection.objects.unlink(temp_obj)
            context.scene.collection.children.unlink(temp_collection)
            bpy.data.collections.remove(temp_collection)
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_mesh)

    def export_dynamic_mesh_glb(self, context, initial_positions, triangle_count, out_dir, asset_name, texture_width, texture_height, rows_per_frame):
        vertex_count = triangle_count * 3
        if vertex_count <= 0:
            raise ValueError("Dynamic mesh export has no triangles to export.")

        mesh_name = f"{asset_name}_mesh"
        temp_mesh = bpy.data.meshes.new(mesh_name)

        # The dynamic-mesh render mesh is only a stable carrier for UV/indexing.
        # Actual animated positions come entirely from the VAT textures, so we do
        # not need to bind this mesh to any particular simulation frame.
        placeholder_triangle = (
            (0.0, 0.0, 0.0),
            (0.0001, 0.0, 0.0),
            (0.0, 0.0001, 0.0),
        )
        vertices = [placeholder_triangle[i % 3] for i in range(vertex_count)]
        faces = [(i, i + 1, i + 2) for i in range(0, vertex_count, 3)]
        temp_mesh.from_pydata(vertices, [], faces)
        temp_mesh.update()

        uv_layer = temp_mesh.uv_layers.new(name="UVMap")
        for poly in temp_mesh.polygons:
            for loop_offset, loop_index in enumerate(poly.loop_indices):
                vertex_index = poly.vertices[loop_offset]
                row_index = vertex_index // texture_width
                col_index = vertex_index % texture_width
                uv_layer.data[loop_index].uv = (
                    texel_center(col_index, texture_width),
                    texel_center(row_index, rows_per_frame)
                )

        self.export_temp_mesh_glb(context, temp_mesh, out_dir, asset_name)

    def save_exr(self, data, suffix, out_dir, name, flip_vertical=False):
        height, width = data.shape[0], data.shape[1]
        channels = data.shape[2] if len(data.shape) > 2 else 1
        
        img = bpy.data.images.new(f"{name}_{suffix}", width=width, height=height, alpha=(channels == 4), float_buffer=True)
        rgba_pixels = np.ones((height, width, 4), dtype=np.float32)
        if channels == 3:
            rgba_pixels[:, :, 0:3] = data
        elif channels == 4:
            rgba_pixels[:, :, 0:4] = data
        else:
            rgba_pixels[:, :, 0] = data
            rgba_pixels[:, :, 1] = data
            rgba_pixels[:, :, 2] = data

        if flip_vertical:
            rgba_pixels = np.flipud(rgba_pixels).copy()
            
        img.pixels.foreach_set(rgba_pixels.ravel())
        img.filepath_raw = os.path.join(out_dir, f"{name}_{suffix}.exr")
        img.file_format = 'OPEN_EXR'
        img.save()
        bpy.data.images.remove(img)

    def save_texture(self, data, suffix, out_dir, name, props, min_vec=None, max_vec=None, flip_vertical=False):
        if props.export_hdr_textures:
            self.save_exr(data, suffix, out_dir, name, flip_vertical=flip_vertical)
            return

        if suffix == "pos":
            if min_vec is None or max_vec is None:
                raise ValueError("Position textures need bounds for quantized PNG export.")
            encoded = normalize_to_bounds(data, min_vec, max_vec)
        else:
            encoded = normalize_signed(data)

        self.save_png(encoded, suffix, out_dir, name, flip_vertical=flip_vertical)

    def save_png(self, data, suffix, out_dir, name, flip_vertical=False):
        height, width = data.shape[0], data.shape[1]
        img = bpy.data.images.new(f"{name}_{suffix}", width=width, height=height, alpha=True, float_buffer=False)
        channels = data.shape[2] if len(data.shape) > 2 else 1
        rgba_pixels = np.ones((height, width, 4), dtype=np.float32)
        if channels == 4:
            rgba_pixels[:, :, 0:4] = np.clip(data, 0.0, 1.0)
        elif channels == 3:
            rgba_pixels[:, :, 0:3] = np.clip(data, 0.0, 1.0)
        else:
            clamped = np.clip(data, 0.0, 1.0)
            rgba_pixels[:, :, 0] = clamped
            rgba_pixels[:, :, 1] = clamped
            rgba_pixels[:, :, 2] = clamped

        if flip_vertical:
            rgba_pixels = np.flipud(rgba_pixels).copy()

        img.pixels.foreach_set(rgba_pixels.ravel())
        img.filepath_raw = os.path.join(out_dir, f"{name}_{suffix}.png")
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)

    def save_json(self, vat_type, frames, vertices, g_min, g_max, out_dir, name, scene, props, extra_fields=None):
        axis_system = "Right-Handed Y-Up"
        export_fps = get_scene_fps(scene)

        # Start from the minimal Houdini-style sidecar fields shared across variants.
        entry = {
            "VAT Type": vat_type,
            "Name": name,
            "Axis System": axis_system,
            "Frame Count": frames,
            "Houdini FPS": float(export_fps),
            "Use HDR Textures": 1 if props.export_hdr_textures else 0,
            "Vertex Count": vertices,
            "Active Pixels Ratio X": 1.0,
            "Active Pixels Ratio Y": 1.0,
            "Bound Min X": float(g_min[0]),
            "Bound Min Y": float(g_min[1]),
            "Bound Min Z": float(g_min[2]),
            "Bound Max X": float(g_max[0]),
            "Bound Max Y": float(g_max[1]),
            "Bound Max Z": float(g_max[2]),
            "Spare Color Texture": 0,
            "Two Position Textures": 0
        }

        # Keep only variant-specific metadata that is actually consumed by the
        # current export/runtime path and not already encoded in the mesh/textures.
        if vat_type == "Softbody":
            entry["Invert Frame V"] = True
        elif vat_type == "Rigidbody":
            entry["Invert Frame V"] = True
        elif vat_type == "Particles":
            entry["Particle Pieces Scale Are In Position Alpha"] = True
        elif vat_type == "DynamicMesh":
            entry["Use Lookup Texture"] = 0
            entry["Dynamic Mesh Packed Position Alpha Mask"] = True

        if extra_fields:
            entry.update(extra_fields)

        sidecar = [entry]
        
        json_path = os.path.join(out_dir, f"{name}_data.json")
        with open(json_path, 'w') as jf:
            json.dump(sidecar, jf, indent=4)

# --- USER INTERFACE AND STORAGE PROPERTIES ---
class VATProperties(bpy.types.PropertyGroup):
    frame_start: bpy.props.IntProperty(name="Start Frame", default=1)
    frame_end: bpy.props.IntProperty(name="End Frame", default=250)
    export_path: bpy.props.StringProperty(
        name="Export Path",
        subtype='DIR_PATH',
        default="C:\\tmp\\VAT"
    )
    asset_name: bpy.props.StringProperty(
        name="Asset Name",
        description="Folder and file basename for exported assets",
        default="vat_export",
        update=on_asset_name_update
    )
    export_hdr_textures: bpy.props.BoolProperty(
        name="HDR Textures",
        description="Export position/rotation textures as EXR instead of quantized PNG",
        default=True
    )
    max_texture_width: bpy.props.IntProperty(
        name="Max Texture Width",
        description="Maximum atlas width for exported VAT textures",
        default=4096,
        min=64,
        soft_max=16384
    )
    vat_mode: bpy.props.EnumProperty(
        name="VAT Mode",
        description="Select the VAT layout type",
        items=[
            ('FLUID', "Dynamic / Fluid Mesh", "Unwelded unique triangles for dynamic topology"),
            ('SOFT_BODY', "Soft Body", "Continuous vertex deformation tracking position and quaternions"),
            ('RIGID', "Rigid Body / Chunks", "Instance-based pivot tracking for fracturing simulations"),
            ('PARTICLES', "Particles", "Instance-based particle points")
        ],
        default='SOFT_BODY',
        update=on_vat_mode_update
    )
    asset_name_customized: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    vat_mode_customized: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    last_inferred_object: bpy.props.StringProperty(default="", options={'HIDDEN'})
    last_inferred_asset_name: bpy.props.StringProperty(default="", options={'HIDDEN'})
    last_inferred_vat_mode: bpy.props.StringProperty(default="SOFT_BODY", options={'HIDDEN'})

class VIEW3D_PT_vat_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VAT Exporter'
    bl_label = "VAT Exporter"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.vat_props
        active_object = context.active_object
        
        col = layout.column(align=True)
        if active_object:
            col.label(text=f"Selected: {active_object.name}")
            col.label(text="Asset name and VAT mode auto-fill from selection until edited.")
        col.prop(props, "frame_start")
        col.prop(props, "frame_end")
        col.prop(props, "export_path")
        col.prop(props, "asset_name")
        col.prop(props, "export_hdr_textures")
        col.prop(props, "max_texture_width")
        col.prop(props, "vat_mode")
        col.label(text=f"Sampling at scene FPS: {get_scene_fps(scene):.3f}")
        
        layout.separator()
        layout.operator("object.export_vat", icon='RENDER_STILL')

classes = (
    VATProperties,
    OBJECT_OT_export_vat,
    VIEW3D_PT_vat_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vat_props = bpy.props.PointerProperty(type=VATProperties)
    if vat_selection_sync_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(vat_selection_sync_handler)
    try:
        sync_props_from_selection()
    except Exception:
        pass

def unregister():
    if vat_selection_sync_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(vat_selection_sync_handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.vat_props

if __name__ == "__main__":
    register()
