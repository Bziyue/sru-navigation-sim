"""Utilities to build static collision geometry for shared-map drone navigation."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils

from pxr import Gf, Usd, UsdGeom, UsdPhysics


@dataclass(slots=True)
class _MeshData:
    points: list[Gf.Vec3f]
    face_counts: list[int]
    face_indices: list[int]


def _get_all_matching_child_prims(stage: Usd.Stage, prim_path: str, predicate):
    """Compat wrapper for Isaac Lab versions with and without traverse_instance_prims."""
    try:
        return sim_utils.get_all_matching_child_prims(
            prim_path,
            predicate,
            stage=stage,
            traverse_instance_prims=True,
        )
    except TypeError:
        return sim_utils.get_all_matching_child_prims(
            prim_path,
            predicate,
            stage=stage,
        )


def _transform_points(points: list[Gf.Vec3f], transform: Gf.Matrix4d) -> list[Gf.Vec3f]:
    """Apply a world transform to mesh points."""
    transformed_points: list[Gf.Vec3f] = []
    for point in points:
        transformed = transform.Transform(Gf.Vec3d(point[0], point[1], point[2]))
        transformed_points.append(Gf.Vec3f(float(transformed[0]), float(transformed[1]), float(transformed[2])))
    return transformed_points


def _extract_mesh_data(mesh: UsdGeom.Mesh, transform: Gf.Matrix4d | None = None) -> _MeshData | None:
    """Read mesh geometry and optionally apply a transform to the points."""
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    if not points or not face_counts or not face_indices:
        return None

    point_list = list(points)
    if transform is not None:
        point_list = _transform_points(point_list, transform)

    return _MeshData(
        points=point_list,
        face_counts=[int(value) for value in face_counts],
        face_indices=[int(value) for value in face_indices],
    )


def _collect_prototype_meshes(stage: Usd.Stage, prototype_prim: Usd.Prim) -> dict[str, _MeshData]:
    """Collect all mesh geometry under a point-instancer prototype."""
    prototype_meshes: dict[str, _MeshData] = {}
    for child in _get_all_matching_child_prims(stage, prototype_prim.GetPath(), lambda prim: prim.IsA(UsdGeom.Mesh)):
        mesh_data = _extract_mesh_data(UsdGeom.Mesh(child))
        if mesh_data is None:
            continue
        prototype_meshes[child.GetPath().pathString] = mesh_data
    return prototype_meshes


def _append_mesh_data(
    mesh_data: _MeshData,
    all_points: list[Gf.Vec3f],
    all_face_counts: list[int],
    all_face_indices: list[int],
):
    """Append mesh geometry into the merged buffers."""
    vertex_offset = len(all_points)
    all_points.extend(mesh_data.points)
    all_face_counts.extend(mesh_data.face_counts)
    all_face_indices.extend(index + vertex_offset for index in mesh_data.face_indices)


def _append_mesh_prim(prim: Usd.Prim, xform_cache: UsdGeom.XformCache, all_points, all_face_counts, all_face_indices):
    """Append a standard mesh prim using its world transform."""
    mesh_data = _extract_mesh_data(UsdGeom.Mesh(prim), transform=xform_cache.GetLocalToWorldTransform(prim))
    if mesh_data is not None:
        _append_mesh_data(mesh_data, all_points, all_face_counts, all_face_indices)


def _append_point_instancer(
    stage: Usd.Stage,
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    all_points,
    all_face_counts,
    all_face_indices,
):
    """Append point-instancer prototype geometry as concrete world-space meshes."""
    instancer = UsdGeom.PointInstancer(prim)
    positions = instancer.GetPositionsAttr().Get() or []
    if not positions:
        return

    orientations = instancer.GetOrientationsAttr().Get() or []
    scales = instancer.GetScalesAttr().Get() or []
    proto_indices = instancer.GetProtoIndicesAttr().Get() or []
    prototype_paths = instancer.GetPrototypesRel().GetTargets()
    if not prototype_paths:
        return

    num_instances = len(positions)
    if len(orientations) != num_instances:
        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * num_instances
    if len(scales) != num_instances:
        scales = [Gf.Vec3f(1.0, 1.0, 1.0)] * num_instances
    if len(proto_indices) != num_instances:
        proto_indices = list(proto_indices[:num_instances]) + [0] * max(0, num_instances - len(proto_indices))

    prototype_meshes: dict[str, list[_MeshData]] = {}
    for prototype_path in prototype_paths:
        prototype_prim = stage.GetPrimAtPath(prototype_path)
        if not prototype_prim.IsValid():
            continue
        mesh_group = _collect_prototype_meshes(stage, prototype_prim)
        if mesh_group:
            prototype_meshes[prototype_path.pathString] = list(mesh_group.values())

    if not prototype_meshes:
        return

    instancer_world_transform = xform_cache.GetLocalToWorldTransform(prim)
    for index, position in enumerate(positions):
        prototype_index = min(max(int(proto_indices[index]), 0), len(prototype_paths) - 1)
        prototype_key = prototype_paths[prototype_index].pathString
        mesh_group = prototype_meshes.get(prototype_key)
        if not mesh_group:
            continue

        scale = scales[index]
        orientation = orientations[index]

        scale_transform = Gf.Matrix4d().SetScale(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
        rotation_transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(orientation)))
        local_transform = rotation_transform * scale_transform
        local_transform.SetRow(3, Gf.Vec4d(float(position[0]), float(position[1]), float(position[2]), 1.0))
        final_transform = instancer_world_transform * local_transform

        for prototype_mesh in mesh_group:
            transformed_mesh = _MeshData(
                points=_transform_points(prototype_mesh.points, final_transform),
                face_counts=prototype_mesh.face_counts,
                face_indices=prototype_mesh.face_indices,
            )
            _append_mesh_data(transformed_mesh, all_points, all_face_counts, all_face_indices)


def build_merged_collision_mesh(
    stage: Usd.Stage,
    source_prim_expr: str,
    merged_mesh_path: str,
    hide_merged_mesh: bool = True,
) -> str:
    """Flatten matching source prims into a unified world-space collision mesh."""
    matched_paths = sim_utils.find_matching_prim_paths(source_prim_expr, stage=stage)
    if not matched_paths:
        raise ValueError(f"No prims matched source expression: {source_prim_expr}")

    if stage.GetPrimAtPath(merged_mesh_path).IsValid():
        sim_utils.delete_prim(merged_mesh_path, stage=stage)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    all_points: list[Gf.Vec3f] = []
    all_face_counts: list[int] = []
    all_face_indices: list[int] = []

    for source_path in matched_paths:
        source_prim = stage.GetPrimAtPath(source_path)
        for prim in _get_all_matching_child_prims(stage, source_path, lambda candidate: True):
            prim_path = prim.GetPath().pathString
            if "/prototypes/" in prim_path:
                continue
            if prim == source_prim and prim.IsA(UsdGeom.Mesh):
                _append_mesh_prim(prim, xform_cache, all_points, all_face_counts, all_face_indices)
                continue
            if prim.IsA(UsdGeom.PointInstancer):
                _append_point_instancer(stage, prim, xform_cache, all_points, all_face_counts, all_face_indices)
            elif prim.IsA(UsdGeom.Mesh):
                _append_mesh_prim(prim, xform_cache, all_points, all_face_counts, all_face_indices)

    if not all_points or not all_face_counts or not all_face_indices:
        raise RuntimeError(f"No mesh geometry found under prims matched by: {source_prim_expr}")

    merged_mesh = UsdGeom.Mesh.Define(stage, merged_mesh_path)
    merged_mesh.GetPointsAttr().Set(all_points)
    merged_mesh.GetFaceVertexCountsAttr().Set(all_face_counts)
    merged_mesh.GetFaceVertexIndicesAttr().Set(all_face_indices)
    merged_mesh.GetSubdivisionSchemeAttr().Set("none")

    mesh_prim = merged_mesh.GetPrim()
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision_api.CreateCollisionEnabledAttr().Set(True)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("sdf")

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr().Set(False)
    rigid_body_api.CreateKinematicEnabledAttr().Set(True)

    if hide_merged_mesh:
        UsdGeom.Imageable(mesh_prim).MakeInvisible()

    return merged_mesh_path


def compute_prim_world_aabb(stage: Usd.Stage, prim_path: str) -> tuple[list[float], list[float]]:
    """Compute the world-space axis-aligned bounding box for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path for AABB computation: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    world_bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = world_bbox.GetMin()
    bbox_max = world_bbox.GetMax()
    return [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])], [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])]


def spawn_aabb_boundary_walls(
    stage: Usd.Stage,
    bbox_min: list[float],
    bbox_max: list[float],
    walls_root_path: str = "/World/Boundaries",
    padding: float = 2.0,
):
    """Spawn four kinematic cuboid walls around a world-space AABB."""
    if stage.GetPrimAtPath(walls_root_path).IsValid():
        sim_utils.delete_prim(walls_root_path, stage=stage)

    span_x = max((bbox_max[0] - bbox_min[0]) + 2.0 * padding, 1.0)
    span_y = max((bbox_max[1] - bbox_min[1]) + 2.0 * padding, 1.0)
    wall_height = max(3.0, (bbox_max[2] - bbox_min[2]) + 1.0)
    wall_thickness = max(0.2, max(span_x, span_y, wall_height) / 40.0)
    wall_center_z = bbox_min[2] + wall_height / 2.0

    x_center = (bbox_min[0] + bbox_max[0]) / 2.0
    y_center = (bbox_min[1] + bbox_max[1]) / 2.0
    y_pos_wall = bbox_max[1] + padding + wall_thickness / 2.0
    y_neg_wall = bbox_min[1] - padding - wall_thickness / 2.0
    x_pos_wall = bbox_max[0] + padding + wall_thickness / 2.0
    x_neg_wall = bbox_min[0] - padding - wall_thickness / 2.0

    wall_cfg_x = sim_utils.MeshCuboidCfg(
        size=(span_x, wall_thickness, wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8), opacity=0.15),
    )
    wall_cfg_y = wall_cfg_x.replace(size=(span_y, wall_thickness, wall_height))

    wall_cfg_x.func(f"{walls_root_path}/north", wall_cfg_x, translation=(x_center, y_pos_wall, wall_center_z))
    wall_cfg_x.func(f"{walls_root_path}/south", wall_cfg_x, translation=(x_center, y_neg_wall, wall_center_z))
    wall_cfg_y.func(
        f"{walls_root_path}/east",
        wall_cfg_y,
        translation=(x_pos_wall, y_center, wall_center_z),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),
    )
    wall_cfg_y.func(
        f"{walls_root_path}/west",
        wall_cfg_y,
        translation=(x_neg_wall, y_center, wall_center_z),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),
    )
