import os

import bpy


# Classes -----------------------------------------------------------------------------------------


class Something3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class AnchorPoint(Something3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)


class EdgeLengths(Something3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)


# Functions ---------------------------------------------------------------------------------------


def add_light(light_counter, location, energy):
    light_data = bpy.data.lights.new(f"light{light_counter}", type="POINT")
    light = bpy.data.objects.new(f"light{light_counter}", light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    bpy.context.collection.objects[f"light{light_counter}"].data.energy = energy

    return light_counter + 1


def add_camera(location, rotation_euler):
    """Add a camera to the scene.

    Inputs
    ------
    location : tuple
        The location of the camera (in meters).
    rotation_euler : tuple
        The rotation of the camera (in radians).
    """

    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = rotation_euler


def clear_scene():
    """Clear the current Blender scene."""
    scene = bpy.context.scene

    for c in scene.collection.children:
        scene.collection.children.unlink(c)


def create_cuboid_mesh(
    anchor_point: AnchorPoint,
    edge_lengths: EdgeLengths,
    name: str,
    return_top_mesh_separately: bool = False,
):
    """Create a cuboid mesh.

    Inputs
    ------
    anchor_point :  AnchorPoint
        The anchor point of the cuboid.
    edge_lengths : EdgeLengths
        The edge lengths of the cuboid.
    name : str
        The name of the mesh.
    return_top_mesh_separately : bool
        Whether to return the top mesh separately.
        Default is False.
    """

    vertices = [
        (anchor_point.x, anchor_point.y, anchor_point.z),
        (anchor_point.x + edge_lengths.x, anchor_point.y, anchor_point.z),
        (anchor_point.x, anchor_point.y + edge_lengths.y, anchor_point.z),
        (
            anchor_point.x + edge_lengths.x,
            anchor_point.y + edge_lengths.y,
            anchor_point.z,
        ),
        (anchor_point.x, anchor_point.y, anchor_point.z + edge_lengths.z),
        (
            anchor_point.x + edge_lengths.x,
            anchor_point.y,
            anchor_point.z + edge_lengths.z,
        ),
        (
            anchor_point.x,
            anchor_point.y + edge_lengths.y,
            anchor_point.z + edge_lengths.z,
        ),
        (
            anchor_point.x + edge_lengths.x,
            anchor_point.y + edge_lengths.y,
            anchor_point.z + edge_lengths.z,
        ),
    ]

    edges = []
    faces = [(0, 1, 3, 2), (0, 1, 5, 4), (1, 3, 7, 5), (3, 7, 6, 2), (0, 2, 6, 4)]

    if return_top_mesh_separately:
        faces_top = [(0, 1, 3, 2)]
        vertices_top = vertices[4:]
        mesh_top = bpy.data.meshes.new(f"{name}_top_mesh")
        mesh_top.from_pydata(vertices_top, edges, faces_top)
        mesh_top.update()
    else:
        faces = faces + [(4, 5, 7, 6)]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()

    if return_top_mesh_separately:
        return mesh, mesh_top
    else:
        return mesh


def create_obstacle(obstacle_counter, anchor_point, edge_lengths, materials, material):
    """Add an obstacle to the scene.

    obstacle_counter : int
        The number of obstacles in the scene.
    anchor_point : AnchorPoint
        The anchor point of the obstacle.
    edge_lengths : EdgeLengths
        The edge lengths of the obstacle.
    materials : dict
        The defined materials of the scene.
    material : str
        The material of the obstacle.
    """
    name = f"obstacle{obstacle_counter}"
    obstacle_mesh = create_cuboid_mesh(anchor_point, edge_lengths, name, False)
    obstacle_object = bpy.data.objects.new(f"{name}_object", obstacle_mesh)
    obstacle_object.data.materials.append(materials[material])
    return (obstacle_counter + 1, obstacle_object)


def create_description_file(output_path, room_size, **kwargs):
    """Create a description file for the scenario.

    output_path : str
        The path to the output directory.
    room_size : EdgeLengths
        The edge lengths of the room.
    """

    description_file = open(os.path.join(output_path, "description.txt"), "w")
    if "object_collection" in kwargs:
        object_collection = kwargs["object_collection"]
        print("Scenario with obstacles\n", file=description_file)
        print(f"Room size: {room_size}", file=description_file)
        print(f"Number of obstacles: {len(object_collection)-1}", file=description_file)
    else:
        print("Scenario without obstacles\n", file=description_file)
        print(f"Room size: {room_size}", file=description_file)

    description_file.close()


def init_materials():
    """Initialize the materials for the scenario.

    Returns
    -------
    materials : dict
        A dictionary containing the initialized materials."""

    materials = {}
    materials["concrete"] = bpy.data.materials.new(name="itu-concrete")
    materials["metal"] = bpy.data.materials.new(name="itu-metal")

    return materials


def link_objects_to_view_layer(object_collection):
    view_layer = bpy.context.view_layer

    for obj in object_collection:
        view_layer.active_layer_collection.collection.objects.link(obj)
