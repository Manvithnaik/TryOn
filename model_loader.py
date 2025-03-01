import open3d as o3d
import numpy as np

class ModelLoader:
    def __init__(self):
        self.body_model = self.load_mesh("models/body.obj")
        self.shirt_model = self.load_mesh("models/shirt.obj", is_shirt=True)

    def load_mesh(self, path, is_shirt=False):
        try:
            mesh = o3d.io.read_triangle_mesh(path)

            # Apply colors if missing
            if not mesh.has_vertex_colors():
                if is_shirt:
                    mesh.paint_uniform_color([0.2, 0.3, 0.8])  # Blue shirt
                else:
                    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray body

            mesh.compute_vertex_normals()
            center = mesh.get_center()
            mesh.translate(-center)
            scale = 1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
            mesh.scale(scale, np.array([0, 0, 0]))

            # Adjust position for shirt
            if is_shirt:
                mesh.translate(np.array([0, 0.05, 0]))

            return mesh
        except Exception as e:
            print(f"Error loading mesh {path}: {e}")
            return o3d.geometry.TriangleMesh.create_box()
