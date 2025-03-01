import open3d as o3d
import os

class ModelLoader:
    def __init__(self):
        self.body_model = self.load_mesh("models/body.obj")
        self.shirt_model = self.load_mesh("models/shirt.obj")

    def load_mesh(self, path):
        if not os.path.exists(path):
            print(f"❌ ERROR: {path} not found!")
            return o3d.geometry.TriangleMesh()  # Return empty mesh to avoid crashes

        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_vertices():
            print(f"❌ ERROR: {path} failed to load!")
            return o3d.geometry.TriangleMesh()

        print(f"✅ Loaded: {path}")
        return mesh
