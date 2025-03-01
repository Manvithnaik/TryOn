import open3d as o3d
import numpy as np

class Visualizer:
    def __init__(self, body_mesh, shirt_mesh):
        """Initialize Open3D Visualizer and Load Models"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Try-On", width=800, height=600)

        self.body_mesh = body_mesh
        self.shirt_mesh = shirt_mesh

        self.body_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light Gray
        self.shirt_mesh.paint_uniform_color([0.9, 0.2, 0.2])  # Red

        self.vis.add_geometry(self.body_mesh)
        self.vis.add_geometry(self.shirt_mesh)

        # ✅ Set the camera view to ensure models are visible
        self.view_control = self.vis.get_view_control()
        self.view_control.set_zoom(1.5)  # Zoom in on the models
        self.view_control.set_front([0, 0, -1])  # Set the viewing direction
        self.view_control.set_up([0, -1, 0])  # Align with the correct axis
        self.view_control.set_lookat([0, 0, 0])  # Focus on the center

        print("📸 Open3D Camera Adjusted")

    def update_models(self, position, scale_factor):
        """Update the position and scale of 3D models dynamically"""
        body_vertices = np.asarray(self.body_mesh.vertices) * scale_factor + position
        shirt_vertices = np.asarray(self.shirt_mesh.vertices) * scale_factor + position

        self.body_mesh.vertices = o3d.utility.Vector3dVector(body_vertices)
        self.shirt_mesh.vertices = o3d.utility.Vector3dVector(shirt_vertices)

        self.body_mesh.compute_vertex_normals()
        self.shirt_mesh.compute_vertex_normals()

        self.vis.update_geometry(self.body_mesh)
        self.vis.update_geometry(self.shirt_mesh)

    def render(self):
        """Ensure rendering happens in the main thread"""
        self.vis.poll_events()
        self.vis.update_renderer()

    def force_refresh(self):
        """Fix disappearing models by clearing and re-adding them every frame"""
        print("🔄 Refreshing Open3D Scene...")
        print(f"🟢 Body Mesh Vertices: {len(self.body_mesh.vertices)}")
        print(f"🟢 Shirt Mesh Vertices: {len(self.shirt_mesh.vertices)}")

        self.vis.clear_geometries()
        self.vis.add_geometry(self.body_mesh)
        self.vis.add_geometry(self.shirt_mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
        print("✅ Open3D Updated Successfully")

    def close(self):
        """Properly close Open3D visualizer"""
        self.vis.destroy_window()
print("👋 Open3D Closed Successfully")