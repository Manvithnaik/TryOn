import open3d as o3d
import numpy as np

def visualize_fitted_models(models_tuple):
    """
    Visualize the fitted body and clothing models with proper colors
    
    Args:
        models_tuple (tuple): (body_mesh, clothing_mesh) to visualize
    """
    body_mesh, clothing_mesh = models_tuple
    
    # Make sure vertices have normals for proper shading
    if not body_mesh.has_vertex_normals():
        body_mesh.compute_vertex_normals()
    if not clothing_mesh.has_vertex_normals():
        clothing_mesh.compute_vertex_normals()
    
    # Set different colors for body and clothing
    # Create a clear distinction between body and clothing
    body_mesh_colored = o3d.geometry.TriangleMesh(body_mesh)
    clothing_mesh_colored = o3d.geometry.TriangleMesh(clothing_mesh)
    
    # Paint with bright, distinct colors
    body_mesh_colored.paint_uniform_color([0.75, 0.6, 0.5])  # Skin-like color
    clothing_mesh_colored.paint_uniform_color([0.2, 0.4, 0.8])  # Blue for shirt
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Virtual Fitting", width=1024, height=768)
    
    # Add the meshes to the visualization
    vis.add_geometry(body_mesh_colored)
    vis.add_geometry(clothing_mesh_colored)
    
    # Set up lighting for better visualization
    opt = vis.get_render_option()
    opt.background_color = np.array([0.8, 0.8, 0.8])  # Light gray background
    opt.point_size = 5.0
    opt.line_width = 2.0
    opt.light_on = True
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.COLOR
    opt.mesh_show_wireframe = False
    opt.show_coordinate_frame = True
    
    # Set viewpoint for better visualization
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    
    # Update view once
    vis.update_geometry(body_mesh_colored)
    vis.update_geometry(clothing_mesh_colored)
    vis.poll_events()
    vis.update_renderer()
    
    # Save a screenshot before running the interactive visualizer
    vis.capture_screen_image("fitting_result.png")
    
    # Run the visualization loop
    print("Visualization window opened. Press 'q' or 'ESC' to exit.")
    print("A screenshot has been saved as 'fitting_result.png'")
    vis.run()
    vis.destroy_window()

def create_screenshot(models_tuple, filename="virtual_fitting.png"):
    """
    Create a screenshot of the fitted models
    
    Args:
        models_tuple (tuple): (body_mesh, clothing_mesh) to visualize
        filename (str): Output filename for the screenshot
    """
    body_mesh, clothing_mesh = models_tuple
    
    # Make sure vertices have normals for proper shading
    if not body_mesh.has_vertex_normals():
        body_mesh.compute_vertex_normals()
    if not clothing_mesh.has_vertex_normals():
        clothing_mesh.compute_vertex_normals()
    
    # Set different colors for body and clothing
    body_mesh_colored = o3d.geometry.TriangleMesh(body_mesh)
    clothing_mesh_colored = o3d.geometry.TriangleMesh(clothing_mesh)
    
    # Paint with bright, distinct colors
    body_mesh_colored.paint_uniform_color([0.75, 0.6, 0.5])  # Skin-like color
    clothing_mesh_colored.paint_uniform_color([0.2, 0.4, 0.8])  # Blue for shirt
    
    # Create a visualization window (off-screen)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Screenshot", width=1024, height=768, visible=False)
    
    # Add the meshes to the visualization
    vis.add_geometry(body_mesh_colored)
    vis.add_geometry(clothing_mesh_colored)
    
    # Set rendering options
    render_options = vis.get_render_option()
    render_options.background_color = np.array([0.8, 0.8, 0.8])  # Light gray background
    render_options.point_size = 5.0
    render_options.line_width = 2.0
    render_options.light_on = True
    render_options.mesh_shade_option = o3d.visualization.MeshShadeOption.COLOR
    
    # Update the view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    
    # Capture the image
    vis.update_geometry(body_mesh_colored)
    vis.update_geometry(clothing_mesh_colored)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    
    # Convert to numpy array and save
    img_array = np.asarray(image)
    o3d.io.write_image(filename, img_array)
    
    vis.destroy_window()
    print(f"Screenshot saved as {filename}")