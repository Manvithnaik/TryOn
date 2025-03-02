import numpy as np
import open3d as o3d

def scale_body_model(body_model_path, measurements):
    """
    Scale the body model according to the measurements
    
    Args:
        body_model_path (str): Path to the body model file
        measurements (dict): Body measurements
        
    Returns:
        o3d.geometry.TriangleMesh: Scaled body model
    """
    # Load the body model
    body_mesh = o3d.io.read_triangle_mesh(body_model_path)
    
    # Ensure the mesh has vertex normals for proper rendering
    if not body_mesh.has_vertex_normals():
        body_mesh.compute_vertex_normals()
    
    # Get model's original dimensions
    body_vertices = np.asarray(body_mesh.vertices)
    min_bound = body_vertices.min(axis=0)
    max_bound = body_vertices.max(axis=0)
    original_dimensions = max_bound - min_bound
    
    # Calculate scale factors
    # Assuming the model is oriented such that:
    # x-axis is width (left to right)
    # y-axis is height (bottom to top)
    # z-axis is depth (back to front)
    
    # Get current model measurements
    original_height = original_dimensions[1]
    original_shoulder_width = estimate_shoulder_width(body_mesh)
    original_chest_width = estimate_chest_width(body_mesh)
    original_waist_width = estimate_waist_width(body_mesh)
    
    # Calculate scale factors
    height_scale = measurements["height"] / original_height
    shoulder_scale = measurements["shoulder_width"] / original_shoulder_width
    chest_scale = measurements["chest_width"] / original_chest_width
    waist_scale = measurements["waist_width"] / original_waist_width
    
    # Apply weighted average for x-scale (width) from shoulders, chest, and waist
    x_scale = (shoulder_scale * 0.4 + chest_scale * 0.4 + waist_scale * 0.2)
    
    # Use height scale for y (height)
    y_scale = height_scale
    
    # Use average of x and height scale for z (depth)
    z_scale = (x_scale + height_scale) / 2
    
    # Print the scaling factors for debugging
    print(f"Body scaling factors: x={x_scale:.4f}, y={y_scale:.4f}, z={z_scale:.4f}")
    
    # Create scaling matrix
    scaling_matrix = np.array([
        [x_scale, 0, 0, 0],
        [0, y_scale, 0, 0],
        [0, 0, z_scale, 0],
        [0, 0, 0, 1]
    ])
    
    # Apply scaling
    body_mesh.transform(scaling_matrix)
    
    # Verify the mesh still has vertex normals after transformation
    if not body_mesh.has_vertex_normals():
        body_mesh.compute_vertex_normals()
    
    # Save scaled model
    o3d.io.write_triangle_mesh("scaled_body.obj", body_mesh)
    
    return body_mesh

def scale_clothing_model(clothing_model_path, measurements):
    """
    Scale the clothing model to match the body measurements
    
    Args:
        clothing_model_path (str): Path to the clothing model file
        measurements (dict): Body measurements
        
    Returns:
        o3d.geometry.TriangleMesh: Scaled clothing model
    """
    # Load the clothing model
    clothing_mesh = o3d.io.read_triangle_mesh(clothing_model_path)
    
    # Ensure the mesh has vertex normals for proper rendering
    if not clothing_mesh.has_vertex_normals():
        clothing_mesh.compute_vertex_normals()
    
    # Get model's original dimensions
    clothing_vertices = np.asarray(clothing_mesh.vertices)
    min_bound = clothing_vertices.min(axis=0)
    max_bound = clothing_vertices.max(axis=0)
    original_dimensions = max_bound - min_bound
    
    # For a shirt, we need to focus on shoulders, chest, and height
    original_height = original_dimensions[1]
    original_width = original_dimensions[0]
    
    # Estimate target shirt dimensions based on body measurements
    # Shirts need to be larger than the body to fit properly
    target_chest_width = measurements["chest_width"] * 1.1  # 10% larger than body
    target_shoulder_width = measurements["shoulder_width"] * 1.05  # 5% larger than body
    target_waist_width = measurements["waist_width"] * 1.1  # 10% larger for comfort
    
    # Calculate the overall width scale as a weighted average
    width_scale = (target_shoulder_width / original_width * 0.4 + 
                   target_chest_width / original_width * 0.4 + 
                   target_waist_width / original_width * 0.2)
    
    # Height scale based on body height
    # Assuming the shirt is approximately 30% of the body height
    shirt_height_proportion = 0.3
    target_height = measurements["height"] * shirt_height_proportion
    height_scale = target_height / original_height
    
    # Depth scale (front to back) - use width scale as approximation
    depth_scale = width_scale
    
    # Print the scaling factors for debugging
    print(f"Shirt scaling factors: x={width_scale:.4f}, y={height_scale:.4f}, z={depth_scale:.4f}")
    
    # Create scaling matrix
    scaling_matrix = np.array([
        [width_scale, 0, 0, 0],
        [0, height_scale, 0, 0],
        [0, 0, depth_scale, 0],
        [0, 0, 0, 1]
    ])
    
    # Apply scaling
    clothing_mesh.transform(scaling_matrix)
    
    # Verify the mesh still has vertex normals after transformation
    if not clothing_mesh.has_vertex_normals():
        clothing_mesh.compute_vertex_normals()
    
    # Save scaled model
    o3d.io.write_triangle_mesh("scaled_shirt.obj", clothing_mesh)
    
    return clothing_mesh

def estimate_shoulder_width(body_mesh):
    """
    Estimate the shoulder width of the body model
    
    Args:
        body_mesh (o3d.geometry.TriangleMesh): Body mesh
        
    Returns:
        float: Estimated shoulder width
    """
    # This is a simplified estimation
    # In a real implementation, you would need to identify shoulder landmarks
    vertices = np.asarray(body_mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    dimensions = max_bound - min_bound
    
    # Approximate shoulder width as 80% of the maximum width
    # This is a heuristic that would need to be adjusted based on the specific model
    return dimensions[0] * 0.8

def estimate_chest_width(body_mesh):
    """
    Estimate the chest width of the body model
    
    Args:
        body_mesh (o3d.geometry.TriangleMesh): Body mesh
        
    Returns:
        float: Estimated chest width
    """
    # Simplified estimation
    vertices = np.asarray(body_mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    dimensions = max_bound - min_bound
    
    # Approximate chest width as 70% of the maximum width
    return dimensions[0] * 0.7

def estimate_waist_width(body_mesh):
    """
    Estimate the waist width of the body model
    
    Args:
        body_mesh (o3d.geometry.TriangleMesh): Body mesh
        
    Returns:
        float: Estimated waist width
    """
    # Simplified estimation
    vertices = np.asarray(body_mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    dimensions = max_bound - min_bound
    
    # Approximate waist width as 60% of the maximum width
    return dimensions[0] * 0.6