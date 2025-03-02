import numpy as np
import open3d as o3d

def fit_clothing_to_body(body_mesh, clothing_mesh):
    """
    Fit the clothing model onto the body model
    
    Args:
        body_mesh (o3d.geometry.TriangleMesh): Scaled body model
        clothing_mesh (o3d.geometry.TriangleMesh): Scaled clothing model
        
    Returns:
        tuple: (body_mesh, clothing_mesh) after fitting
    """
    # Get centers of both meshes
    body_center = body_mesh.get_center()
    clothing_center = clothing_mesh.get_center()
    
    # Calculate translation to align centers horizontally (x and z axes)
    translation = np.array([
        body_center[0] - clothing_center[0],  # X alignment
        0,  # We'll set Y separately
        body_center[2] - clothing_center[2]   # Z alignment
    ])
    
    # Get body bounds
    body_vertices = np.asarray(body_mesh.vertices)
    body_min_bound = body_vertices.min(axis=0)
    body_max_bound = body_vertices.max(axis=0)
    body_height = body_max_bound[1] - body_min_bound[1]
    
    # Get clothing bounds
    clothing_vertices = np.asarray(clothing_mesh.vertices)
    clothing_min_bound = clothing_vertices.min(axis=0)
    clothing_max_bound = clothing_vertices.max(axis=0)
    clothing_height = clothing_max_bound[1] - clothing_min_bound[1]
    
    # Position shirt at upper body
    # Shirt bottom should be around chest area (approximately 60% from bottom of body)
    body_chest_y = body_min_bound[1] + body_height * 0.6
    shirt_bottom_y = clothing_min_bound[1]
    
    # Set Y translation to align shirt bottom with chest position
    translation[1] = body_chest_y - shirt_bottom_y
    
    # Create translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    
    # Apply translation to clothing mesh
    clothing_mesh.transform(translation_matrix)
    
    # Actually move the clothing to outside the body
    # Make shirt slightly bigger to ensure it's not inside the body
    scale_for_fitting = 1.15  # 15% larger to ensure it goes around the body
    
    # Create scaling matrix centered at the current position
    current_center = clothing_mesh.get_center()
    
    # Move to origin
    T1 = np.eye(4)
    T1[:3, 3] = -current_center
    
    # Scale
    S = np.eye(4)
    S[0, 0] = scale_for_fitting  # X scale (width)
    S[2, 2] = scale_for_fitting  # Z scale (depth)
    
    # Move back
    T2 = np.eye(4)
    T2[:3, 3] = current_center
    
    # Combine transformations
    clothing_mesh.transform(T1)
    clothing_mesh.transform(S)
    clothing_mesh.transform(T2)
    
    # Save the fitted models
    o3d.io.write_triangle_mesh("fitted_body.obj", body_mesh)
    o3d.io.write_triangle_mesh("fitted_shirt.obj", clothing_mesh)
    
    # Create a combined mesh for visualization
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh += body_mesh
    combined_mesh += clothing_mesh
    o3d.io.write_triangle_mesh("fitted_combined.obj", combined_mesh)
    
    return (body_mesh, clothing_mesh)