import os
import requests
import argparse
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import replicate
from pixel_processor import SeamlessTileProcessor
import base64
import json
import sys
import random
import anthropic  # Add import for Anthropic client

# Load API keys from .env
load_dotenv()

# API keys
GETIMG_API_KEY = os.getenv("GETIMG_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Add Anthropic API key

# Initialize Anthropic client if API key is available
anthropic_client = None
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    masked_key = ANTHROPIC_API_KEY[:4] + "..." + ANTHROPIC_API_KEY[-4:] if len(ANTHROPIC_API_KEY) > 8 else "***"
    print(f"Anthropic API key loaded: {masked_key}")
else:
    print("WARNING: ANTHROPIC_API_KEY not found in environment variables!")
    print("Make sure your .env file contains: ANTHROPIC_API_KEY=your_key_here")

# Debug API key loading
if GETIMG_API_KEY:
    masked_key = GETIMG_API_KEY[:4] + "..." + GETIMG_API_KEY[-4:] if len(GETIMG_API_KEY) > 8 else "***"
    print(f"GetImg API key loaded: {masked_key}")
else:
    print("WARNING: GETIMG_API_KEY not found in environment variables!")
    print("Make sure your .env file contains: GETIMG_API_KEY=your_key_here")

if STABILITY_API_KEY:
    masked_key = STABILITY_API_KEY[:4] + "..." + STABILITY_API_KEY[-4:] if len(STABILITY_API_KEY) > 8 else "***"
    print(f"Stability API key loaded: {masked_key}")
else:
    print("WARNING: STABILITY_API_KEY not found in environment variables!")
    print("Make sure your .env file contains: STABILITY_API_KEY=your_key_here")

if REPLICATE_API_TOKEN:
    masked_key = REPLICATE_API_TOKEN[:4] + "..." + REPLICATE_API_TOKEN[-4:] if len(REPLICATE_API_TOKEN) > 8 else "***"
    print(f"Replicate API token loaded: {masked_key}")
else:
    print("WARNING: REPLICATE_API_TOKEN not found in environment variables!")
    print("Make sure your .env file contains: REPLICATE_API_TOKEN=your_token_here")

# Create output directories
OUTPUT_DIR = "outputted_tilesets"
FINAL_TILESET_DIR = os.path.join(OUTPUT_DIR, "final_tileset")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
PIXELATED_DIR = os.path.join(OUTPUT_DIR, "pixelated")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_TILESET_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PIXELATED_DIR, exist_ok=True)

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def create_inpainting_mask(mask_image, line_width=10):
    """
    Create an inpainting mask with a solid black line at the boundary.
    
    Args:
        mask_image: Original mask image
        line_width: Width of the boundary line to inpaint
    
    Returns:
        Grayscale image with white at boundary (where inpainting should occur)
    """
    print(f"Creating inpainting mask with wider boundary line (width={line_width})...")
    
    # Convert to numpy array
    mask_array = np.array(mask_image)
    height, width = mask_array.shape[:2]
    
    # Create binary mask (1 for red, 0 for black)
    binary_mask = (mask_array[:, :, 0] > 128).astype(np.uint8)
    
    # Find the exact boundary with dilation/erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=line_width)
    eroded = cv2.erode(binary_mask, kernel, iterations=line_width)
    boundary = dilated - eroded  # Boundary region
    
    # Create inpainting mask (white where we want to inpaint, black elsewhere)
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    inpaint_mask[boundary > 0] = 255  # White boundary on black background
    
    # Add a slight feathering to the mask edges for smoother transitions
    inpaint_mask = cv2.GaussianBlur(inpaint_mask, (5, 5), 0)
    
    return Image.fromarray(inpaint_mask, 'L')  # Return as grayscale image

def create_composite_for_inpainting(primary_texture, secondary_texture, mask_image, inpaint_mask):
    """
    Create a composite image with primary and secondary textures, and a gradual blended transition.
    
    Args:
        primary_texture: Primary texture (red regions in mask)
        secondary_texture: Secondary texture (black regions in mask)
        mask_image: Original mask image
        inpaint_mask: Mask showing where to inpaint (white boundary)
    
    Returns:
        Composite image with gradual transition at boundaries
    """
    print("Creating composite image with gradient transition for inpainting...")
    
    # Ensure all images are the same size
    width, height = mask_image.size
    primary_resized = primary_texture.resize((width, height), Image.LANCZOS)
    secondary_resized = secondary_texture.resize((width, height), Image.LANCZOS)
    
    # Convert to numpy arrays
    primary_array = np.array(primary_resized)
    secondary_array = np.array(secondary_resized)
    mask_array = np.array(mask_image)
    inpaint_array = np.array(inpaint_mask)
    
    # Create binary mask (1 for red, 0 for black)
    binary_mask = (mask_array[:, :, 0] > 128).astype(np.float32)
    
    # Create a blurred version of the binary mask for smoother transitions
    blurred_mask = cv2.GaussianBlur(binary_mask, (31, 31), 0)  # Set to (31, 31) - between (21, 21) and (41, 41)
    
    # Create output array
    output_array = np.zeros_like(primary_array, dtype=np.float32)
    
    # Mix textures with alpha blending in boundary region
    for y in range(height):
        for x in range(width):
            if inpaint_array[y, x] > 200:  # Strong inpainting region (center of boundary)
                # Set to black - this will be the area to fully inpaint
                output_array[y, x] = [0, 0, 0, 255]
            elif inpaint_array[y, x] > 0:  # Transition area (feathered edge of boundary)
                # Calculate blend factor based on the mask value
                blend_factor = blurred_mask[y, x]
                # Create a smooth transition between textures
                output_array[y, x] = primary_array[y, x] * blend_factor + secondary_array[y, x] * (1 - blend_factor)
            elif binary_mask[y, x] > 0.5:
                # Primary texture region
                output_array[y, x] = primary_array[y, x]
            else:
                # Secondary texture region
                output_array[y, x] = secondary_array[y, x]
    
    return Image.fromarray(output_array.astype(np.uint8))

def inpaint_boundary_getimg(composite_image, inpaint_mask, primary_name, secondary_name, output_filename):
    """
    Use available API to inpaint the boundary between two textures.
    
    Args:
        composite_image: Image with black boundary line to inpaint
        inpaint_mask: Mask showing where to inpaint (white boundary)
        primary_name: Name of primary texture
        secondary_name: Name of secondary texture
        output_filename: Where to save the result
    """
    print(f"Inpainting boundary between {primary_name} and {secondary_name} using API...")
    
    # Save temporary files
    composite_path = os.path.join(TEMP_DIR, f"composite_{os.path.basename(output_filename)}")
    mask_path = os.path.join(TEMP_DIR, f"mask_{os.path.basename(output_filename)}")
    composite_image.save(composite_path)
    inpaint_mask.save(mask_path)
    
    # Create more specific inpainting prompt with explicit instructions to avoid blue borders
    prompt = (
        f"Create a highly realistic, natural transition between {primary_name} and {secondary_name} textures. "
        f"The transition must have natural colors only - use browns, tans, greens, and grays. "
        f"NO BLUE OR PURPLE BORDERS. "
        f"Include small pebbles, soil, tiny plants, and natural debris in the transition zone. "
        f"If stone meets grass, show gradual weathering, moss growth, and soil accumulation. "
        f"Create worn edges on stone with cracks where small plants grow through. "
        f"The boundary should appear as if it naturally formed over time through weather and use. "
        f"Strictly maintain the original colors of the textures without introducing new color palettes."
    )
    
    negative_prompt = "artificial edges, sharp lines, blue borders, purple edges, colored borders, unnatural transitions, distinct boundaries, tiling artifacts, pixelated, blurry, distortion"
    
    # Try Replicate API if available
    if REPLICATE_API_TOKEN:
        try:
            # Configure environment variable for replicate
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
            
            print("Running enhanced inpainting with Replicate API...")
            
            # Use Replicate's Stable Diffusion inpainting model with the exact version ID
            # Adjusted parameters for better blending
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "prompt": prompt,
                    "image": open(composite_path, "rb"),
                    "mask": open(mask_path, "rb"),
                    "num_outputs": 1,
                    "prompt_strength": 0.85,  
                    "num_inference_steps": 50,
                    "guidance_scale": 10.0  # Increased for more prompt adherence
                }
            )
            
            # Get the output URL
            if output and len(output) > 0:
                image_url = output[0]
                
                # Download the image
                image_response = requests.get(image_url)
                
                if image_response.status_code == 200:
                    # Convert to PIL Image
                    inpainted_image = Image.open(BytesIO(image_response.content)).convert("RGBA")
                    
                    # Apply color correction to remove any blue/purple artifacts
                    try:
                        # Convert to numpy array
                        img_array = np.array(inpainted_image)
                        
                        # Create a mask for areas that might have blue/purple borders
                        # This detects pixels with high blue component compared to red and green
                        blue_mask = ((img_array[:,:,2] > img_array[:,:,0] * 1.2) & 
                                    (img_array[:,:,2] > img_array[:,:,1] * 1.2) &
                                    (img_array[:,:,2] > 80))
                        
                        # Reduce the blue channel in these areas
                        img_array[blue_mask, 2] = (img_array[blue_mask, 0] + img_array[blue_mask, 1]) / 2
                        
                        # Apply softening blur to the entire image
                        blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
                        
                        # Only apply the blur to the transition areas (where mask > 0)
                        mask_array = np.array(inpaint_mask)
                        mask_array = mask_array.astype(float) / 255.0
                        # Expand mask to match the image dimensions
                        if len(mask_array.shape) == 2:
                            mask_expanded = np.stack([mask_array] * 4, axis=2)
                        else:
                            mask_expanded = np.stack([mask_array] * 4, axis=2)
                            
                        # Blend original and blurred based on mask
                        final_array = img_array * (1 - mask_expanded) + blurred * mask_expanded
                        
                        # Convert back to image
                        corrected_image = Image.fromarray(final_array.astype(np.uint8), 'RGBA')
                        
                        # Save the enhanced result
                        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        corrected_image.save(output_filename, "PNG")
                        print(f"Enhanced inpainting with color correction complete! Saved to {output_filename}")
                        
                        return corrected_image
                        
                    except Exception as e:
                        print(f"Warning: Post-processing failed, saving original inpainted image: {e}")
                        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                        inpainted_image.save(output_filename, "PNG")
                        return inpainted_image
                else:
                    print(f"Error: Failed to download inpainted image, status code: {image_response.status_code}")
                    return None
            else:
                print(f"Error: No output received from Replicate API")
                return None
                
        except Exception as e:
            print(f"Error in Replicate inpainting: {e}")
            return None

    # Fall back to GetImg if Replicate failed or not available    
    try:
        # Check if API key is available
        if not GETIMG_API_KEY:
            print("Error: GETIMG_API_KEY not found in environment variables. Please set it.")
            return None
            
        # Convert images to base64
        image_base64 = image_to_base64(composite_image)
        mask_base64 = image_to_base64(inpaint_mask)
        
        # Prepare API request
        url = "https://api.getimg.ai/v1/stable-diffusion/inpaint"
        
        payload = {
            "model": "stable-diffusion-xl-1024-v1-0",  # Using SDXL for better quality
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image_base64,
            "mask_image": mask_base64,
            "strength": 1.0,  # Max strength to fully replace the masked area
            "width": composite_image.width,
            "height": composite_image.height,
            "steps": 40,
            "guidance": 8.0,
            "seed": 0,  # Random seed
            "scheduler": "dpmpp_2m",  # More advanced scheduler for better quality
            "output_format": "png",
            "response_format": "url"  # Get URL back
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": GETIMG_API_KEY
        }
        
        print("Sending request to getimg.ai API...")
        
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        # Parse the response
        response_data = response.json()
        
        # Get the image URL
        if response_data.get("image_url"):
            # Download the image
            image_url = response_data["image_url"]
            image_response = requests.get(image_url)
            
            if image_response.status_code == 200:
                # Convert to PIL Image
                inpainted_image = Image.open(BytesIO(image_response.content)).convert("RGBA")
                
                # Save the result
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                inpainted_image.save(output_filename, "PNG")
                print(f"Inpainting complete! Saved to {output_filename}")
                
                return inpainted_image
            else:
                print(f"Error: Failed to download inpainted image, status code: {image_response.status_code}")
                return None
        else:
            print(f"Error: No image URL found in response: {response_data}")
            return None
            
    except Exception as e:
        print(f"Error in inpainting: {e}")
        return None

def pixelate_tileset(input_dir, output_dir, pixel_size="medium", method="advanced", grid_size=None):
    """
    Apply pixelation to all tiles in a directory using the SeamlessTileProcessor.
    
    Args:
        input_dir: Directory containing tiles to pixelate
        output_dir: Directory to save pixelated tiles
        pixel_size: Size of pixels - "small" (8x8), "medium" (16x16), or "large" (32x32) or "custom"
        method: Processing method - "simple" or "advanced"
        grid_size: Custom grid size (integer) to use when pixel_size is "custom"
    """
    print(f"Pixelating all tiles in {input_dir}...")
    
    # Determine grid size based on pixel_size parameter
    if grid_size is not None:
        # Use custom grid size if provided
        grid_size_value = grid_size
        size_label = f"{grid_size_value}x{grid_size_value}"
        print(f"Using custom grid size: {grid_size_value}x{grid_size_value}")
    elif pixel_size == "small":
        grid_size_value = 8
        size_label = "8x8"
    elif pixel_size == "medium":
        grid_size_value = 16
        size_label = "16x16"
    elif pixel_size == "large":
        grid_size_value = 32
        size_label = "32x32"
    else:
        # Default to medium if unrecognized
        grid_size_value = 16  
        size_label = "16x16"
        print(f"Unrecognized pixel size '{pixel_size}', defaulting to medium (16x16)")
    
    # Create processor with the specified grid size
    processor = SeamlessTileProcessor(
        num_colors=16,
        grid_size=grid_size_value,
        scale_factor=1,
        force_16x16=False
    )
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.webp'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        if method == "simple":
            output_path = os.path.join(output_dir, f"{base_name}_{size_label}.png")
            
            # Load image
            img = Image.open(input_path).convert("RGBA")
            
            # Apply simple pixelation
            pixelated = processor.pixelate(img)
            
            # Save result
            pixelated.save(output_path)
            print(f"Applied simple pixelation to {image_file}, saved to {output_path}")
            
        else:  # Advanced processing
            output_path = os.path.join(output_dir, f"{base_name}_{size_label}_processed.png")
            
            # Load image
            img = Image.open(input_path).convert("RGBA")
            
            # Apply full processing
            processor.process_texture(
                img,
                output_path=output_path,
                make_seamless=True,
                harmonize=True
            )
            
            print(f"Applied advanced processing to {image_file}, saved to {output_path}")

def generate_tileset(primary_texture_path, secondary_texture_path, mask_dir, 
                     primary_name=None, secondary_name=None,
                     output_dir=FINAL_TILESET_DIR,
                     apply_pixelation=False, pixel_size="medium", method="advanced",
                     line_width=10, theme_name="custom", grid_size=None):
    """
    Generate a complete tileset by combining two texture images with masked inpainting.
    Pixelation is completely separate and only applied at the end if requested.
    
    Args:
        primary_texture_path: Path to primary texture image
        secondary_texture_path: Path to secondary texture image
        mask_dir: Directory containing mask images
        primary_name: Name of primary texture (for AI prompts)
        secondary_name: Name of secondary texture (for AI prompts)
        output_dir: Directory to save the final tileset
        apply_pixelation: Whether to apply pixelation to the final tiles
        pixel_size: Size of pixels - "small" (8x8), "medium" (16x16), "large" (32x32) or "custom"
        method: Processing method - "simple" or "advanced"
        line_width: Width of the boundary line to inpaint
        theme_name: Name to use for the output files
        grid_size: Custom grid size to use when pixel_size is "custom"
    """
    print(f"Generating tileset using textures: {primary_texture_path} and {secondary_texture_path}")
    
    # Use filenames if names not provided
    if not primary_name:
        primary_name = os.path.splitext(os.path.basename(primary_texture_path))[0]
    if not secondary_name:
        secondary_name = os.path.splitext(os.path.basename(secondary_texture_path))[0]
    
    # Make sure the output directory exists
    output_dir_highres = os.path.join(output_dir, "highres")
    os.makedirs(output_dir_highres, exist_ok=True)
    
    # Load the input textures
    try:
        primary_texture = Image.open(primary_texture_path).convert("RGBA")
        secondary_texture = Image.open(secondary_texture_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading textures: {e}")
        return
    
    # Get all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    
    if not mask_files:
        print(f"No mask files found in the directory: {mask_dir}")
        return
    
    # Create each tile in the tileset - HIGH RESOLUTION ONLY
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        base_filename = os.path.splitext(mask_file)[0]
        output_filename = os.path.join(output_dir_highres, f"{base_filename}_{theme_name}.png")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Load mask image
        mask_image = Image.open(mask_path).convert("RGBA")
        
        # Create inpainting mask and composite image
        inpaint_mask = create_inpainting_mask(mask_image, line_width=line_width)
        composite_image = create_composite_for_inpainting(
            primary_texture, secondary_texture, mask_image, inpaint_mask)
        
        # Use getimg.ai for inpainting - NO PIXELATION HERE
        print(f"Processing tile: {base_filename}")
        inpainted_image = inpaint_boundary_getimg(
            composite_image,
            inpaint_mask,
            primary_name,
            secondary_name,
            output_filename
        )
        
        if inpainted_image is None:
            print(f"Failed to inpaint textures for {base_filename}, skipping...")
            continue
    
    print(f"\nComplete high-resolution tileset generated in '{output_dir_highres}'!")
    
    # After all tiles are created, apply pixelation ONLY if requested
    if apply_pixelation:
        print("Applying pixelation to all generated tiles as a SEPARATE STEP...")
        pixelated_output_dir = os.path.join(PIXELATED_DIR, f"{theme_name}_pixelated")
        pixelate_tileset(
            output_dir_highres,  # Input the high-res blended tiles
            pixelated_output_dir,
            pixel_size=pixel_size,
            method=method,
            grid_size=grid_size
        )
        print(f"Pixelated tileset saved to {pixelated_output_dir}")

def generate_single_texture_tileset(base_texture_path, mask_dir, 
                            base_name=None, target_name=None,
                            output_dir=FINAL_TILESET_DIR,
                            apply_pixelation=False, pixel_size="medium", method="advanced",
                            line_width=10, theme_name="custom", grid_size=None):
    """
    Generate a tileset using a single base texture and inpainting for transitions.
    
    Args:
        base_texture_path: Path to the base texture image
        mask_dir: Directory containing mask images
        base_name: Name of the base texture (for AI prompts)
        target_name: Name of the texture to generate in masked areas (for AI prompts)
        output_dir: Directory to save the final tileset
        apply_pixelation: Whether to apply pixelation to the final tiles
        pixel_size: Size of pixels - "small" (8x8), "medium" (16x16), "large" (32x32) or "custom"
        method: Processing method - "simple" or "advanced"
        line_width: Width of the boundary line to inpaint
        theme_name: Name to use for the output files
        grid_size: Custom grid size to use when pixel_size is "custom"
    """
    print(f"Generating tileset using single base texture: {base_texture_path}")
    print(f"Target texture to generate in masked areas: {target_name}")
    
    # Use filenames if names not provided
    if not base_name:
        base_name = os.path.splitext(os.path.basename(base_texture_path))[0]
    
    # Make sure the output directory exists
    output_dir_highres = os.path.join(output_dir, "highres")
    os.makedirs(output_dir_highres, exist_ok=True)
    
    # Load the base texture
    try:
        base_texture = Image.open(base_texture_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading base texture: {e}")
        return
    
    # Get all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    
    if not mask_files:
        print(f"No mask files found in the directory: {mask_dir}")
        return
    
    # Create each tile in the tileset - HIGH RESOLUTION ONLY
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        base_filename = os.path.splitext(mask_file)[0]
        output_filename = os.path.join(output_dir_highres, f"{base_filename}_{theme_name}.png")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Load mask image
        mask_image = Image.open(mask_path).convert("RGBA")
        
        # Create inpainting mask
        print(f"Processing tile: {base_filename}")
        
        # Create a mask where we want to inpaint
        # Red areas in the mask will remain as the base texture
        # Black areas will be inpainted with the target texture
        mask_array = np.array(mask_image)
        inpaint_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        
        # Set black areas of mask to white in inpaint_mask (areas to inpaint)
        inpaint_mask[mask_array[:, :, 0] < 128] = 255
        
        # Create boundary mask (dilated - eroded)
        kernel = np.ones((3, 3), np.uint8)
        red_mask = (mask_array[:, :, 0] > 128).astype(np.uint8)
        black_mask = (mask_array[:, :, 0] < 128).astype(np.uint8)
        
        # Dilate both masks
        dilated_red = cv2.dilate(red_mask, kernel, iterations=line_width)
        dilated_black = cv2.dilate(black_mask, kernel, iterations=line_width)
        
        # Find boundary area - only the overlap of dilations
        boundary_mask = (dilated_red & dilated_black).astype(np.uint8) * 255
        
        # Create the controlled inpainting area - only inpaint the boundary + black areas
        controlled_mask = (boundary_mask | (black_mask * 255)).astype(np.uint8)
        
        # Apply slight feathering for smooth edges
        inpaint_mask_blurred = cv2.GaussianBlur(controlled_mask, (5, 5), 0)
        
        # Convert to PIL Image
        inpaint_mask_img = Image.fromarray(inpaint_mask_blurred)
        
        # Create composite image
        # Start with base texture, then let the AI inpaint the target areas
        composite_image = base_texture.resize(mask_image.size, Image.LANCZOS)
        
        # Save temporary files
        composite_path = os.path.join(TEMP_DIR, f"composite_{os.path.basename(output_filename)}")
        mask_path = os.path.join(TEMP_DIR, f"mask_{os.path.basename(output_filename)}")
        composite_image.save(composite_path)
        inpaint_mask_img.save(mask_path)
        
        # Create specialized prompt for stone-to-grass transition
        if "stone" in base_name.lower() and "grass" in target_name.lower():
            prompt = (
                f"Create a realistic transition from stone floor to grass ground. "
                f"The stone area should have large flat stone slabs with minimal cracks. "
                f"The grass should have tufts growing between stones at edges, with some small pebbles and dirt. "
                f"Center of grass area should have medium-sized grass clumps with plenty of space between them. "
                f"The transition should look extremely natural, with grass partially growing over stone edges, "
                f"dirt accumulating in the cracks of stones near the grass, and weathered stone edges. "
                f"Maintain the same stone color and pattern but add realistic grass growth patterns. "
                f"The texture must be completely seamless with no borders or edges. "
                f"Create oversized grass elements - large grass clumps with plenty of space between them. "
                f"Use natural earth tones - greens, browns, tans, and grays only."
            )
        else:
            # Fallback to generic prompt for other texture combinations
            prompt = (
                f"Create a highly detailed {target_name} texture transitioning from {base_name}. "
                f"IMPORTANT: This is a BACKGROUND texture with FEW, LARGE elements. Include just 3-5 major features that are LARGER than normal. "
                f"Make this a SIMPLE, CLEAN texture that won't compete with foreground elements. "
                f"The texture must be designed for TRULY SEAMLESS tiling with ABSOLUTELY NO BORDERS, EDGES, FRAMES or boundary indicators. "
                f"It should appear as though it continues infinitely in all directions. "
                f"The {target_name} should have natural but minimal variations - oversized elements with plenty of space between them. "
                f"Blend naturally with the surrounding {base_name} texture at transitions. "
                f"Make the transition zone appear naturally eroded and weathered, with gradual blending between textures."
            )
        
        negative_prompt = (
            "sharp edges, blue borders, purple borders, distinct boundaries, unnatural colors, geometric patterns, "
            "artificial lines, tiling artifacts, small details, dense elements, text, writing, letters, numbers, "
            "words, labels, symbols, characters, fonts, typography, watermarks, signatures, gradients, color fades, "
            "smooth color transitions, gradient overlays, shading effects, color blending, soft shadows, "
            "gradient backgrounds, fading effects, color ramps, vignettes, radial gradients"
        )
        
        # Use Replicate API for inpainting
        if REPLICATE_API_TOKEN:
            try:
                # Configure environment variable for replicate
                os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
                
                print(f"Inpainting {target_name} texture using Replicate API...")
                
                # Use Replicate's Stable Diffusion inpainting model with enhanced parameters
                output = replicate.run(
                    "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                    input={
                        "prompt": prompt,
                        "image": open(composite_path, "rb"),
                        "mask": open(mask_path, "rb"),
                        "num_outputs": 1,
                        "prompt_strength": 0.95,
                        "num_inference_steps": 75,
                        "guidance_scale": 12.0,
                        "seed": random.randint(1, 1000000)  # Use random seed for variation
                    }
                )
                
                # Get the output URL
                if output and len(output) > 0:
                    image_url = output[0]
                    
                    # Download the image
                    image_response = requests.get(image_url)
                    
                    if image_response.status_code == 200:
                        # Convert to PIL Image
                        inpainted_image = Image.open(BytesIO(image_response.content)).convert("RGBA")
                        
                        # Post-process to maintain color consistency
                        try:
                            # Convert to numpy array
                            inpainted_array = np.array(inpainted_image)
                            base_array = np.array(base_texture.resize(inpainted_image.size, Image.LANCZOS))
                            
                            # Create mask arrays for the original red and black areas
                            red_area = np.stack((red_mask,) * 4, axis=-1).astype(np.float32)
                            
                            # Calculate blending weight based on distance from boundary
                            dist_transform = cv2.distanceTransform(red_mask, cv2.DIST_L2, 3)
                            max_dist = np.max(dist_transform)
                            if max_dist > 0:
                                color_blend_factor = np.minimum(dist_transform / (max_dist * 0.3), 1.0)
                                color_blend_factor = np.stack((color_blend_factor,) * 4, axis=-1)
                                
                                # Blend more of original color deeper into the red area 
                                blend_mask = red_area * color_blend_factor
                                
                                # Final image: use inpainted image, but blend original colors in red areas
                                final_array = inpainted_array * (1 - blend_mask) + base_array * blend_mask
                                
                                # Create final image
                                final_image = Image.fromarray(final_array.astype(np.uint8), 'RGBA')
                                final_image.save(output_filename)
                                print(f"Enhanced inpainting with color preservation complete! Saved to {output_filename}")
                            else:
                                inpainted_image.save(output_filename)
                        except Exception as e:
                            print(f"Warning: Post-processing failed, saving original inpainted image: {e}")
                            inpainted_image.save(output_filename)
                            
                        print(f"Single texture inpainting complete! Saved to {output_filename}")
                    else:
                        print(f"Error: Failed to download inpainted image, status code: {image_response.status_code}")
                else:
                    print(f"Error: No output received from Replicate API")
                    
            except Exception as e:
                print(f"Error in Replicate inpainting: {e}")
    
    print(f"\nComplete high-resolution tileset generated in '{output_dir_highres}'!")
    
    # After all tiles are created, apply pixelation ONLY if requested
    if apply_pixelation:
        print("Applying pixelation to all generated tiles as a SEPARATE STEP...")
        pixelated_output_dir = os.path.join(PIXELATED_DIR, f"{theme_name}_pixelated")
        pixelate_tileset(
            output_dir_highres,  # Input the high-res blended tiles
            pixelated_output_dir,
            pixel_size=pixel_size,
            method=method,
            grid_size=grid_size
        )
        print(f"Pixelated tileset saved to {pixelated_output_dir}")

def blend_textures(texture1_path, texture2_path, mask_dir, 
                  output_dir=FINAL_TILESET_DIR,
                  texture1_name=None, texture2_name=None,
                  apply_pixelation=False, pixel_size="medium", method="advanced",
                  boundary_width=20, theme_name=None, grid_size=None, mask_file=None):
    """
    Create a specialized tileset blending any two textures with AI inpainting
    focused only on the boundary transitions using dynamic prompt generation.
    
    Args:
        texture1_path: Path to the first texture image
        texture2_path: Path to the second texture image
        mask_dir: Directory containing mask images
        output_dir: Directory to save the final tileset
        texture1_name: Name of first texture (for AI prompts)
        texture2_name: Name of second texture (for AI prompts)
        apply_pixelation: Whether to apply pixelation to the final tiles
        pixel_size: Size of pixels - "small" (8x8), "medium" (16x16), "large" (32x32) or "custom"
        method: Processing method - "simple" or "advanced"
        boundary_width: Width of the transition boundary to inpaint
        theme_name: Name to use for the output files
        grid_size: Custom grid size to use when pixel_size is "custom"
        mask_file: Optional specific mask file to process (instead of all masks in the directory)
    """
    # Extract texture names from filenames if not provided
    if not texture1_name:
        texture1_name = os.path.splitext(os.path.basename(texture1_path))[0]
    if not texture2_name:
        texture2_name = os.path.splitext(os.path.basename(texture2_path))[0]
    
    # Generate theme name if not provided
    if theme_name is None:
        theme_name = f"{texture1_name}_{texture2_name}_blend"
    
    print(f"Generating specialized {texture1_name}-{texture2_name} tileset")
    print(f"Texture 1: {texture1_path}")
    print(f"Texture 2: {texture2_path}")
    
    # Make sure the output directory exists with theme-specific subdirectory
    output_dir_highres = os.path.join(output_dir, "highres", theme_name)
    os.makedirs(output_dir_highres, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Load the textures
    try:
        texture1 = Image.open(texture1_path).convert("RGBA")
        texture2 = Image.open(texture2_path).convert("RGBA")
        print(f"Loaded textures: {texture1_name} {texture1.size}, {texture2_name} {texture2.size}")
    except Exception as e:
        print(f"Error loading textures: {e}")
        return
    
    # Get mask files
    if mask_file:
        mask_files = [mask_file]
        mask_dir = os.path.dirname(os.path.join(mask_dir, mask_file))
    else:
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    
    if not mask_files:
        print(f"No mask files found in the directory: {mask_dir}")
        return
    
    # Generate the specialized prompt for this texture combination
    specialized_prompt = generate_transition_prompt(texture1_name, texture2_name)
    print(f"Generated specialized prompt for {texture1_name}-{texture2_name} transition")
    
    # General negative prompt for all textures
    negative_prompt = (
        "sharp edges, blue borders, purple borders, distinct boundaries, unnatural colors, geometric patterns, "
        "artificial lines, tiling artifacts, small details, dense elements, text, writing, letters, numbers, "
        "words, labels, symbols, characters, fonts, typography, watermarks, signatures, gradients, color fades, "
        "smooth color transitions, gradient overlays, shading effects, color blending, soft shadows, "
        "gradient backgrounds, fading effects, color ramps, vignettes, radial gradients"
    )
    
    # Process each mask file
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        base_filename = os.path.splitext(mask_file)[0]
        output_filename = os.path.join(output_dir_highres, f"{base_filename}_{theme_name}.png")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Load mask image
        mask_image = Image.open(mask_path).convert("RGBA")
        width, height = mask_image.size
        print(f"Processing tile: {base_filename} ({width}x{height})")
        
        # Resize textures to match mask size
        texture1_resized = texture1.resize((width, height), Image.LANCZOS)
        texture2_resized = texture2.resize((width, height), Image.LANCZOS)
        
        # Convert mask to numpy array
        mask_array = np.array(mask_image)
        
        # Create binary masks for texture1 (red) and texture2 (black) areas
        texture1_mask = (mask_array[:, :, 0] > 128).astype(np.uint8)
        texture2_mask = (mask_array[:, :, 0] <= 128).astype(np.uint8)
        
        # Create dilated versions of each mask to find the boundary
        kernel = np.ones((5, 5), np.uint8)
        texture1_dilated = cv2.dilate(texture1_mask, kernel, iterations=boundary_width//2)
        texture2_dilated = cv2.dilate(texture2_mask, kernel, iterations=boundary_width//2)
        
        # The boundary is where the dilated masks overlap
        boundary = (texture1_dilated & texture2_dilated).astype(np.uint8) * 255
        
        # Apply gaussian blur to create a smooth transition
        # Ensure kernel size is odd (required by GaussianBlur)
        kernel_size = boundary_width if boundary_width % 2 == 1 else boundary_width + 1
        boundary_blurred = cv2.GaussianBlur(boundary, (kernel_size, kernel_size), 0)
        
        # Create initial composite with texture1 and texture2 according to the mask
        composite_array = np.zeros((height, width, 4), dtype=np.uint8)
        texture1_array = np.array(texture1_resized)
        texture2_array = np.array(texture2_resized)
        
        # Fill composite with texture1 in texture1 areas and texture2 in texture2 areas
        for y in range(height):
            for x in range(width):
                if texture1_mask[y, x]:
                    composite_array[y, x] = texture1_array[y, x]
                else:
                    composite_array[y, x] = texture2_array[y, x]
        
        # Convert boundary to mask for inpainting (white = area to inpaint)
        inpaint_mask = Image.fromarray(boundary_blurred)
        composite_image = Image.fromarray(composite_array)
        
        # Save temporary files for inpainting
        temp_composite_path = os.path.join(TEMP_DIR, f"composite_{base_filename}.png")
        temp_mask_path = os.path.join(TEMP_DIR, f"mask_{base_filename}.png")
        composite_image.save(temp_composite_path)
        inpaint_mask.save(temp_mask_path)
        
        # Use Replicate API for inpainting only the boundary area
        if REPLICATE_API_TOKEN:
            try:
                # Configure environment variable for replicate
                os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
                
                print(f"Inpainting {texture1_name}-{texture2_name} boundary using Replicate API...")
                
                # Print the prompt before sending to API
                print("\n=== INPAINTING PROMPT ===")
                print(specialized_prompt)
                print("=== END OF PROMPT ===\n")
                
                print("\n=== NEGATIVE PROMPT ===")
                print(negative_prompt)
                print("=== END OF NEGATIVE PROMPT ===\n")
                
                # Use Replicate's Stable Diffusion inpainting model with focused parameters
                output = replicate.run(
                    "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                    input={
                        "prompt": specialized_prompt,
                        "negative_prompt": negative_prompt,
                        "image": open(temp_composite_path, "rb"),
                        "mask": open(temp_mask_path, "rb"),
                        "num_outputs": 1,
                        "prompt_strength": 0.8,  # Lower for subtler effect on boundary
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                        "seed": 12345  # Fixed seed for consistency
                    }
                )
                
                # Get the output URL
                if output and len(output) > 0:
                    image_url = output[0]
                    
                    # Download the image
                    image_response = requests.get(image_url)
                    
                    if image_response.status_code == 200:
                        # Convert to PIL Image
                        inpainted_image = Image.open(BytesIO(image_response.content)).convert("RGBA")
                        
                        # Save the result
                        inpainted_image.save(output_filename)
                        print(f"{texture1_name}-{texture2_name} blending complete! Saved to {output_filename}")
                    else:
                        print(f"Error: Failed to download inpainted image, status code: {image_response.status_code}")
                else:
                    print(f"Error: No output received from Replicate API")
                    
            except Exception as e:
                print(f"Error in Replicate inpainting: {e}")
                # Save the composite as fallback
                composite_image.save(output_filename)
                print(f"Saved composite image without inpainting as fallback")
    
    print(f"\nComplete high-resolution tileset generated in '{output_dir_highres}'!")
    
    # Apply pixelation if requested
    if apply_pixelation:
        print("Applying pixelation to all generated tiles...")
        pixelated_output_dir = os.path.join(PIXELATED_DIR, f"{theme_name}_pixelated")
        pixelate_tileset(
            output_dir_highres,  # Input only the theme-specific high-res tiles
            pixelated_output_dir,
            pixel_size=pixel_size,
            method=method,
            grid_size=grid_size
        )
        print(f"Pixelated tileset saved to {pixelated_output_dir}")

def generate_transition_prompt(texture1_name, texture2_name):
    """
    Generate a specialized transition prompt between two textures.
    
    If Anthropic API is available, uses Claude to generate a custom prompt.
    Otherwise, raises an error (except for stone-grass special case).
    
    Args:
        texture1_name: Name of the first texture
        texture2_name: Name of the second texture
        
    Returns:
        A detailed prompt for image generation
        
    Raises:
        RuntimeError: If Anthropic API is not available for non-special case prompts
    """
    # Check if we have special cases
    if ("stone" in texture1_name.lower() and "grass" in texture2_name.lower()) or \
       ("grass" in texture1_name.lower() and "stone" in texture2_name.lower()):
        # Return specialized stone-grass prompt
        return (
            f"Create a natural transition between {texture1_name} and {texture2_name}. "
            f"The {texture1_name} areas should maintain large flat elements with minimal interruptions. "
            f"The {texture2_name} area should have medium-sized elements with visible space between them. "
            f"At the transition zone: "
            f"1. Small elements of {texture2_name} growing between the gaps in {texture1_name} near the edge "
            f"2. Scattered debris and soil accumulating at the edges "
            f"3. Worn, weathered edges where textures meet "
            f"4. Some small elements of {texture2_name} partially overlapping the {texture1_name} "
            f"5. Gradual thinning of {texture2_name} as it approaches the {texture1_name} "
            f"Create a photorealistic, seamless transition with natural colors. "
            f"No blue, purple, or unnatural borders. "
            f"Use only earthy tones - natural colors that match both textures. "
            f"The transition should look completely natural, as if it formed over years of weather and use."
        )
    
    # If Anthropic client is available, use it
    if anthropic_client:
        try:
            prompt_template = f"""
            You are a terrain texture expert. Create a detailed prompt for AI image generation to blend {texture1_name} and {texture2_name} textures seamlessly.

            Your response should follow this exact format:
            1. One sentence: "Create a natural transition between [texture1] and [texture2]."
            2. One sentence describing key visual characteristics of {texture1_name} to maintain.
            3. One sentence describing key visual characteristics of {texture2_name} to maintain.
            4. Five numbered points (1-5) describing specific transition effects at the boundary, including:
               - How elements of {texture1_name} gradually change into {texture2_name}
               - Natural weathering or interaction effects between the textures
               - Small details that make the transition realistic (debris, partial elements, etc.)
               - How colors and textures blend at the boundary
               - How shadows or lighting would naturally occur at this transition
            5. Two sentences on maintaining photorealism and seamlessness.
            6. One sentence prohibiting unnatural artifacts (blue borders, sharp edges, etc.)
            7. One sentence specifying the color palette to use (based on the natural colors of both textures).
            8. One final sentence emphasizing the transition should look natural and weathered.

            Return ONLY the formatted prompt text without explanations, introductions or quotation marks.
            """
            
            print(f"Generating specialized prompt for {texture1_name}-{texture2_name} transition using Claude...")
            
            # Call Claude with the template
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.7,
                system="You are a terrain texture expert who creates detailed prompts for AI image generation.",
                messages=[{"role": "user", "content": prompt_template}]
            )
            
            # Extract the generated text from the response
            return response.content[0].text
            
        except Exception as e:
            print(f"Error generating prompt with Anthropic: {e}")
            raise RuntimeError(f"Failed to generate AI prompt using Anthropic API: {e}")
    
    # No fallback template - raise error if we don't have Anthropic client
    raise RuntimeError("Anthropic API key not available. Please set ANTHROPIC_API_KEY in .env file to use dynamic prompt generation.")

# Add a new command line option for specialized stone-grass blending
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a tileset with AI inpainting")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Original dual-texture tileset generation command
    dual_parser = subparsers.add_parser("dual", help="Generate a tileset by combining two textures")
    dual_parser.add_argument("primary_texture", help="Path to the primary texture image")
    dual_parser.add_argument("secondary_texture", help="Path to the secondary texture image")
    dual_parser.add_argument("mask_dir", help="Directory containing mask images")
    dual_parser.add_argument("--primary-name", help="Name of primary texture (for AI prompts)")
    dual_parser.add_argument("--secondary-name", help="Name of secondary texture (for AI prompts)")
    dual_parser.add_argument("--output-dir", default=FINAL_TILESET_DIR, help="Directory to save the final tileset")
    dual_parser.add_argument("--theme-name", default="custom", help="Name to use for the output files")
    
    # Inpainting options for dual mode
    dual_parser.add_argument("--line-width", type=int, default=10, help="Width of the boundary line to inpaint")
    
    # Pixelation options for dual mode
    dual_parser.add_argument("--pixelate", action="store_true", help="Apply pixelation to final tiles")
    dual_parser.add_argument("--pixel-size", choices=["small", "medium", "large", "custom"], default="medium",
                       help="Size of pixels - small (8x8), medium (16x16), large (32x32), or custom")
    dual_parser.add_argument("--grid-size", type=int, help="Custom grid size (when pixel-size is 'custom')")
    dual_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced",
                       help="Pixelation method - simple (grid only) or advanced (with color quantization)")
    
    # New single-texture tileset generation command
    single_parser = subparsers.add_parser("single", help="Generate a tileset using a single base texture and inpainting")
    single_parser.add_argument("base_texture", help="Path to the base texture image")
    single_parser.add_argument("mask_dir", help="Directory containing mask images")
    single_parser.add_argument("target_name", help="Name of the texture to generate in masked areas (for AI prompts)")
    single_parser.add_argument("--base-name", help="Name of base texture (for AI prompts)")
    single_parser.add_argument("--output-dir", default=FINAL_TILESET_DIR, help="Directory to save the final tileset")
    single_parser.add_argument("--theme-name", default="custom", help="Name to use for the output files")
    
    # Inpainting options for single mode
    single_parser.add_argument("--line-width", type=int, default=10, help="Width of the transition area to inpaint")
    
    # Pixelation options for single mode
    single_parser.add_argument("--pixelate", action="store_true", help="Apply pixelation to final tiles")
    single_parser.add_argument("--pixel-size", choices=["small", "medium", "large", "custom"], default="medium",
                       help="Size of pixels - small (8x8), medium (16x16), large (32x32), or custom")
    single_parser.add_argument("--grid-size", type=int, help="Custom grid size (when pixel-size is 'custom')")
    single_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced",
                       help="Pixelation method - simple (grid only) or advanced (with color quantization)")
    
    # Generic texture blending command (replaces specialized stone-grass)
    blend_parser = subparsers.add_parser("blend", help="Create specialized transitions between any two textures with AI")
    blend_parser.add_argument("texture1", help="Path to the first texture image")
    blend_parser.add_argument("texture2", help="Path to the second texture image")
    blend_parser.add_argument("mask_dir", help="Directory containing mask images")
    blend_parser.add_argument("--texture1-name", help="Name of first texture (for AI prompts)")
    blend_parser.add_argument("--texture2-name", help="Name of second texture (for AI prompts)")
    blend_parser.add_argument("--output-dir", default=FINAL_TILESET_DIR, help="Directory to save the final tileset")
    blend_parser.add_argument("--theme-name", help="Name to use for the output files (default: texture1_texture2_blend)")
    
    # Transition options
    blend_parser.add_argument("--boundary-width", type=int, default=20, help="Width of the transition boundary to inpaint")
    blend_parser.add_argument("--mask-file", help="Specific mask file to process (instead of all masks in the directory)")
    
    # Pixelation options for blend mode
    blend_parser.add_argument("--pixelate", action="store_true", help="Apply pixelation to final tiles")
    blend_parser.add_argument("--pixel-size", choices=["small", "medium", "large", "custom"], default="medium",
                       help="Size of pixels - small (8x8), medium (16x16), large (32x32), or custom")
    blend_parser.add_argument("--grid-size", type=int, help="Custom grid size (when pixel-size is 'custom')")
    blend_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced",
                       help="Pixelation method - simple (grid only) or advanced (with color quantization)")
    
    # Also add a standalone pixelate command
    pixelate_parser = subparsers.add_parser("pixelate", help="Apply pixelation to existing tiles")
    pixelate_parser.add_argument("input_dir", help="Path to the directory containing tiles to pixelate")
    pixelate_parser.add_argument("output_dir", help="Path to save the pixelated tiles")
    pixelate_parser.add_argument("--pixel-size", choices=["small", "medium", "large", "custom"], default="medium",
                       help="Size of pixels - small (8x8), medium (16x16), large (32x32), or custom")
    pixelate_parser.add_argument("--grid-size", type=int, help="Custom grid size (when pixel-size is 'custom')")
    pixelate_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced",
                       help="Pixelation method - simple (grid only) or advanced (with color quantization)")
    
    args = parser.parse_args()
    
    if args.command == "dual":
        generate_tileset(
            args.primary_texture,
            args.secondary_texture,
            args.mask_dir,
            args.primary_name,
            args.secondary_name,
            args.output_dir,
            args.pixelate,
            args.pixel_size,
            args.method,
            args.line_width,
            args.theme_name,
            args.grid_size if args.pixel_size == "custom" else None
        )
    elif args.command == "single":
        generate_single_texture_tileset(
            args.base_texture,
            args.mask_dir,
            args.base_name,
            args.target_name,
            args.output_dir,
            args.pixelate,
            args.pixel_size,
            args.method,
            args.line_width,
            args.theme_name,
            args.grid_size if args.pixel_size == "custom" else None
        )
    elif args.command == "blend":
        # Use the new generic blend_textures function
        blend_textures(
            args.texture1,
            args.texture2,
            args.mask_dir,
            args.output_dir,
            args.texture1_name,
            args.texture2_name,
            args.pixelate,
            args.pixel_size,
            args.method,
            args.boundary_width,
            args.theme_name,
            args.grid_size if args.pixel_size == "custom" else None,
            args.mask_file
        )
    elif args.command == "pixelate":
        # Direct pixelation of existing tiles
        pixelate_tileset(
            args.input_dir,
            args.output_dir,
            args.pixel_size,
            args.method,
            args.grid_size if args.pixel_size == "custom" else None
        )
    else:
        # Default to dual mode for backward compatibility
        if len(sys.argv) >= 4:
            # Assume the first three arguments are primary_texture, secondary_texture, and mask_dir
            generate_tileset(
                sys.argv[1],  # primary_texture
                sys.argv[2],  # secondary_texture
                sys.argv[3],  # mask_dir
                None,  # primary_name
                None,  # secondary_name
                FINAL_TILESET_DIR,  # output_dir
                False,  # pixelate
                "medium",  # pixel_size
                "advanced",  # method
                10,  # line_width
                "custom"  # theme_name
            )
        else:
            parser.print_help()

