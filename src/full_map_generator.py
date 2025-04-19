import sys
import os
from PIL import Image

# Since we're now in the src directory, we don't need to add it to the path again
# We just need to make sure the current directory is in the path
if '' not in sys.path:
    sys.path.append('')

from tileset_generator import blend_textures, pixelate_tileset

# Get the base directory (parent of the src directory)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths using absolute paths
texture1_path = os.path.join(base_dir, "fullmap", "floor.png")
texture2_path = os.path.join(base_dir, "fullmap", "wall.png")
mask_dir = os.path.join(base_dir, "fullmap")
output_dir = os.path.join(base_dir, "fullmap")
mask_file = "fullmap.png"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the mask image to get dimensions
mask_image = Image.open(os.path.join(mask_dir, mask_file))
width, height = mask_image.size

# Tile the floor and wall textures with a specified number of tiles across
def tile_texture(texture_path, width, height, tiles_across):
    texture = Image.open(texture_path)
    tile_width = width // tiles_across
    tile_height = int(tile_width * (texture.height / texture.width))
    tiled_texture = Image.new('RGBA', (width, height))
    scaled_texture = texture.resize((tile_width, tile_height), Image.LANCZOS)
    for x in range(0, width, tile_width):
        for y in range(0, height, tile_height):
            tiled_texture.paste(scaled_texture, (x, y))
    return tiled_texture

# Create tiled textures with 7 tiles across
tiles_across = 10
tiled_floor = tile_texture(texture1_path, width, height, tiles_across)
tiled_wall = tile_texture(texture2_path, width, height, tiles_across)

# Save tiled textures temporarily
tiled_floor_path = os.path.join(mask_dir, "tiled_floor.png")
tiled_wall_path = os.path.join(mask_dir, "tiled_wall.png")
tiled_floor.save(tiled_floor_path)
tiled_wall.save(tiled_wall_path)

# Call the blend_textures function
blend_textures(
    tiled_floor_path,
    tiled_wall_path,
    mask_dir,
    output_dir=output_dir,
    texture1_name="floor",
    texture2_name="wall",
    apply_pixelation=False,
    pixel_size="custom",
    method="advanced",
    boundary_width=3,
    theme_name="floor_wall_blend",
    mask_file=mask_file,
    grid_size=2
)

# Ensure the output image matches these dimensions
highres_output_path = os.path.join(output_dir, "highres/floor_wall_blend/fullmap_floor_wall_blend.png")
output_image = Image.open(highres_output_path)
output_image = output_image.resize((width, height), Image.LANCZOS)
output_image.save(highres_output_path)

print(f"High-res output dimensions: {output_image.size}")

# Pixelate the high-res image and save to fullmap folder
pixelated_output_path = os.path.join(output_dir, "fullmap_floor_wall_blend_2x2_processed.png")
pixelate_tileset(
    os.path.dirname(highres_output_path),
    os.path.dirname(pixelated_output_path),
    pixel_size="custom",
    method="advanced",
    grid_size=2
)

print(f"Pixelated output saved to: {pixelated_output_path}")