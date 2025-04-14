#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import json
from pathlib import Path

def load_config(config_path):
    """
    Load game configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Tuple of (game_description, game_theme)
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config['game_description'], config['game_theme']
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None, None

def generate_base_tiles(game_description, game_theme):
    """
    Calls generate_base_tiles.py to generate base terrain tiles.
    
    Args:
        game_description: Description of the game
        game_theme: Theme of the game
        
    Returns:
        List of paths to generated tile files
    """
    print(f"Generating base tiles for game: '{game_description}' with theme: '{game_theme}'")
    
    # Import the generate_base_tiles module
    sys.path.append(os.getcwd())
    import generate_base_tiles
    
    # Call the function to generate terrains (with pixelation disabled)
    output_files, _ = generate_base_tiles.generate_terrains_for_game(
        game_description, 
        game_theme, 
        generate_pixel_art=False
    )
    
    print(f"Generated {len(output_files)} base terrain tiles")
    return output_files

def generate_tileset_blends(base_tile_paths, mask_dir="masks"):
    """
    Uses tileset_generator.py to blend the generated base tiles with pixelation.
    
    Args:
        base_tile_paths: List of paths to base terrain tiles
        mask_dir: Directory containing mask images
        
    Returns:
        Paths to directories containing the generated tilesets
    """
    print(f"Generating blended tilesets from {len(base_tile_paths)} base tiles")
    
    # Import the tileset_generator module
    import tileset_generator
    
    # Create masks directory if it doesn't exist
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        print(f"Created masks directory: {mask_dir}")
        print("WARNING: You need to add mask images to this directory!")
    
    output_dirs = []
    
    # Generate all possible combinations of terrain transitions
    for i in range(len(base_tile_paths)):
        for j in range(i+1, len(base_tile_paths)):
            texture1_path = base_tile_paths[i]
            texture2_path = base_tile_paths[j]
            
            # Extract texture names from file paths
            texture1_name = Path(texture1_path).stem.replace('_tile', '')
            texture2_name = Path(texture2_path).stem.replace('_tile', '')
            
            theme_name = f"{texture1_name}_{texture2_name}_blend"
            
            # Create a specific high-res output directory for this combination
            highres_output_dir = os.path.join(tileset_generator.OUTPUT_DIR, "highres", theme_name)
            os.makedirs(highres_output_dir, exist_ok=True)
            
            # Create pixelated output directory
            pixelated_output_dir = os.path.join(tileset_generator.PIXELATED_DIR, f"{theme_name}_pixelated")
            
            print(f"\nBlending textures: {texture1_name} + {texture2_name}")
            
            # Call the blend_textures function with the specific output directory
            tileset_generator.blend_textures(
                texture1_path,
                texture2_path,
                mask_dir,
                output_dir=os.path.dirname(os.path.dirname(highres_output_dir)),  # Use the parent of highres dir
                texture1_name=texture1_name,
                texture2_name=texture2_name,
                apply_pixelation=True,
                pixel_size="custom",
                method="advanced",
                grid_size=8,  # Changed from 6 to 8 for larger pixel size
                theme_name=theme_name
            )
            
            output_dirs.append(pixelated_output_dir)
    
    return output_dirs

def cleanup_directories():
    """
    Cleans up temporary and unused directories after tile generation.
    """
    import shutil
    
    # List of directories to clean up
    cleanup_dirs = [
        "temp",
        os.path.join("outputted_tilesets", "final_tileset")
    ]
    
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up directory: {dir_path}")
            except Exception as e:
                print(f"Warning: Could not remove directory {dir_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate game background tiles and blended tilesets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to game configuration JSON file")
    group.add_argument("--description", help="Description of the game")
    parser.add_argument("--theme", help="Theme of the game (required if using --description)", required=False)
    parser.add_argument("--mask-dir", default="masks", help="Directory containing mask images (default: masks)")
    
    args = parser.parse_args()
    
    # Get game description and theme either from config or command line
    if args.config:
        game_description, game_theme = load_config(args.config)
        if not game_description or not game_theme:
            sys.exit(1)
    else:
        if not args.theme:
            parser.error("--theme is required when using --description")
        game_description = args.description
        game_theme = args.theme
    
    # Step 1: Generate base tiles
    base_tile_paths = generate_base_tiles(game_description, game_theme)
    
    if not base_tile_paths:
        print("Failed to generate base tiles!")
        sys.exit(1)
    
    # Step 2: Generate blended tilesets
    output_dirs = generate_tileset_blends(base_tile_paths, args.mask_dir)
    
    # Summary
    print("\n=== Background Tile Generation Complete ===")
    print(f"Base tiles: {len(base_tile_paths)}")
    print("Base tile locations:")
    for path in base_tile_paths:
        print(f"  - {path}")
    
    print(f"\nBlended tilesets: {len(output_dirs)}")
    print("Tileset locations:")
    for path in output_dirs:
        print(f"  - {path}")
    
    # Step 3: Clean up temporary and unused directories
    cleanup_directories()
    
    print("\nAll done! You can now use these tiles in your game.")

if __name__ == "__main__":
    main()