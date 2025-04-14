import os
import requests
import argparse
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import anthropic
import replicate
import numpy as np
from pixel_processor import SeamlessTileProcessor
import json
import sys

# Load API keys from .env
load_dotenv()

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Create output directories
OUTPUT_DIR = "generated_tiles"
PIXEL_OUTPUT_DIR = "pixelated_tiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PIXEL_OUTPUT_DIR, exist_ok=True)

# Initialize the tile processor - updated for chunkier pixel effect
tile_processor = SeamlessTileProcessor(
    num_colors=16,        # Number of colors in the palette
    grid_size=16,         # Increased from 8 to 16 for chunkier pixels
    scale_factor=1,       # No upscaling by default
    force_16x16=False     # Don't force to 16x16, preserve original dimensions
)

def generate_structured_tile_prompt(base_terrain, tile_type="ground"):
    """
    Uses Claude to generate tile_subject, tile_contents, and color_palette for a terrain description.
    
    Args:
        base_terrain: Base terrain type (e.g., 'dirt', 'rock', 'moss')
        tile_type: Type of tile to generate ('ground', 'path', or 'wall')
    """
    # Define complexity levels for different tile types
    tile_instructions = {
        "ground": {
            "description": (
                "You are designing an extremely subtle and simple ground texture tile for a top-down game.\n"
                "Given a terrain type, return the following JSON with NATURAL descriptions:\n"
                "{\n"
                "  'tile_subject': 'basic ground type (2-3 words)',\n"
                "  'tile_contents': 'description of minimal scattered details',\n"
                "  'color_palette': 'base color with very subtle variations'\n"
                "}\n"
                "CRITICAL RULES:\n"
                "1. Focus on EXTREMELY subtle, natural ground textures\n"
                "2. Details must be VERY sparse and barely noticeable\n"
                "3. Use minimal color variation from base color\n"
                "4. NO distinct shapes or patterns\n"
                "5. Think background texture, not features\n"
                "6. NO borders or edges\n"
                "7. Details should be extremely sparse\n\n"
                "Example for 'dirt':\n"
                "{\n"
                "  'tile_subject': 'bare earth surface',\n"
                "  'tile_contents': 'tiny soil marks',\n"
                "  'color_palette': 'earth brown with minimal variation'\n"
                "}"
            )
        },
        "path": {
            "description": (
                "You are designing a simple path material texture tile for a top-down game.\n"
                "Given a terrain type, return the following JSON with NATURAL descriptions:\n"
                "{\n"
                "  'tile_subject': 'basic path material (2-3 words)',\n"
                "  'tile_contents': 'description of visible material details',\n"
                "  'color_palette': 'base color with moderate variations'\n"
                "}\n"
                "CRITICAL RULES:\n"
                "1. Focus on the actual path MATERIAL texture (dirt, stone, etc)\n"
                "2. Details should be material-specific and evenly distributed\n"
                "3. Use moderate color variation within the material theme\n"
                "4. Simple material patterns allowed but keep them natural\n"
                "5. Think close-up ground material texture\n"
                "6. NO borders or edges\n"
                "7. Must fill entire frame with the material\n\n"
                "Example for 'stone':\n"
                "{\n"
                "  'tile_subject': 'packed gravel surface',\n"
                "  'tile_contents': 'small embedded stones',\n"
                "  'color_palette': 'gray stone with earth tones'\n"
                "}"
            )
        },
        "wall": {
            "description": (
                "You are designing a ZOOMED-IN wall texture tile for a top-down game.\n"
                "Given a terrain type, return the following JSON with NATURAL descriptions:\n"
                "{\n"
                "  'tile_subject': 'detailed wall type (2-3 words)',\n"
                "  'tile_contents': 'description of prominent surface details',\n"
                "  'color_palette': 'base color with rich variations'\n"
                "}\n"
                "CRITICAL RULES:\n"
                "1. Focus on EXTREME CLOSE-UP of just 1-2 massive stones\n"
                "2. Each stone should take up nearly half the frame\n"
                "3. Use rich color variation within theme\n"
                "4. Think ZOOMED IN view of individual blocks\n"
                "5. This is NOT a full wall pattern - just huge pieces\n"
                "6. NO borders or edges\n"
                "7. NO small details or mortar lines\n\n"
                "Example for 'stone':\n"
                "{\n"
                "  'tile_subject': 'giant stone block',\n"
                "  'tile_contents': 'massive weathered surface',\n"
                "  'color_palette': 'gray stone with deep shadows'\n"
                "}\n\n"
                "Example for 'grass':\n"
                "{\n"
                "  'tile_subject': 'huge mossy block',\n"
                "  'tile_contents': 'massive stone with grass patches',\n"
                "  'color_palette': 'weathered stone with green growth'\n"
                "}"
            )
        }
    }

    # Get the appropriate instruction set
    if tile_type not in tile_instructions:
        print(f"Warning: Unknown tile type '{tile_type}', defaulting to 'ground'")
        tile_type = "ground"

    tile_instruction = tile_instructions[tile_type]["description"]

    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=300,
        temperature=0.6,
        system=tile_instruction,
        messages=[{"role": "user", "content": f"{base_terrain} {tile_type}"}]
    )

    try:
        content = response.content[0].text.strip()
        tile_data = json.loads(content.replace("'", '"'))
        return tile_data
    except Exception as e:
        print("Failed to parse tile data:", e)
        return None

def build_final_tile_prompt(tile_subject, tile_contents, color_palette, tile_type="ground"):
    """
    Build the final prompt for image generation based on tile type and components.
    
    Args:
        tile_subject: Basic description of the tile
        tile_contents: Description of the details/features
        color_palette: Color scheme to use
        tile_type: Type of tile ('ground', 'path', or 'wall')
    """
    # Define complexity levels for different tile types
    prompt_templates = {
        "ground": {
            "detail_level": "extremely subtle and minimal",
            "requirements": [
                "The base texture must be VERY subtle and natural-looking",
                "Any details must be TINY and EXTREMELY SPARSE",
                "NO distinct shapes or patterns whatsoever",
                "Texture should be almost imperceptible at edges",
                "Think flat, background texture only"
            ]
        },
        "path": {
            "detail_level": "simple but visible",
            "requirements": [
                "The base texture must fill the ENTIRE frame with the path material",
                "Details should be noticeable but kept simple and evenly distributed",
                "Focus on the material itself (dirt, stone, etc) not the surroundings",
                "Texture should blend seamlessly at edges",
                "Think close-up ground material, not aerial view"
            ]
        },
        "wall": {
            "detail_level": "massive and zoomed-in",
            "requirements": [
                "The texture must show just 1-2 HUGE stone blocks",
                "Each block should take up nearly half the frame",
                "Think extreme close-up of individual stones",
                "This is NOT a full wall pattern",
                "Focus on massive individual elements only"
            ]
        }
    }

    # Get the appropriate template
    if tile_type not in prompt_templates:
        print(f"Warning: Unknown tile type '{tile_type}', defaulting to 'ground'")
        tile_type = "ground"
    
    template = prompt_templates[tile_type]
    
    # Build the prompt
    requirements = "\n".join(f"- {req}" for req in template["requirements"])
    
    return (
        f"A seamless top-down {tile_type} texture tile showing {tile_subject}. "
        f"CRITICAL: Create a {template['detail_level']} texture that will tile seamlessly. "
        f"\nCRITICAL TEXTURE REQUIREMENTS:\n"
        f"{requirements}\n"
        f"- NO borders, frames, or distinct edges\n\n"
        f"IMPORTANT RULES:\n"
        f"- Use {color_palette}\n"
        f"- Add {tile_contents}\n"
        f"- Ensure texture fades naturally at edges for seamless tiling\n"
        f"The final result should look like a natural {tile_type} texture that tiles seamlessly."
    )

def process_tile_to_pixel_art(input_image, base_name, method="simple", pixel_size="medium"):
    """
    Process a generated tile image into pixel art.
    
    Args:
        input_image: PIL Image object of the generated tile
        base_name: Base name for the output file
        method: Processing method - "simple" for grid only, "advanced" for full processing
        pixel_size: Size of pixels - "small" (8x8), "medium" (16x16), or "large" (32x32)
        
    Returns:
        Path to the processed pixel art file
    """
    # Save the original for reference
    raw_output = f"{PIXEL_OUTPUT_DIR}/raw_{base_name}.png"
    input_image.save(raw_output)
    
    # Determine grid size based on pixel_size parameter
    if pixel_size == "small":
        grid_size = 8
        size_label = "8x8"
    elif pixel_size == "medium":
        grid_size = 16
        size_label = "16x16"
    elif pixel_size == "large":
        grid_size = 32
        size_label = "32x32"
    else:
        grid_size = 16  # Default
        size_label = "16x16"
    
    # Create a temporary processor with the specified grid size
    temp_processor = SeamlessTileProcessor(
        num_colors=16,
        grid_size=grid_size,
        scale_factor=1,
        force_16x16=False
    )
    
    if method == "simple":
        print(f"Applying {size_label} grid pixelation effect...")
        
        # Simple pixelation: Just apply grid
        pixelated = temp_processor.pixelate(input_image)
        
        # Save the pixelated version
        pixelated_output = f"{PIXEL_OUTPUT_DIR}/{base_name}_{size_label}.png"
        pixelated.save(pixelated_output)
        
        print(f"Saved {size_label} pixelated version to {pixelated_output}")
        return pixelated_output
        
    else:  # Advanced processing
        print(f"Applying advanced pixel art processing with {size_label} pixels...")
        
        # Override the grid_size in the processor
        temp_processor.grid_size = grid_size
        
        # Full processing with color quantization and seamless enhancement
        pixel_art = temp_processor.process_texture(
            input_image,
            output_path=f"{PIXEL_OUTPUT_DIR}/{base_name}_{size_label}_processed.png",
            make_seamless=True,
            harmonize=True
        )
        
        print(f"Saved processed pixel art to {PIXEL_OUTPUT_DIR}/{base_name}_{size_label}_processed.png")
        return f"{PIXEL_OUTPUT_DIR}/{base_name}_{size_label}_processed.png"

def generate_texture_from_theme(theme, tile_type="ground", seed=None):
    """
    Generate a texture based on theme and tile type.
    
    Args:
        theme: Base theme/terrain type (e.g., 'dirt', 'rock', 'moss')
        tile_type: Type of tile to generate ('ground', 'path', or 'wall')
        seed: Optional seed for consistent variations
    """
    tile_data = generate_structured_tile_prompt(theme, tile_type)
    if not tile_data:
        return None
        
    # Skip validation for path tiles, only validate ground and wall
    if tile_type != "path" and not validate_tile_description(tile_data, tile_type):
        print("Generated description too complex for tile type, retrying...")
        return generate_texture_from_theme(theme, tile_type, seed)

    prompt = build_final_tile_prompt(
        tile_data['tile_subject'],
        tile_data['tile_contents'],
        tile_data['color_palette'],
        tile_type
    )

    print(f"Final full prompt:\n{prompt}\n")

    # Use the black-forest-labs/flux-1.1-pro-ultra model
    input_params = {
        "prompt": prompt,
        "aspect_ratio": "1:1",  # Keeping 1:1 for tiles
        "output_format": "png"  # Changed from webp to png as supported by this model
    }
    
    # Add seed if provided
    if seed is not None:
        input_params["seed"] = seed
        print(f"Using seed: {seed}")

    output = replicate.run(
        "black-forest-labs/flux-1.1-pro-ultra",
        input=input_params
    )

    if not output:
        print("No output from Replicate")
        return None

    # Handle output based on its type
    original_image = None
    try:
        if isinstance(output, str):
            # URL string case
            response = requests.get(output)
            if response.status_code != 200:
                print("Failed to download image")
                return None
            original_image = Image.open(BytesIO(response.content)).convert("RGBA")
        elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], str):
            # List of URLs case
            response = requests.get(output[0])
            if response.status_code != 200:
                print("Failed to download image")
                return None
            original_image = Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            # Try to handle FileOutput or other object types
            if hasattr(output, 'download'):
                image_data = output.download()
                original_image = Image.open(BytesIO(image_data)).convert("RGBA")
            elif hasattr(output, 'read'):
                original_image = Image.open(output).convert("RGBA")
            else:
                print(f"Cannot process output of type: {type(output)}")
                return None
    except Exception as e:
        print(f"Error processing output: {e}")
        return None
        
    return original_image

def identify_terrain_types(game_description, game_theme):
    """
    Uses Claude to identify suitable terrain types based on game description and theme.
    Returns a list of 3 terrain types with their appropriate type classification.
    Raises an exception if unable to identify valid terrain types.
    """
    terrain_instruction = (
        "You are a game design assistant helping to identify appropriate textures for a top-down game.\n"
        "Given a game description and theme, identify exactly 3 distinct texture types that would be suitable for the game.\n"
        "For each texture, determine if it should be used as a ground (basic walkable surface), path (special surface), or wall (vertical/blocking element).\n"
        "Return your answer as a JSON array of objects, with each object containing:\n"
        "- 'name': A concise texture description IN SINGULAR FORM (not plural)\n"
        "- 'type': Either 'ground', 'path', or 'wall'\n"
        "- 'purpose': Brief explanation of its role in the game\n\n"
        "IMPORTANT RULES:\n"
        "1. Focus ONLY on textures with FEW, LARGE elements - these are BACKGROUND tiles\n"
        "2. Each texture should have just 3-5 major features that are LARGER than normal\n"
        "3. Think in terms of SIMPLE, CLEAN textures that won't compete with foreground elements\n"
        "4. Keep texture names simple, focusing on a few key oversized elements\n"
        "5. These will be used as BACKGROUND tiles that other game elements will be placed on top of\n"
        "6. Textures MUST be designed for TRULY SEAMLESS tiling with NO BORDERS or EDGES\n"
        "7. ALWAYS USE SINGULAR FORM, not plural\n\n"
        "Example response for a fantasy RPG:\n"
        "[\n"
        "    {\n"
        "        \"name\": \"worn stone block\",\n"
        "        \"type\": \"ground\",\n"
        "        \"purpose\": \"Basic walkable dungeon floor\"\n"
        "    },\n"
        "    {\n"
        "        \"name\": \"mossy cobblestone\",\n"
        "        \"type\": \"path\",\n"
        "        \"purpose\": \"Special pathway through garden areas\"\n"
        "    },\n"
        "    {\n"
        "        \"name\": \"cracked granite slab\",\n"
        "        \"type\": \"wall\",\n"
        "        \"purpose\": \"Imposing dungeon barrier\"\n"
        "    }\n"
        "]\n\n"
        "Return ONLY the JSON array, nothing else. Do not include any explanations or text outside the JSON array."
    )

    prompt = f"Game Description: {game_description}\nGame Theme: {game_theme}"
    
    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        temperature=0.7,
        system=terrain_instruction,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.content[0].text.strip()
    print(f"Raw response from Claude: {content}")
    
    try:
        terrain_types = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Claude's response as JSON: {e}")
        raise ValueError(f"Could not parse terrain types from Claude's response: {content}")
    
    if not isinstance(terrain_types, list):
        raise TypeError(f"Expected a list of terrain types, got {type(terrain_types)}")
    
    if len(terrain_types) < 3:
        raise ValueError(f"Expected at least 3 terrain types, only got {len(terrain_types)}: {terrain_types}")
    
    # Validate each terrain type has required fields
    for terrain in terrain_types:
        if not isinstance(terrain, dict):
            raise TypeError(f"Expected terrain to be a dictionary, got {type(terrain)}")
        if 'name' not in terrain or 'type' not in terrain:
            raise ValueError(f"Terrain missing required fields: {terrain}")
        if terrain['type'] not in ['ground', 'path', 'wall']:
            raise ValueError(f"Invalid terrain type '{terrain['type']}' in {terrain}")
    
    # Take exactly 3 terrain types
    return terrain_types[:3]

def generate_terrains_for_game(game_description, game_theme, method="advanced", pixel_size="medium", generate_pixel_art=True, terrain_type=None):
    """
    Identifies and generates 3 ground texture tiles based on game description and theme.
    Returns a list of output filenames and their descriptions.
    """
    print(f"Game Description: {game_description}")
    print(f"Game Theme: {game_theme}")
    print("Identifying suitable textures...")
    
    try:
        terrains = identify_terrain_types(game_description, game_theme)
        print("\nIdentified textures:")
        for terrain in terrains:
            print(f"- {terrain['name']} ({terrain['type']}): {terrain['purpose']}")
    except Exception as e:
        print(f"ERROR: Failed to identify textures: {e}")
        sys.exit(1)
    
    output_files = []
    pixel_art_files = []
    descriptions = []
    
    for terrain in terrains:
        print(f"\nGenerating tile for: {terrain['name']} as {terrain['type']}")
        image = generate_texture_from_theme(terrain['name'], terrain['type'])
        
        if image:
            # Save the original high-res version
            base_name = terrain['name'].lower().replace(' ', '_')
            output_filename = f"{OUTPUT_DIR}/{base_name}_tile.webp"
            image.save(output_filename)
            print(f"Tile saved to {output_filename}")
            output_files.append(output_filename)
            descriptions.append(terrain['purpose'])
            
            # Process into pixel art if enabled
            if generate_pixel_art:
                pixel_art_filename = process_tile_to_pixel_art(
                    image, 
                    base_name, 
                    method=method,
                    pixel_size=pixel_size
                )
                pixel_art_files.append(pixel_art_filename)
        else:
            print(f"Failed to generate tile for {terrain['name']}")
    
    return output_files, descriptions

def validate_tile_description(tile_data, tile_type="ground"):
    """
    Validate that the tile description meets complexity requirements for the given tile type.
    
    Args:
        tile_data: Dictionary containing tile description components
        tile_type: Type of tile ('ground', 'path', or 'wall')
    """
    # Define complexity limits for different tile types
    complexity_limits = {
        "ground": {
            "subject_words": 3,
            "contents_words": 4,
            "color_words": 2,
            "forbidden_words": ['detailed', 'pattern', 'complex', 'intricate', 'varied'],
            "allowed_words": ['subtle', 'minimal', 'simple', 'bare']
        },
        "path": {
            "subject_words": 4,
            "contents_words": 5,
            "color_words": 4,
            "forbidden_words": ['complex', 'intricate', 'ornate', 'scene', 'view'],
            "allowed_words": ['packed', 'worn', 'weathered', 'basic', 'surface', 'material']
        },
        "wall": {
            "subject_words": 5,
            "contents_words": 6,
            "color_words": 4,
            "forbidden_words": ['subtle', 'minimal', 'bare'],
            "allowed_words": ['detailed', 'structured', 'defined', 'prominent']
        }
    }
    
    # Get appropriate limits
    if tile_type not in complexity_limits:
        print(f"Warning: Unknown tile type '{tile_type}', defaulting to 'ground'")
        tile_type = "ground"
    
    limits = complexity_limits[tile_type]
    
    # Check word counts
    if len(tile_data['tile_subject'].split()) > limits["subject_words"]:
        return False
    if len(tile_data['tile_contents'].split()) > limits["contents_words"]:
        return False
        
    # Check for forbidden words based on tile type
    description_text = ' '.join(tile_data.values()).lower()
    if any(word in description_text for word in limits["forbidden_words"]):
        return False
            
    # For ground tiles, ensure at least one allowed word is present
    if tile_type == "ground" and not any(word in description_text for word in limits["allowed_words"]):
        return False
            
    # Check color count
    color_words = tile_data['color_palette'].lower().count('shade') + \
                  tile_data['color_palette'].lower().count('color') + \
                  tile_data['color_palette'].lower().count('variation')
    if color_words > limits["color_words"]:
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate game tiles from terrain descriptions or game details")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subparser for single terrain generation
    terrain_parser = subparsers.add_parser("terrain", help="Generate a single terrain tile")
    terrain_parser.add_argument("terrain_type", help="The terrain type to generate (e.g., 'forest', 'desert')")
    terrain_parser.add_argument("--tile-type", choices=["ground", "path", "wall"], default="ground",
                               help="Type of tile to generate (affects detail level)")
    terrain_parser.add_argument("--output", default=None, help="Output filename (default: terrain_type_tile.webp)")
    terrain_parser.add_argument("--no-pixel-art", action="store_true", help="Skip pixel art generation")
    terrain_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced", 
                               help="Pixel art processing method - simple (grid only) or advanced (with color quantization)")
    terrain_parser.add_argument("--pixel-size", choices=["small", "medium", "large"], default="medium",
                               help="Size of pixels - small (8x8), medium (16x16), or large (32x32)")
    terrain_parser.add_argument("--seed", type=int, help="Seed for consistent variations")
    
    # Subparser for game-based generation
    game_parser = subparsers.add_parser("game", help="Generate multiple terrain tiles based on game description and theme")
    game_parser.add_argument("description", help="Description of the game")
    game_parser.add_argument("theme", help="Theme of the game (e.g., 'fantasy medieval', 'sci-fi')")
    game_parser.add_argument("--tile-types", nargs="+", choices=["ground", "path", "wall"], default=["ground"],
                            help="Types of tiles to generate for each terrain (affects detail level)")
    game_parser.add_argument("--no-pixel-art", action="store_true", help="Skip pixel art generation")
    game_parser.add_argument("--method", choices=["simple", "advanced"], default="advanced",
                            help="Pixel art processing method - simple (grid only) or advanced (with color quantization)")
    game_parser.add_argument("--pixel-size", choices=["small", "medium", "large"], default="medium",
                            help="Size of pixels - small (8x8), medium (16x16), or large (32x32)")
    
    args = parser.parse_args()
    
    if args.command == "terrain":
        # Process as a single terrain type
        print(f"Generating {args.tile_type} tile for terrain: {args.terrain_type}")
        image = generate_texture_from_theme(args.terrain_type, args.tile_type, args.seed)
        
        if image:
            # Save the original high-res version
            base_name = f"{args.terrain_type}_{args.tile_type}".lower().replace(' ', '_')
            output_filename = args.output or f"{OUTPUT_DIR}/{base_name}_tile.webp"
            image.save(output_filename)
            print(f"Tile saved to {output_filename}")
            
            # Process into pixel art by default, unless --no-pixel-art is specified
            if not args.no_pixel_art:
                pixel_art_filename = process_tile_to_pixel_art(
                    image, 
                    base_name, 
                    method=args.method,
                    pixel_size=args.pixel_size
                )
                print(f"Generated pixel art version: {pixel_art_filename}")
        else:
            print("Failed to generate tile")
    
    elif args.command == "game":
        # Process based on game description and theme
        generate_pixel_art = not args.no_pixel_art
        all_output_files = []
        all_pixel_art_files = []
        
        # Get terrain types first
        try:
            terrains = identify_terrain_types(args.description, args.theme)
            print(f"Successfully identified terrain types: {terrains}")
        except Exception as e:
            print(f"ERROR: Failed to identify terrain types: {e}")
            sys.exit(1)
        
        # Generate each terrain type with each specified tile type
        for terrain in terrains:
            for tile_type in args.tile_types:
                print(f"\nGenerating {tile_type} tile for terrain: {terrain['name']}")
                image = generate_texture_from_theme(terrain['name'], tile_type, args.seed)
                
                if image:
                    # Save the original high-res version
                    base_name = f"{terrain['name']}_{tile_type}".lower().replace(' ', '_')
                    output_filename = f"{OUTPUT_DIR}/{base_name}_tile.webp"
                    image.save(output_filename)
                    print(f"Tile saved to {output_filename}")
                    all_output_files.append(output_filename)
                    
                    # Process into pixel art if enabled
                    if generate_pixel_art:
                        pixel_art_filename = process_tile_to_pixel_art(
                            image, 
                            base_name, 
                            method=args.method,
                            pixel_size=args.pixel_size
                        )
                        all_pixel_art_files.append(pixel_art_filename)
                else:
                    print(f"Failed to generate {tile_type} tile for {terrain['name']}")
        
        return all_output_files, all_pixel_art_files
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
