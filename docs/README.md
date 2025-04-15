# Tileset Generator

A modular tool for generating pixel-perfect, seamless tilesets for game development using AI-powered texture generation and blending.

## Project Structure

```
tileset_generator/
├── src/                    # Source code
│   ├── tileset_generator.py      # Main tileset generation tool
│   ├── pixel_processor.py        # Pixel art processing
│   ├── generate_base_tiles.py    # Base texture generation
│   └── background_tileset_agent.py
├── masks/                  # Mask files for processing
├── outputted_tilesets/     # Generated tilesets
├── output/                 # Additional output files
│   └── base_textures/     # Generated base textures
├── config/                 # Configuration files
│   ├── requirements.txt    # Python dependencies
│   └── .env.example       # Example environment variables
└── docs/                  # Documentation
```

## Examples

### Generated Base Textures
Here are some examples of AI-generated base textures:

| Ground Textures | Wall Textures |
|----------------|---------------|
| ![Grass Ground](../readme_examples/generated_tiles/grass_base_ground_tile.webp) | ![Crystal Wall](../readme_examples/generated_tiles/crystal_formation_wall_tile.webp) |
| ![Void Ground](../readme_examples/generated_tiles/void_ground_detailed_ground_tile.webp) | ![Energy Barrier](../readme_examples/generated_tiles/energy_barrier_wall_tile.webp) |
| ![Tech Floor](../readme_examples/generated_tiles/tech_floor_detailed_ground_tile.webp) | |

### Pixelated Examples
Examples of different textures after pixel art processing (32x32):

| | | |
|:---:|:---:|:---:|
| ![Ancient Ruins](../readme_examples/generated_tiles/pixelated_examples/ancient_ruins_wall_32x32_processed.png)<br>Ancient Ruins | ![Crystal Ground](../readme_examples/generated_tiles/pixelated_examples/crystal_ground_detailed_ground_32x32_processed.png)<br>Crystal Ground | ![Metal Grate](../readme_examples/generated_tiles/pixelated_examples/metal_grate_path_32x32_processed.png)<br>Metal Grate |
| ![Wooden Boardwalk](../readme_examples/generated_tiles/pixelated_examples/wooden_boardwalk_path_32x32_processed.png)<br>Wooden Boardwalk | | |

### Blended Tileset Examples
Examples of generated tilesets showing transitions between different textures:

#### Grass Ground to Cobblestone Path (8x8)
![Grass to Cobblestone](../readme_examples/outputted_tilesets/pixelated/grass%20ground_cobblestone%20path_blend_pixelated/left_border_grass%20ground_cobblestone%20path_blend_8x8_processed.png)

#### Tech Floor to Energy Barrier (8x8)
![Tech to Energy](../readme_examples/outputted_tilesets/pixelated/tech%20floor_energy%20barrier_blend_pixelated/4_corners_tech%20floor_energy%20barrier_blend_8x8_processed.png)

#### Void Ground to Crystal Wall (8x8)
![Void to Crystal](../readme_examples/outputted_tilesets/pixelated/void%20ground_crystal%20wall_blend_pixelated/left_corner_inside_void%20ground_crystal%20wall_blend_8x8_processed.png)

## Prerequisites

Before setting up the project, you'll need:
1. Python 3.x
2. Homebrew (for macOS users)
3. At least one of the following API keys:
   - Anthropic API key
   - GetImg API key
   - Stability AI API key
   - Replicate API token
   - ImgBB API key

## Setup

1. Install SDL2 dependencies (required for pygame):
```bash
# On macOS
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r config/requirements.txt
```

4. Create and configure your environment variables:
```bash
# Copy the example environment file
cp config/.env.example .env

# Edit .env with your API keys
# You need at least one of these:
ANTHROPIC_API_KEY=your_anthropic_api_key_here
RETRODIFFUSION_API_KEY=your_retrodiffusion_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
GETIMG_API_KEY=your_getimg_api_key_here
REPLICATE_API_TOKEN=your_replicate_api_token_here
IMGBB_API_KEY=your_imgbb_api_key_here
```

## Usage

The tileset generator provides several commands for different aspects of tileset generation:

### 1. Generate Base Textures
```bash
# Generate a single terrain texture
python src/generate_base_tiles.py terrain "terrain_type" [options] --output output/base_textures/name.webp

# Generate multiple textures based on game description
python src/generate_base_tiles.py game "game description" "game theme" --output output/base_textures/name.webp
```

#### Terrain Generation Options
- `terrain_type`: Descriptive name of the terrain (e.g., 'grass', 'forest', 'desert')
- `--tile-type {ground,path,wall}`: Type of tile to generate
  - `ground`: For base terrain textures
  - `path`: For walkable paths/roads
  - `wall`: For vertical surfaces
- `--output`: Custom output filename (defaults to terrain_type_tile.webp)
- `--no-pixel-art`: Skip pixel art generation
- `--method {simple,advanced}`: Pixel art processing method
  - `simple`: Grid-based pixelation
  - `advanced`: Includes color quantization
- `--pixel-size {small,medium,large}`: Size of pixels
  - `small`: 8x8 pixels
  - `medium`: 16x16 pixels
  - `large`: 32x32 pixels

Example:
```bash
# Generate a grass texture with medium-sized pixels
python src/generate_base_tiles.py terrain "grass" --tile-type ground --pixel-size medium
```

### 2. Generate Tilesets
```bash
# Create specialized transitions between textures
python src/tileset_generator.py blend texture1.png texture2.png masks/ [options]

# Apply pixelation to existing tiles
python src/tileset_generator.py pixelate input_dir/ output_dir/
```

#### Blend Options
- `--pixelate`: Apply pixelation to final tiles
- `--pixel-size`: Choose from small (8x8), medium (16x16), or large (32x32)
- `--method`: Choose pixelation method (simple or advanced)
- `--theme-name`: Custom name for the output files
- `--boundary-width`: Width of transition areas (default: 20)
- `--grid-size`: Custom grid size for pixelation

Example:
```bash
# Generate a cyberpunk tileset with small pixels
python src/tileset_generator.py blend output/base_textures/cyber_ground.webp output/base_textures/cyber_wall.webp masks --pixelate --pixel-size small --method advanced --theme-name cyber_blend
```