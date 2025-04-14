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

### 3. Generate Complete Background Tilesets
```bash
# Using command line arguments
python src/background_tileset_agent.py --description "Your game description" --theme "your game theme"

# Using JSON configuration
python src/background_tileset_agent.py --config config/game_template.json
```

The background tileset agent automates the entire process by:
1. Generating base textures based on game description
2. Creating all necessary tile transitions
3. Applying pixelation and processing

#### JSON Configuration
Create a game template in `config/game_template.json`:
```json
{
    "game_description": "A dark cyberpunk game set in a neon-lit industrial complex with high-tech machinery and worn metal surfaces",
    "game_theme": "cyberpunk industrial"
}
```

#### Options
- `--config`: Path to game configuration JSON file
- `--description`: Game description (when not using config file)
- `--theme`: Game theme (required with --description)
- `--mask-dir`: Directory containing mask images (default: masks)

## Output Directories

- `outputted_tilesets/`: Main output directory for generated tilesets
  - `highres/`: High-resolution versions of the tiles
  - `final_tileset/`: Final processed tiles
  - `pixelated/`: Pixelated versions of the tiles
  - `temp/`: Temporary files used during generation

## Features

- AI-powered texture generation and blending
- Seamless tile transitions
- Multiple pixelation methods and sizes
- Support for various texture types (ground, path, wall)
- Color harmonization and quantization
- Customizable transition effects
- Support for batch processing with mask directories

## Tips

1. Start with generating base textures using `generate_base_tiles.py`
2. Use the generated textures with `tileset_generator.py` to create complete tilesets
3. Experiment with different pixelation settings for desired aesthetic
4. Use the `blend` command for specialized transitions between textures
5. Check the `masks/` directory for available transition patterns