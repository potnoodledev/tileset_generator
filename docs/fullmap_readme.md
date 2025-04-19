# Vibe Dungeon Map Generation System

A comprehensive system for generating and viewing complete dungeon maps using AI-powered texture generation, blending, and visualization.

## Overview

The Vibe Dungeon Map Generation System consists of two main components:
1. **Full Map Generator** (`full_map_generator.py`) - Processes and blends textures to create complete dungeon maps
2. **Vibe Dungeon Viewer** (`vibe_dungeon_viewer.py`) - Interactive viewer for generated maps with real-time theme-based texture generation

Together, these tools allow you to:
- Generate themed floor and wall textures using AI
- Blend them together into seamless dungeon maps
- View and navigate the generated maps in real-time
- Experiment with different visual themes through a simple interface

## Directory Structure

```
tileset_generator/
├── src/                            # Source code
│   ├── vibe_dungeon_viewer.py      # Interactive map viewer
│   ├── full_map_generator.py       # Map generation script
│   ├── tileset_generator.py        # Core tileset generation functions
│   ├── pixel_processor.py          # Pixel art processing
│   ├── generate_base_tiles.py      # Base texture generation with AI
│   └── background_tileset_agent.py # Agent for automated generation
├── fullmap/                        # Map-related files
│   ├── floor.png                   # Generated floor texture
│   ├── wall.png                    # Generated wall texture
│   ├── fullmap.png                 # Map layout mask
│   ├── tiled_floor.png             # Tiled floor texture
│   ├── tiled_wall.png              # Tiled wall texture
│   ├── fullmap_floor_wall_blend_2x2_processed.png  # Final processed map
│   └── highres/                    # High-resolution outputs
└── ...                             # Other project directories
```

## Full Map Generator

The `full_map_generator.py` script takes floor and wall textures and blends them together to create a complete dungeon map.

### How It Works

1. **Input Textures**: Uses floor and wall textures from `fullmap/floor.png` and `fullmap/wall.png`
2. **Mask-Based Blending**: Uses the mask file in `fullmap/fullmap.png` to define the layout
3. **Texture Tiling**: Tiles the input textures across the map dimensions
4. **Blending Process**: Creates smooth transitions between floor and wall textures
5. **Pixelation**: Applies pixel art-style processing to the blended map
6. **Output**: Generates both high-resolution and pixelated versions of the map

### Technical Details

- **Grid System**: Uses a customizable grid size (default: 2x2) for the pixel processing
- **Transition Boundaries**: Configurable boundary width for smoother transitions
- **Resolution Control**: Ensures consistent dimensions between input and output files

## Vibe Dungeon Viewer

The `vibe_dungeon_viewer.py` provides an interactive interface to view and generate new themed maps.

### Features

- **Real-time Navigation**: Pan around the map using arrow keys
- **Theme-Based Generation**: Enter a theme (e.g., "ancient ruins", "crystal cave") to generate matching textures
- **Progress Tracking**: Displays generation status and timing information
- **Intuitive Interface**: Simple UI with theme input and generation button

### How It Works

1. **Texture Generation**: Uses AI models through the `generate_texture_from_theme()` function to create themed textures
2. **Scripted Processing**: Calls the `full_map_generator.py` script to process and blend the textures
3. **Dynamic Reloading**: Automatically reloads the map once generation is complete
4. **Background Processing**: Performs generation in a background thread to keep the UI responsive

### Generation Process

1. User enters a theme in the input box
2. System generates both floor and wall textures based on the theme
3. Textures are saved to the fullmap directory
4. The full_map_generator script is called to blend them
5. The viewer loads and displays the new map once ready

### API Integration

The viewer requires API credentials for:
- **Anthropic API**: For language model-based generation guidance
- **Replicate API**: For image model-based texture generation

API keys should be configured in the `.env` file at the project root.

## Running the Tools

### Running the Map Generator Directly

```bash
cd src
python full_map_generator.py
```

This will generate a map using the existing floor.png and wall.png textures in the fullmap directory.

### Running the Interactive Viewer

```bash
cd src
python vibe_dungeon_viewer.py
```

Once the viewer is running:
1. Type a theme in the input box (top right)
2. Click the "GENERATE TILE" button or press Enter
3. Wait for the generation process to complete
4. Use arrow keys to navigate the new map

## Customization

### Customizing the Map Layout

To customize the map layout:
1. Edit the `fullmap/fullmap.png` mask file
   - White areas represent floor
   - Black areas represent walls
2. Run the generator or viewer to create a new map with your custom layout

### Customizing Generation Parameters

In `full_map_generator.py`:
- Adjust `tiles_across` to change the tiling density
- Modify `boundary_width` to adjust transition zones
- Change `grid_size` to alter pixelation level

## Integration with Other Tools

The fullmap system leverages the core tileset generation functionality but applies it to complete maps rather than individual tiles.

- **Base Texture Generation**: Uses the same AI-powered generation as the tileset tools
- **Pixelation Processing**: Applies consistent pixel art styling across all components
- **Blending Logic**: Uses enhanced blending algorithms optimized for larger maps

## Examples

| Theme | Description | Preview |
|-------|-------------|---------|
| Forest Ruins | Ancient stone structures overgrown with moss and vines | (Screenshot) |
| Crystal Cave | Glowing crystal formations with smooth stone floors | (Screenshot) |
| Tech Facility | Sleek metal floors with energy barrier walls | (Screenshot) |

## Troubleshooting

### Common Issues

1. **API Authentication Errors**:
   - Ensure your API keys are correctly set in the `.env` file
   - Check that the APIs have sufficient quota/credits remaining

2. **Image Loading Errors**:
   - Verify that the required images exist in the fullmap directory
   - Check that the fullmap directory is accessible to the scripts

3. **Generation Timeout**:
   - For complex themes, generation may take longer
   - Check console output for detailed progress information 