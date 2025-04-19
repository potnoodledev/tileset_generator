import pygame
import sys
import os
import subprocess
import time
import threading
from pygame.locals import *
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize timer at startup for testing
print("Initializing timer at startup...")
timer_active = True
start_time = time.time()
print(f"Initial start_time: {start_time}")

def test_api_credentials():
    """Test if API credentials are working correctly."""
    try:
        import anthropic
        import replicate
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        
        if not anthropic_key:
            print("ERROR: ANTHROPIC_API_KEY is not set in .env file")
            return False
            
        if not replicate_token:
            print("ERROR: REPLICATE_API_TOKEN is not set in .env file")
            return False
            
        # Test Anthropic client initialization
        try:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            print("Anthropic client initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Anthropic client: {e}")
            return False
            
        # Test Replicate client initialization
        try:
            replicate_client = replicate.Client(api_token=replicate_token)
            print("Replicate client initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Replicate client: {e}")
            return False
            
        return True
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        print("Please make sure all required packages are installed.")
        return False

# Check for required API keys
if not test_api_credentials():
    print("API credentials test failed. Please check your .env file and installed packages.")
    sys.exit(1)
else:
    print("API credentials verified successfully!")

# Since we're now in the src directory, we don't need to add it to the path
# Just import directly
from generate_base_tiles import generate_texture_from_theme

# Initialize Pygame
pygame.init()

# Get the base directory (parent of the src directory)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the map image using absolute path
map_image = pygame.image.load(os.path.join(base_dir, "fullmap", "fullmap_floor_wall_blend_2x2_processed.png"))
map_rect = map_image.get_rect()

# Constants
ZOOM_FACTOR = 3  # Increased from 4 to 6 for more zoom
WINDOW_WIDTH, WINDOW_HEIGHT = 600, 400  # Real game dimensions
MOVE_SPEED = 5  # Slower movement speed for more incremental control
FONT = pygame.font.Font(None, 20)  # Standardized font size
SMALL_FONT = pygame.font.Font(None, 20)  # Match the same size

# UI Colors
BUTTON_COLOR = (50, 50, 50)  # Dark gray for buttons - will be used for borders only now
HIGHLIGHT_COLOR = (70, 70, 70)  # Lighter gray for highlighted buttons
TEXT_COLOR = (240, 240, 240)  # White for text
ACTIVE_COLOR = (200, 200, 200)  # Light gray for active elements
INACTIVE_COLOR = (130, 130, 130)  # Medium gray for inactive elements
SUCCESS_COLOR = TEXT_COLOR  # Use white for all status messages
ERROR_COLOR = TEXT_COLOR    # Use white for all status messages
WARNING_COLOR = TEXT_COLOR  # Use white for all status messages

# Calculate the size of the visible area
visible_width = WINDOW_WIDTH // ZOOM_FACTOR
visible_height = WINDOW_HEIGHT // ZOOM_FACTOR

# Create the display window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Vibe Dungeon Viewer")

# Fix timer initialization - set it here explicitly
print("Setting timer start time explicitly in main code")
if start_time is None:
    start_time = time.time()
    print(f"Timer initialized with start_time: {start_time}")

# Initial position (center the view)
x, y = (map_rect.width - visible_width) // 2, (map_rect.height - visible_height) // 2

# Textbox setup - adjust the initial position
input_box = pygame.Rect(WINDOW_WIDTH - 190, 10, 170, 32)
color = INACTIVE_COLOR
active = False
text = ''

# Generate button
generate_button = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 70, 200, 50)
generate_button_hover = False

# Current theme tracking
current_floor_theme = "default"
current_wall_theme = "default"

# Timer variables
timer_active = False
start_time = None
last_generation_time = "00:00"  # Store the last generation time

# Status variables
status_message = ''
status_color = pygame.Color('white')
generating = False

# Add a global variable for storing generation status
generation_status = ""

# Function to start the timer
def start_timer():
    """Start the timer to measure generation time."""
    global timer_active, start_time
    print("Starting generation timer...")
    timer_active = True
    start_time = time.time()
    print(f"Timer started at: {start_time}")
    return start_time

# Function to get the time to display
def get_display_time():
    """Get the time to display - either active timer or last generation time."""
    if timer_active and start_time is not None:
        # Show active timer
        elapsed_seconds = int(time.time() - start_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    else:
        # Show last generation time
        return last_generation_time

# Stop the timer and update the last generation time
def stop_timer():
    """Stop the timer and update the last generation time."""
    global timer_active, last_generation_time
    
    if not timer_active or start_time is None:
        return "00:00"
    
    elapsed_seconds = int(time.time() - start_time)
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    last_generation_time = f"{minutes:02d}:{seconds:02d}"
    
    print(f"Generation completed in: {last_generation_time}")
    timer_active = False
    return last_generation_time

def run_blend_script():
    """Run the full_map_generator.py to blend floor and wall tiles."""
    try:
        print("Running blend script to generate tileset...")
        # Since we're in the same directory now, we can just use the filename
        blend_script_path = os.path.join(os.path.dirname(__file__), "full_map_generator.py")
        result = subprocess.run([sys.executable, blend_script_path], 
                               capture_output=True, text=True, check=True)
        print("Blend script output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running blend script: {e}")
        print(f"Script output: {e.stdout}")
        print(f"Script error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running blend script: {e}")
        return False

# Define the generate_tiles function to handle the tile generation process
def generate_tiles(theme_text):
    global generating, status_message, status_color, current_floor_theme, current_wall_theme, map_image, map_rect, generation_status, need_reload
    
    generating = True
    generation_status = f"Starting generation for: {theme_text}"
    need_reload = False  # Reset the reload flag when starting a new generation
    
    try:
        print(f"Starting tile generation for theme: {theme_text}")
        
        # Generate floor tile
        generation_status = "Generating detailed ground tile..."
        print(generation_status)
        floor_image = generate_texture_from_theme(theme_text, 'detailed_ground')
        if floor_image:
            print("Detailed ground tile generated successfully")
            # Save the floor image using absolute path
            floor_path = os.path.join(base_dir, 'fullmap', 'floor.png')
            floor_image.save(floor_path)
            print(f"Floor tile saved to {floor_path}")
            generation_status = "Floor tile generated successfully"
        else:
            print("Failed to generate detailed ground tile")
            generation_status = "Failed to generate floor tile"

        # Generate wall tile
        generation_status = "Generating wall tile..."
        print(generation_status)
        wall_image = generate_texture_from_theme(theme_text, 'wall')
        if wall_image:
            print("Wall tile generated successfully")
            # Save the wall image using absolute path
            wall_path = os.path.join(base_dir, 'fullmap', 'wall.png')
            wall_image.save(wall_path)
            print(f"Wall tile saved to {wall_path}")
            generation_status = "Wall tile generated successfully"
        else:
            print("Failed to generate wall tile")
            generation_status = "Failed to generate wall tile"
            
        # Check if both images were generated and saved
        if floor_image and wall_image:
            # Run the blend script to create the blended tileset
            generation_status = "Blending tiles..."
            current_floor_theme = theme_text  # Store the theme for floor
            current_wall_theme = theme_text   # Store the theme for wall
            
            if run_blend_script():
                elapsed_time = stop_timer()  # Stop and get the elapsed time
                generation_status = "Tileset generation complete!"
                status_message = f"Tileset generated successfully! Time: {elapsed_time}"
                status_color = SUCCESS_COLOR
                
                # Reload the blended map image from the main thread
                # We'll do this in the main loop to avoid threading issues
            else:
                elapsed_time = stop_timer()  # Stop and get the elapsed time
                generation_status = "Blending failed"
                status_message = f"Tiles generated but blending failed. Time: {elapsed_time}"
                status_color = ERROR_COLOR
        else:
            elapsed_time = stop_timer()  # Stop and get the elapsed time
            generation_status = "Generation failed"
            status_message = f"Failed to generate tiles. Time: {elapsed_time}"
            status_color = ERROR_COLOR
            
    except Exception as e:
        elapsed_time = stop_timer()  # Stop and get the elapsed time
        print(f"Error during tile generation: {str(e)}")
        import traceback
        traceback.print_exc()
        generation_status = f"Error: {str(e)}"
        status_message = f"Error: {str(e)} Time: {elapsed_time}"
        status_color = ERROR_COLOR
    
    generating = False

# Define the generate_tiles function to run in a separate thread
def generate_tiles_thread(theme_text):
    """Background thread for generating tiles."""
    thread = threading.Thread(target=generate_tiles, args=(theme_text,))
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()
    return thread

# Modified draw_rounded_rect to not fill with background color
def draw_rounded_rect(surface, rect, color, radius=15, alpha=180, border_only=True):
    """Draw a rounded rectangle with transparency - optional border only"""
    rect = pygame.Rect(rect)
    
    if border_only:
        # Just draw the border
        pygame.draw.rect(surface, color, rect, 2, border_radius=radius)
    else:
        # Create a surface with an alpha channel
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))  # Fill with transparent black
        
        # Draw the rounded rectangle on the surface
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect.width, rect.height), border_radius=radius)
        
        # Blit the surface onto the target surface
        surface.blit(s, rect)
    
    return rect

# Main loop
running = True
last_reload_check = 0
need_reload = False

while running:
    current_time = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if input_box.collidepoint(event.pos):
                # Toggle the active variable.
                active = not active
            elif generate_button.collidepoint(event.pos) and text and not generating:
                # Start the timer and begin generation in a background thread
                print("Generate button clicked, starting generation timer...")
                start_timer()
                generate_tiles_thread(text)
                # Clear the text field immediately for better UX
                input_text = text
                text = ''
            else:
                active = False
            # Change the current color of the input box.
            color = ACTIVE_COLOR if active else INACTIVE_COLOR
        if event.type == KEYDOWN:
            if active:
                if event.key == K_RETURN and text and not generating:
                    print("Enter key pressed, starting generation timer...")
                    start_timer()
                    generate_tiles_thread(text)
                    # Clear the text field immediately for better UX
                    input_text = text
                    text = ''
                elif event.key == K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode
        
        # Check if mouse is hovering over the generate button
        if event.type == pygame.MOUSEMOTION:
            generate_button_hover = generate_button.collidepoint(event.pos)

    # Check if we need to reload the map (after generation is completed)
    if not generating and generation_status == "Tileset generation complete!" and not need_reload:
        try:
            # Only try to reload once
            need_reload = True
            map_image = pygame.image.load(os.path.join(base_dir, "fullmap", "fullmap_floor_wall_blend_2x2_processed.png"))
            map_rect = map_image.get_rect()
            print("Reloaded map successfully")
            # Reset the generation status to avoid reloading again
            generation_status = "Map reloaded successfully"
        except pygame.error as e:
            print(f"Could not reload map: {e}")
            status_message = f"Generation complete but couldn't reload map."
            status_color = pygame.Color('yellow')

    # Get key states
    keys = pygame.key.get_pressed()

    # Move the view based on key presses
    if keys[pygame.K_LEFT]:
        x -= MOVE_SPEED
    if keys[pygame.K_RIGHT]:
        x += MOVE_SPEED
    if keys[pygame.K_UP]:
        y -= MOVE_SPEED
    if keys[pygame.K_DOWN]:
        y += MOVE_SPEED

    # Limit scrolling to map boundaries
    x = max(min(x, map_rect.width - visible_width), 0)
    y = max(min(y, map_rect.height - visible_height), 0)

    # Draw the map
    screen.fill((0, 0, 0))
    # Display the zoomed-in portion of the map, scaled to fill the window
    zoomed_area = pygame.transform.scale(map_image.subsurface((x, y, visible_width, visible_height)), (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(zoomed_area, (0, 0))

    # Render the label and input box without backgrounds
    # Properly position the label and input box
    theme_label_rect = pygame.Rect(WINDOW_WIDTH - 280, 5, 75, 42)  # Make the label rect smaller
    label = FONT.render("Theme:", True, TEXT_COLOR)
    screen.blit(label, (WINDOW_WIDTH - 270, 10))  # Move the text slightly

    # Render the current text
    txt_surface = FONT.render(text, True, TEXT_COLOR if active else INACTIVE_COLOR)

    # Fixed width input box (doesn't grow)
    input_box_width = 170
    input_box.x = WINDOW_WIDTH - 190  # Fixed position
    input_box.y = 10
    input_box.w = input_box_width
    input_box.h = 32

    # Blit the text, truncate if too long
    if txt_surface.get_width() > input_box_width - 10:
        visible_chars = len(text)
        while visible_chars > 0:
            test_text = text[:visible_chars]
            test_surface = FONT.render(test_text, True, TEXT_COLOR)
            if test_surface.get_width() <= input_box_width - 15:
                break
            visible_chars -= 1
        txt_surface = FONT.render(text[:visible_chars], True, TEXT_COLOR if active else INACTIVE_COLOR)

    # Blit the text centered vertically
    screen.blit(txt_surface, (input_box.x+5, input_box.y + (input_box.height - txt_surface.get_height())//2))

    # Draw the input box rect - just border
    color = ACTIVE_COLOR if active else INACTIVE_COLOR
    pygame.draw.rect(screen, color, input_box, 2, border_radius=10)

    # Draw the generate button - border only, no background
    if text:  # Only show the button if there's text in the input
        button_color = ACTIVE_COLOR if generate_button_hover else INACTIVE_COLOR
        pygame.draw.rect(screen, button_color, generate_button, 2, border_radius=15)
        # Center the text in the button
        button_text = FONT.render("GENERATE TILE", True, TEXT_COLOR)
        text_rect = button_text.get_rect(center=generate_button.center)
        screen.blit(button_text, text_rect)

    # Display current floor and wall themes in bottom left - border only, no background
    if current_floor_theme != "default" or current_wall_theme != "default":
        floor_rect = pygame.Rect(10, WINDOW_HEIGHT - 60, 220, 30)
        wall_rect = pygame.Rect(10, WINDOW_HEIGHT - 30, 220, 30)
        
        # Draw borders only
        pygame.draw.rect(screen, INACTIVE_COLOR, floor_rect, 2, border_radius=10)
        pygame.draw.rect(screen, INACTIVE_COLOR, wall_rect, 2, border_radius=10)
        
        floor_text = FONT.render(f"Floor: {current_floor_theme}", True, TEXT_COLOR)
        wall_text = FONT.render(f"Wall: {current_wall_theme}", True, TEXT_COLOR)
        
        screen.blit(floor_text, (20, WINDOW_HEIGHT - 55))
        screen.blit(wall_text, (20, WINDOW_HEIGHT - 25))

    # Display status message - border only, no background  
    if generating:
        # Show the current generation status during active generation
        status_text = generation_status
        status_c = TEXT_COLOR
    elif status_message:
        # Show the final status message after generation
        status_text = status_message
        if "successfully" in status_text.lower():
            status_c = SUCCESS_COLOR
        elif "failed" in status_text.lower() or "error" in status_text.lower():
            status_c = ERROR_COLOR
        else:
            status_c = WARNING_COLOR
    else:
        status_text = ""
        status_c = TEXT_COLOR
        
    if status_text:
        # Calculate the width needed based on the text
        status_surface = FONT.render(status_text, True, status_c)
        status_width = min(WINDOW_WIDTH - 250, status_surface.get_width() + 20)  # Add padding, limit width
        
        status_rect = pygame.Rect(10, WINDOW_HEIGHT - 90, status_width, 30)
        pygame.draw.rect(screen, INACTIVE_COLOR, status_rect, 2, border_radius=10)
        
        # Truncate text if too long
        if status_surface.get_width() > status_width - 20:
            truncated_text = status_text[:30] + "..."
            status_surface = FONT.render(truncated_text, True, status_c)
        
        screen.blit(status_surface, (20, WINDOW_HEIGHT - 85))

    # Display timer - border only, no background
    timer_text = get_display_time()

    # Position timer at the bottom right of the screen
    timer_surface = FONT.render(timer_text, True, TEXT_COLOR)  # Use consistent text color
    timer_width = timer_surface.get_width() + 20  # Slightly smaller padding
    timer_height = 30  # Match height of other elements

    # Calculate bottom right position with padding
    timer_rect = pygame.Rect(WINDOW_WIDTH - timer_width - 10, WINDOW_HEIGHT - 30, timer_width, timer_height)

    # Draw the timer box - border only
    border_color = ACTIVE_COLOR if timer_active else INACTIVE_COLOR
    pygame.draw.rect(screen, border_color, timer_rect, 2, border_radius=10)

    # Center the text in the box
    text_x = timer_rect.x + (timer_rect.width - timer_surface.get_width()) // 2
    text_y = timer_rect.y + (timer_rect.height - timer_surface.get_height()) // 2
    screen.blit(timer_surface, (text_x, text_y))

    # Add a small label above
    label_text = "GEN TIME"
    if timer_active:
        label_text = "GENERATING..."
        
    label_surface = FONT.render(label_text, True, TEXT_COLOR)
    label_rect = pygame.Rect(timer_rect.x, timer_rect.y - 25, timer_width, 20)
    pygame.draw.rect(screen, INACTIVE_COLOR, label_rect, 2, border_radius=10)
    screen.blit(label_surface, (label_rect.x + (label_rect.width - label_surface.get_width()) // 2, label_rect.y + 2))

    # Update the display at a limited frame rate
    pygame.display.flip()
    pygame.time.Clock().tick(30)  # Limit to 30 FPS to avoid excessive CPU usage

# Quit Pygame
pygame.quit()
sys.exit() 