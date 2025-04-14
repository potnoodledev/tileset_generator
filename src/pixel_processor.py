import numpy as np
from PIL import Image
import os
import cv2
import colorsys
import math

class SeamlessTileProcessor:
    """
    A processor specialized for creating seamless tileable terrain textures.
    Can produce exact 16x16 pixel art tiles appropriate for retro-style games.
    """
    
    def __init__(self, num_colors=16, grid_size=4, scale_factor=1, force_16x16=False):
        """
        Initialize the SeamlessTileProcessor.
        
        Args:
            num_colors: Maximum number of colors in the output palette
            grid_size: Size of pixel blocks (smaller = less pixelated)
            scale_factor: Upscaling factor for the final image
            force_16x16: Whether to force the output to be exactly 16x16 pixels
        """
        self.num_colors = num_colors
        self.grid_size = max(1, grid_size)
        self.scale_factor = max(1, scale_factor)
        self.force_16x16 = force_16x16
    
    def load_image(self, image_path):
        """Load and convert image to RGBA mode."""
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    
    def quantize_colors(self, img):
        """
        Reduce the number of colors in the image.
        Uses k-means clustering to find the optimal color palette.
        
        Args:
            img: PIL Image object
            
        Returns:
            PIL Image with reduced color palette
        """
        # Convert to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create output array
        output = np.zeros_like(img_array)
        
        # Get all non-transparent pixels
        mask = img_array[:, :, 3] > 10
        valid_pixels = img_array[mask]
        
        if len(valid_pixels) == 0:
            print("Warning: Image contains no valid pixels")
            return img
        
        # Extract RGB values for clustering
        pixels = valid_pixels[:, :3].reshape(-1, 3).astype(np.float32)
        
        # Check if we have fewer unique colors than requested
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) <= self.num_colors:
            print(f"Image already has fewer unique colors ({len(unique_colors)}) than requested ({self.num_colors})")
            return img
        
        # Apply k-means clustering to find the optimal palette
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, palette = cv2.kmeans(pixels, self.num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to uint8
        palette = palette.astype(np.uint8)
        
        # Map each pixel to the closest color in the palette
        for y in range(height):
            for x in range(width):
                if img_array[y, x, 3] > 10:  # Only process visible pixels
                    pixel = img_array[y, x, :3].astype(np.float32).reshape(1, 3)
                    
                    # Find the closest color in the palette
                    distances = np.sum((palette - pixel) ** 2, axis=1)
                    closest_color_index = np.argmin(distances)
                    
                    # Assign the closest color
                    output[y, x, :3] = palette[closest_color_index]
                    output[y, x, 3] = 255  # Fully opaque
                else:
                    output[y, x] = [0, 0, 0, 0]  # Fully transparent
        
        return Image.fromarray(output)
    
    def pixelate(self, img):
        """
        Pixelate the image to create a pixel art style.
        If force_16x16 is True, will resize to exactly 16x16 pixels.
        
        Args:
            img: PIL Image object
            
        Returns:
            Pixelated PIL Image
        """
        # Get image dimensions
        width, height = img.size
        
        if self.force_16x16:
            # Direct resize to 16x16
            return img.resize((16, 16), Image.NEAREST)
        else:
            # Standard pixelation process
            # Calculate size for downsampling
            target_w = max(1, width // self.grid_size)
            target_h = max(1, height // self.grid_size)
            
            # Downsample the image
            small_img = img.resize((target_w, target_h), Image.NEAREST)
            
            # Upsample back to original size
            result = small_img.resize((width, height), Image.NEAREST)
            
            return result
    
    def harmonize_colors(self, img, saturation_factor=1.1, value_factor=1.05):
        """
        Optional: Harmonize colors to create a more cohesive palette.
        Slightly increases saturation and brightness for more vibrant terrain textures.
        
        Args:
            img: PIL Image object
            saturation_factor: Factor to adjust saturation (1.0 = no change)
            value_factor: Factor to adjust brightness (1.0 = no change)
            
        Returns:
            PIL Image with harmonized colors
        """
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create output array
        output = np.zeros_like(img_array)
        
        # Adjust colors using HSV color space
        for y in range(height):
            for x in range(width):
                if img_array[y, x, 3] > 10:  # Only process visible pixels
                    r, g, b = img_array[y, x, :3]
                    
                    # Convert to HSV
                    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                    
                    # Adjust saturation and value
                    s = min(1.0, s * saturation_factor)
                    v = min(1.0, v * value_factor)
                    
                    # Convert back to RGB
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    
                    # Scale back to 0-255 range
                    output[y, x, :3] = [int(r*255), int(g*255), int(b*255)]
                    output[y, x, 3] = img_array[y, x, 3]  # Preserve original alpha
                else:
                    output[y, x] = [0, 0, 0, 0]  # Fully transparent
        
        return Image.fromarray(output)
    
    def scale(self, img):
        """
        Scale the image by the configured scale factor.
        If force_16x16 was used, scaling will maintain pixel aspect ratio.
        
        Args:
            img: PIL Image object
            
        Returns:
            Scaled PIL Image
        """
        if self.scale_factor <= 1:
            return img
        
        width, height = img.size
        
        # If we're in 16x16 mode and scaling, maintain that exact ratio
        if self.force_16x16 and width == 16 and height == 16:
            new_width = 16 * self.scale_factor
            new_height = 16 * self.scale_factor
            print(f"Scaling 16x16 image to {new_width}x{new_height}...")
        else:
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
        
        # Scale using nearest neighbor to preserve sharp pixels
        return img.resize((new_width, new_height), Image.NEAREST)
    
    def ensure_seamless(self, img):
        """
        Process a texture to make it more seamless.
        This function has been disabled to prevent edge blending artifacts.
        
        Args:
            img: PIL Image object
            
        Returns:
            Original PIL Image without modification
        """
        # Function disabled - return original image without modifications
        print("Edge blending disabled - returning original image")
        return img
    
    def process_texture(self, img_or_path, output_path=None, make_seamless=True, harmonize=True):
        """
        Process a texture image with all steps.
        
        Args:
            img_or_path: PIL Image object or path to image file
            output_path: Where to save the processed image (optional)
            make_seamless: Whether to make the texture seamless
            harmonize: Whether to harmonize colors
            
        Returns:
            Processed PIL Image
        """
        # Load image if path provided
        if isinstance(img_or_path, str):
            img = self.load_image(img_or_path)
        else:
            img = img_or_path
        
        # Step 1: If forcing 16x16, do immediate resize for faster processing
        if self.force_16x16:
            print("Resizing to 16x16 pixels...")
            img = img.resize((16, 16), Image.LANCZOS)  # Use LANCZOS for better quality initial resize
            
        # Step 2: Color quantization
        print("Quantizing colors...")
        img = self.quantize_colors(img)
        
        # Step 3: Optional color harmonization for more vibrant terrain
        if harmonize:
            print("Harmonizing colors...")
            img = self.harmonize_colors(img)
        
        # Step 4: Make seamless if requested
        if make_seamless:
            print("Making texture seamless...")
            img = self.ensure_seamless(img)
        
        # Step 5: Pixelate (if not already 16x16)
        if not self.force_16x16 or (img.width != 16 or img.height != 16):
            print("Pixelating texture...")
            img = self.pixelate(img)
        
        # Step 6: Scale if needed
        if self.scale_factor > 1:
            print(f"Scaling by factor {self.scale_factor}...")
            img = self.scale(img)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, "PNG")
            print(f"Saved processed texture to {output_path}")
            
            # Also save a reasonable display version for easy viewing if we have a very large scale
            if self.force_16x16 and self.scale_factor >= 8:
                display_size = 128  # Default display size
                display_path = output_path.replace(".png", "_display.png")
                img_resized = img.resize((display_size, display_size), Image.NEAREST)
                img_resized.save(display_path, "PNG")
                print(f"Saved compact display version ({display_size}x{display_size}) to {display_path}")
            # For smaller scales, save a display version only if we're at the base 16x16
            elif self.force_16x16 and self.scale_factor == 1:
                display_path = output_path.replace(".png", "_display.png")
                img.resize((128, 128), Image.NEAREST).save(display_path, "PNG")
                print(f"Saved display version (128x128) to {display_path}")
        
        return img
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images to process
            output_dir: Directory to save processed images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        found_images = False
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                
                print(f"Processing {filename}...")
                self.process_texture(input_path, output_path)
                found_images = True
                
        if not found_images:
            print(f"No image files found in directory: {input_dir}") 