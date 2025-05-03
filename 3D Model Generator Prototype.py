import cv2
import numpy as np
import os
import uuid
import platform
import asyncio
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='3d_converter.log'
)
logger = logging.getLogger('3D_Converter')

# Check optional dependencies
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.error("Trimesh not installed. Run 'pip install trimesh'.")

try:
    from pythreejs import *
    PYTHREEJS_AVAILABLE = True
except ImportError:
    PYTHREEJS_AVAILABLE = False
    logger.warning("Pythreejs not installed. Using matplotlib for visualization.")

try:
    import torch
    from torchvision import transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
    MODEL_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    MODEL_AVAILABLE = False
    logger.warning("PyTorch or models not installed. AI features limited.")

class AIEnhancer:
    """Class to handle AI enhancements for 3D generation."""
    
    def __init__(self):
        self.device = torch.device("cuda:0" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        
        if TORCH_AVAILABLE and MODEL_AVAILABLE:
            try:
                self.model = resnet18(pretrained=True)
                self.model.eval()
                self.model.to(self.device)
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                logger.info("AI model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load AI model: {e}")
                self.model = None
    
    def enhance_image(self, img):
        """Enhance image using AI feature extraction."""
        if not TORCH_AVAILABLE or not MODEL_AVAILABLE or self.model is None:
            logger.info("AI enhancement skipped: model unavailable")
            return img
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                hook_output = []
                def hook_fn(module, input, output):
                    hook_output.append(output)
                hook = self.model.layer2.register_forward_hook(hook_fn)
                _ = self.model(input_tensor)
                hook.remove()
                
                if hook_output:
                    features = hook_output[0][0, 0].cpu().numpy()
                    features = cv2.resize(features, (img.shape[1], img.shape[0]))
                    features = (features - features.min()) / (features.max() - features.min())
                    enhanced_img = img.copy()
                    for c in range(3):
                        enhanced_img[:, :, c] = cv2.addWeighted(
                            img[:, :, c], 0.7, (features * 255).astype(np.uint8), 0.3, 0
                        )
                    logger.info("Image enhanced with AI features")
                    return enhanced_img
        except Exception as e:
            logger.error(f"Error in AI image enhancement: {e}")
        return img
    
    def suggest_3d_params(self, img_path=None, prompt=None):
        """Suggest 3D model parameters based on image or text prompt."""
        suggestions = {
            'depth_factor': 1.0,
            'smoothness': 0.5,
            'detail_level': 'medium',
            'color_enhancement': True
        }
        
        if img_path and os.path.exists(img_path) and TORCH_AVAILABLE and self.model:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                max_idx = outputs.argmax().item()
                if max_idx < 300:  # Natural objects
                    suggestions.update({'depth_factor': 1.3, 'smoothness': 0.6, 'detail_level': 'high'})
                elif max_idx < 600:  # Man-made objects
                    suggestions.update({'depth_factor': 0.9, 'smoothness': 0.4, 'detail_level': 'medium'})
                else:  # Abstract
                    suggestions.update({'depth_factor': 1.0, 'smoothness': 0.5, 'detail_level': 'low'})
                logger.info(f"AI suggested image-based parameters: {suggestions}")
            except Exception as e:
                logger.error(f"Error in AI parameter suggestion for image: {e}")
        
        elif prompt:
            prompt = prompt.lower()
            if any(word in prompt for word in ['detailed', 'complex', 'intricate']):
                suggestions['detail_level'] = 'high'
                suggestions['depth_factor'] = 1.4
            if any(word in prompt for word in ['smooth', 'rounded', 'soft']):
                suggestions['smoothness'] = 0.7
            if any(word in prompt for word in ['sharp', 'angular', 'edged']):
                suggestions['smoothness'] = 0.3
            if any(word in prompt for word in ['flat', 'thin']):
                suggestions['depth_factor'] = 0.7
            if any(word in prompt for word in ['deep', 'thick', 'heavy']):
                suggestions['depth_factor'] = 1.5
            logger.info(f"Text-based AI suggested parameters: {suggestions}")
        
        return suggestions

def preprocess_image(image_path, ai_enhancer=None):
    """Preprocess image: remove background using GrabCut and extract object."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or invalid format: {image_path}")
        
        if ai_enhancer:
            img = ai_enhancer.enhance_image(img)
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (10, 10, w - 20, h - 20)
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img * mask2[:, :, np.newaxis]
        return result, mask2
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        raise ValueError(f"Error in image preprocessing: {str(e)}")

def generate_height_map(img, mask, detail_factor=1.0, smoothness=0.5):
    """Generate height map using Sobel filter for depth estimation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_amount = int(5 * (1 - smoothness)) * 2 + 1
    gray = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
    ksize = max(3, min(7, int(3 + detail_factor * 2)))
    if ksize % 2 == 0:
        ksize += 1
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    height_map = np.sqrt(sobelx ** 2 + sobely ** 2) * detail_factor
    height_map = cv2.normalize(height_map, None, 0, 1, cv2.NORM_MINMAX)
    return height_map * mask

def image_to_3d(image_path, output_path, ai_enhancer=None, depth_factor=1.0, smoothness=0.5, detail_level='medium'):
    """Convert image to 3D model (.obj)."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("Trimesh required. Install with 'pip install trimesh'")
    
    try:
        detail_map = {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        detail_factor = detail_map.get(detail_level, 1.0)
        img, mask = preprocess_image(image_path, ai_enhancer)
        height_map = generate_height_map(img, mask, detail_factor, smoothness)
        
        h, w = height_map.shape
        vertices = []
        faces = []
        vertex_index = {}
        idx = 0
        step = max(1, min(3, int(4 - detail_factor * 3)))
        
        for i in range(0, h, step):
            for j in range(0, w, step):
                if mask[i, j] > 0:
                    vertex_index[(i, j)] = idx
                    vertices.append([j / w, i / h, height_map[i, j] * depth_factor])
                    idx += 1
        
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                if all((x, y) in vertex_index for x, y in [(i, j), (i, j + step), (i + step, j), (i + step, j + step)]):
                    v0 = vertex_index[(i, j)]
                    v1 = vertex_index[(i, j + step)]
                    v2 = vertex_index[(i + step, j)]
                    v3 = vertex_index[(i + step, j + step)]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        uv_coords = [[v[0], v[1]] for v in vertices]
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=trimesh.visual.TextureVisuals(uv=uv_coords)
        )
        mesh.export(output_path)
        logger.info(f"3D model exported to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating 3D model from image: {e}")
        raise ValueError(f"Error generating 3D model from image: {str(e)}")

def parse_text_prompt(prompt):
    """Parse text prompt for object type and scale."""
    prompt = prompt.lower()
    scale = 1.0
    if "small" in prompt:
        scale = 0.5
    elif "large" in prompt:
        scale = 1.5
    
    object_types = {
        "car": "car", "vehicle": "car", "automobile": "car",
        "chair": "chair", "seat": "chair", "stool": "chair",
        "table": "table", "desk": "table",
        "toy": "toy", "figurine": "toy", "action figure": "toy", "doll": "toy",
        "building": "building", "house": "building", "structure": "building",
        "animal": "animal", "creature": "animal",
        "person": "human", "human": "human", "figure": "human",
        "tree": "tree", "plant": "tree",
        "mountain": "landscape", "landscape": "landscape",
        "phone": "device", "device": "device", "computer": "device"
    }
    
    obj_type = "sphere"
    for key, value in object_types.items():
        if key in prompt:
            obj_type = value
            break
    return obj_type, scale

def text_to_3d(prompt, output_path, ai_enhancer=None):
    """Generate 3D model from text prompt."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("Trimesh required. Install with 'pip install trimesh'")
    
    try:
        obj_type, scale = parse_text_prompt(prompt)
        if ai_enhancer:
            suggestions = ai_enhancer.suggest_3d_params(prompt=prompt)
            scale *= suggestions.get('depth_factor', 1.0)
        
        if obj_type == "car":
            mesh1 = trimesh.creation.box(extents=(0.2 * scale, 0.1 * scale, 0.05 * scale))
            mesh2 = trimesh.creation.box(extents=(0.12 * scale, 0.1 * scale, 0.04 * scale), 
                                         transform=trimesh.transformations.translation_matrix([0, 0, 0.045 * scale]))
            mesh = trimesh.util.concatenate([mesh1, mesh2])
        elif obj_type == "chair":
            seat = trimesh.creation.box(extents=(0.1 * scale, 0.1 * scale, 0.02 * scale))
            back = trimesh.creation.box(extents=(0.1 * scale, 0.02 * scale, 0.1 * scale),
                                       transform=trimesh.transformations.translation_matrix([0, -0.04 * scale, 0.06 * scale]))
            legs = [trimesh.creation.cylinder(radius=0.01 * scale, height=0.1 * scale,
                                             transform=trimesh.transformations.translation_matrix([x * scale, y * scale, -0.05 * scale]))
                    for x, y in [(0.04, 0.04), (-0.04, 0.04), (0.04, -0.04), (-0.04, -0.04)]]
            mesh = trimesh.util.concatenate([seat, back] + legs)
        elif obj_type == "table":
            top = trimesh.creation.box(extents=(0.2 * scale, 0.15 * scale, 0.02 * scale))
            legs = [trimesh.creation.cylinder(radius=0.01 * scale, height=0.1 * scale,
                                             transform=trimesh.transformations.translation_matrix([x * scale, y * scale, -0.05 * scale]))
                    for x, y in [(0.08, 0.06), (-0.08, 0.06), (0.08, -0.06), (-0.08, -0.06)]]
            mesh = trimesh.util.concatenate([top] + legs)
        elif obj_type == "human":
            body = trimesh.creation.cylinder(radius=0.03 * scale, height=0.15 * scale)
            head = trimesh.creation.icosphere(radius=0.04 * scale, 
                                             transform=trimesh.transformations.translation_matrix([0, 0, 0.115 * scale]))
            arms = [trimesh.creation.cylinder(radius=0.01 * scale, height=0.1 * scale,
                                             transform=trimesh.transformations.translation_matrix([x * scale, 0, 0.05 * scale]))
                    .apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
                    for x in [0.08, -0.08]]
            legs = [trimesh.creation.cylinder(radius=0.015 * scale, height=0.12 * scale,
                                             transform=trimesh.transformations.translation_matrix([x * scale, 0, -0.135 * scale]))
                    for x in [0.02, -0.02]]
            mesh = trimesh.util.concatenate([body, head] + arms + legs)
        elif obj_type == "building":
            base = trimesh.creation.box(extents=(0.2 * scale, 0.15 * scale, 0.3 * scale))
            roof = trimesh.creation.box(extents=(0.23 * scale, 0.18 * scale, 0.05 * scale),
                                       transform=trimesh.transformations.translation_matrix([0, 0, 0.175 * scale]))
            mesh = trimesh.util.concatenate([base, roof])
        elif obj_type == "animal":
            body = trimesh.creation.cylinder(radius=0.04 * scale, height=0.15 * scale,
                                            transform=trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            head = trimesh.creation.icosphere(radius=0.035 * scale, 
                                            transform=trimesh.transformations.translation_matrix([0.095 * scale, 0, 0.02 * scale]))
            legs = [trimesh.creation.cylinder(radius=0.01 * scale, height=0.08 * scale,
                                             transform=trimesh.transformations.translation_matrix([x * scale, y * scale, -0.06 * scale]))
                    for x, y in [(0.05, 0.03), (0.05, -0.03), (-0.05, 0.03), (-0.05, -0.03)]]
            tail = trimesh.creation.cylinder(radius=0.01 * scale, height=0.06 * scale,
                                            transform=trimesh.transformations.translation_matrix([-0.1 * scale, 0, 0.01 * scale]))
            mesh = trimesh.util.concatenate([body, head] + legs + [tail])
        elif obj_type == "tree":
            trunk = trimesh.creation.cylinder(radius=0.02 * scale, height=0.2 * scale)
            foliage = trimesh.creation.icosphere(radius=0.1 * scale, 
                                               transform=trimesh.transformations.translation_matrix([0, 0, 0.15 * scale]))
            mesh = trimesh.util.concatenate([trunk, foliage])
        elif obj_type == "device":
            body = trimesh.creation.box(extents=(0.1 * scale, 0.06 * scale, 0.01 * scale))
            screen = trimesh.creation.box(extents=(0.09 * scale, 0.05 * scale, 0.002 * scale),
                                         transform=trimesh.transformations.translation_matrix([0, 0, 0.006 * scale]))
            mesh = trimesh.util.concatenate([body, screen])
        elif obj_type == "landscape":
            base = trimesh.creation.box(extents=(0.3 * scale, 0.3 * scale, 0.01 * scale))
            mountains = [trimesh.creation.cone(radius=0.03 * scale, height=(0.05 + 0.08 * np.random.random()) * scale,
                                              transform=trimesh.transformations.translation_matrix([(i-2) * 0.05 * scale, (j-2) * 0.05 * scale, (0.05 + 0.08 * np.random.random()) * scale / 2]))
                         for i in range(5) for j in range(5) if np.random.random() > 0.5]
            mesh = trimesh.util.concatenate([base] + mountains) if mountains else base
        elif obj_type == "toy":
            sphere = trimesh.creation.icosphere(radius=0.08 * scale)
            stripes = [trimesh.creation.cylinder(radius=0.082 * scale, height=0.01 * scale,
                                                transform=trimesh.transformations.translation_matrix([0, 0, z * scale]))
                       for z in [0.04, -0.04]]
            mesh = trimesh.boolean.difference([sphere] + stripes)
        else:
            mesh = trimesh.creation.icosphere(radius=0.1 * scale)
        
        mesh.export(output_path)
        logger.info(f"3D model exported to {output_path} for prompt: {prompt}")
        return True
    except Exception as e:
        logger.error(f"Error generating 3D model from text: {e}")
        raise ValueError(f"Error generating 3D model from text: {str(e)}")

def visualize_3d(obj_path, output_image=None, ax=None):
    """Visualize 3D model using matplotlib."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("Trimesh required. Install with 'pip install trimesh'")
    
    try:
        mesh = trimesh.load(obj_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, cmap='viridis', linewidth=0.2, alpha=0.9, edgecolor='k')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_title('3D Model Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        
        if output_image:
            plt.savefig(output_image, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Static visualization saved to {output_image}")
            return None
        return ax
    except Exception as e:
        logger.error(f"Error visualizing 3D model: {e}")
        raise ValueError(f"Error visualizing 3D model: {str(e)}")

class ModelConfigManager:
    """Manage model configurations and presets."""
    
    def __init__(self, config_file='model_configs.json'):
        self.config_file = config_file
        self.configs = self._load_configs()
    
    def _load_configs(self):
        """Load configurations from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configurations: {e}")
                return self._get_default_configs()
        return self._get_default_configs()
    
    def _get_default_configs(self):
        """Return default configurations."""
        return {
            "presets": {
                "Standard": {"depth_factor": 1.0, "smoothness": 0.5, "detail_level": "medium"},
                "High Detail": {"depth_factor": 1.2, "smoothness": 0.3, "detail_level": "high"},
                "Smooth": {"depth_factor": 0.8, "smoothness": 0.8, "detail_level": "low"},
                "Exaggerated": {"depth_factor": 1.8, "smoothness": 0.4, "detail_level": "medium"}
            }
        }
    
    def save_configs(self):
        """Save configurations to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.configs, f, indent=4)
            logger.info("Configurations saved successfully")
        except Exception as e:
            logger.error(f"Error saving configurations: {e}")
    
    def add_preset(self, name, params):
        """Add a new preset configuration."""
        self.configs["presets"][name] = params
        self.save_configs()
        logger.info(f"Added preset: {name}")

class ModelGeneratorGUI:
    """Tkinter GUI for 3D model generation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("3D Model Generator")
        self.root.geometry("800x600")
        self.ai_enhancer = AIEnhancer()
        self.config_manager = ModelConfigManager()
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI components."""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input selection
        ttk.Label(self.main_frame, text="Input Type:").grid(row=0, column=0, sticky=tk.W)
        self.input_type = tk.StringVar(value="image")
        ttk.Radiobutton(self.main_frame, text="Image", variable=self.input_type, value="image").grid(row=0, column=1)
        ttk.Radiobutton(self.main_frame, text="Text", variable=self.input_type, value="text").grid(row=0, column=2)
        
        # Image input
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Input", padding="5")
        self.image_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.image_path = tk.StringVar()
        ttk.Entry(self.image_frame, textvariable=self.image_path, width=50).grid(row=0, column=0)
        ttk.Button(self.image_frame, text="Browse", command=self.browse_image).grid(row=0, column=1)
        
        # Text input
        self.text_frame = ttk.LabelFrame(self.main_frame, text="Text Input", padding="5")
        self.text_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.text_prompt = tk.StringVar()
        ttk.Entry(self.text_frame, textvariable=self.text_prompt, width=50).grid(row=0, column=0)
        
        # Configuration
        self.config_frame = ttk.LabelFrame(self.main_frame, text="Configuration", padding="5")
        self.config_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        ttk.Label(self.config_frame, text="Preset:").grid(row=0, column=0)
        self.preset = tk.StringVar(value="Standard")
        presets = list(self.config_manager.configs["presets"].keys())
        ttk.Combobox(self.config_frame, textvariable=self.preset, values=presets).grid(row=0, column=1)
        
        # Progress and status
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.main_frame, textvariable=self.status).grid(row=4, column=0, columnspan=3)
        self.progress = ttk.Progressbar(self.main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Generate button
        ttk.Button(self.main_frame, text="Generate 3D Model", command=self.start_generation).grid(row=6, column=0, columnspan=3)
        
        # Visualization canvas
        self.fig = plt.Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=3)
    
    def browse_image(self):
        """Open file dialog to select image."""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path:
            self.image_path.set(path)
    
    def start_generation(self):
        """Start 3D model generation in a separate thread."""
        self.status.set("Generating...")
        self.progress.start()
        threading.Thread(target=self.generate_model, daemon=True).start()
    
    def generate_model(self):
        """Generate 3D model based on input."""
        try:
            output_path = os.path.join(self.output_dir, f"model_{uuid.uuid4()}.obj")
            config = self.config_manager.configs["presets"][self.preset.get()]
            
            if self.input_type.get() == "image":
                image_path = self.image_path.get()
                if not os.path.exists(image_path):
                    raise ValueError("Image file does not exist")
                success = image_to_3d(
                    image_path, output_path, self.ai_enhancer,
                    **config
                )
            else:
                prompt = self.text_prompt.get()
                if not prompt:
                    raise ValueError("Text prompt cannot be empty")
                success = text_to_3d(prompt, output_path, self.ai_enhancer)
            
            if success:
                self.ax.clear()
                visualize_3d(output_path, ax=self.ax)
                self.canvas.draw()
                self.status.set(f"Model generated: {output_path}")
                messagebox.showinfo("Success", f"3D model saved to {output_path}")
        except Exception as e:
            self.status.set("Error occurred")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

async def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = ModelGeneratorGUI(root)
    root.mainloop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())