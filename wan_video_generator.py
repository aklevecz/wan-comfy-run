"""wan_video_generator.py
Unified WanVideo generator that handles both initial generation and continuous extensions.

Usage:
    generator = WanVideoGenerator()
    
    # Initial generation
    generator.generate_initial(video_path="input.mp4", prompt="bellowing flames")
    
    # Single extension
    generator.extend_video()
    
    # Multiple extensions
    generator.extend_video(num_extensions=3)
    
    # Continuous extension with custom parameters
    generator.extend_video(num_extensions=5, extension_frames=50, context_frames=20)
"""

import os, sys, importlib.util
from pathlib import Path
import types
import gc
import torch
from contextlib import contextmanager
import numpy as np
from PIL import Image
import subprocess
from typing import Optional, Tuple, List
import json
from dataclasses import dataclass, asdict

# Enable PyTorch optimizations first
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True) 
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print("✓ PyTorch CUDA optimizations enabled")
else:
    print("⚠️ CUDA not available")

# Fix for ComfyUI stochastic rounding gradient issue
torch.set_grad_enabled(False)  # Disable gradients globally for inference
print("✓ Gradients disabled for inference")

@dataclass
class GenerationConfig:
    """Configuration for video generation parameters."""
    # Paths
    comfy_root: str = r"C:\Users\ariel\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI"
    output_base_dir: str = "wan_outputs"
    
    # Generation parameters
    initial_frames: int = 81
    extension_frames: int = 66
    context_frames: int = 15
    
    # Model parameters
    model_name: str = "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"
    vae_name: str = "Wan2_1_VAE_bf16.safetensors"
    t5_name: str = "umt5-xxl-enc-bf16.safetensors"
    vace_name: str = "Wan2_1-VACE_module_14B_bf16.safetensors"
    
    # LoRA parameters
    base_lora: str = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
    base_lora_strength: float = 1.0
    extra_lora: str = "Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors"
    extra_lora_strength: float = 0.3
    
    # Sampling parameters
    steps: int = 6
    cfg: float = 1.0
    shift: float = 10.0
    base_seed: int = 314525102295492
    scheduler: str = "euler"
    
    # Video parameters
    frame_rate: int = 30
    width: int = 512
    height: int = 512
    
    # Extension parameters
    grey_color: int = 8355711  # 0x7F7F7F
    
    # Block swap parameters
    blocks_to_swap: int = 30
    vace_blocks_to_swap: int = 0
    
    # Helper arguments
    feta_weight: float = 2.0
    slg_blocks: str = "8"
    slg_start_percent: float = 0.1
    cfg_zero_star: bool = True

class WanVideoGenerator:
    """Unified WanVideo generator with continuous extension support."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the generator with optional configuration."""
        self.config = config or GenerationConfig()
        self.current_frame_idx = 0
        self.current_output_dir = None
        self.generation_count = 0
        
        # Model persistence
        self.models_loaded = False
        self.video_model = None
        self.vae_model = None
        self.text_embeds = None
        self.lora_config = None
        self.helper_args = None
        
        # ComfyUI setup
        self._setup_comfyui()
        
        # Load WanVideo nodes
        self._load_wan_nodes()
        
        # State tracking
        self.state_file = Path(self.config.output_base_dir) / "generator_state.json"
        self._load_state()
        
    def _setup_comfyui(self):
        """Setup ComfyUI environment with comprehensive stubs."""
        COMFY_ROOT = Path(self.config.comfy_root)
        CUSTOM_NODES = COMFY_ROOT / "custom_nodes"
        
        sys.path.insert(0, str(COMFY_ROOT))
        sys.path.insert(0, str(CUSTOM_NODES))
        
        # Import ComfyUI's real systems
        try:
            import comfy.model_management as mm
            import comfy.utils
            import folder_paths
            import server
            import app.frontend_management
            import app.user_manager
            import utils.install_util
            
            self.mm = mm
            print("✓ All ComfyUI systems imported successfully")
            
        except ImportError as e:
            print(f"⚠️ Creating stubs for missing ComfyUI systems: {e}")
            self._create_comfyui_stubs()
    
    def _create_comfyui_stubs(self):
        """Create comprehensive stubs for missing ComfyUI modules."""
        # Create fallback memory management
        class FallbackMM:
            @staticmethod
            def soft_empty_cache():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            @staticmethod
            def unload_all_models():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mm = FallbackMM()
        
        # Create module stubs as needed
        if 'comfy.model_management' not in sys.modules:
            sys.modules['comfy.model_management'] = types.ModuleType('comfy.model_management')
        
        # Create comprehensive app stubs
        self._create_app_stubs()
        print("✓ ComfyUI stubs created")
    
    def _create_app_stubs(self):
        """Create app module stubs."""
        # Utils stubs
        if 'utils.install_util' not in sys.modules:
            utils_install_util = types.ModuleType('utils.install_util')
            setattr(utils_install_util, 'get_missing_requirements_message', lambda *args, **kwargs: "")
            setattr(utils_install_util, 'requirements_path', lambda *args, **kwargs: "")
            sys.modules['utils.install_util'] = utils_install_util
            
            utils_root = types.ModuleType('utils')
            setattr(utils_root, 'install_util', utils_install_util)
            sys.modules['utils'] = utils_root
        
        # App stubs
        if 'app.frontend_management' not in sys.modules:
            app_frontend = types.ModuleType('app.frontend_management')
            class FrontendManagerStub:
                pass
            setattr(app_frontend, 'FrontendManager', FrontendManagerStub)
            sys.modules['app.frontend_management'] = app_frontend
            
            # Create comprehensive app root
            app_root = types.ModuleType('app')
            setattr(app_root, 'frontend_management', app_frontend)
            sys.modules['app'] = app_root
        
        # Server stubs
        if 'server' not in sys.modules:
            server_module = types.ModuleType('server')
            class PromptServerStub:
                instance = None
                client_id = 0
                class BinaryEventTypes:
                    PREVIEW_IMAGE = 0
                def send_sync(self, *args, **kwargs):
                    return None
            
            setattr(server_module, 'PromptServer', PromptServerStub)
            # Create instance
            PromptServerStub.instance = PromptServerStub()
            sys.modules['server'] = server_module
    
    def _load_wan_nodes(self):
        """Load WanVideo nodes."""
        wan_nodes_py = Path(self.config.comfy_root) / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "nodes.py"
        
        spec = importlib.util.spec_from_file_location(
            "wan_nodes", wan_nodes_py, submodule_search_locations=[str(wan_nodes_py.parent)]
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load WanVideo nodes at {wan_nodes_py}")
        
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "wan_nodes"
        module.__path__ = [str(wan_nodes_py.parent)]
        sys.modules["wan_nodes"] = module
        spec.loader.exec_module(module)
        
        self.wan = module
        
        # Disable latent preview
        self._disable_latent_preview()
        print("✓ WanVideo nodes loaded successfully")
    
    def _disable_latent_preview(self):
        """Disable latent preview to avoid server import issues."""
        try:
            lp = importlib.import_module("ComfyUI-WanVideoWrapper.latent_preview")
            def _disabled_callback(model, steps, x0_output_dict=None):
                def _cb(step, x0, x, total_steps):
                    return None
                return _cb
            setattr(lp, 'prepare_callback', _disabled_callback)
        except Exception:
            pass
    
    def _load_state(self):
        """Load generator state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.current_frame_idx = state.get('current_frame_idx', 0)
                self.generation_count = state.get('generation_count', 0)
                if state.get('current_output_dir'):
                    self.current_output_dir = Path(state['current_output_dir'])
                print(f"✓ Loaded state: frame_idx={self.current_frame_idx}, generations={self.generation_count}")
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
    
    def _save_state(self):
        """Save generator state to file."""
        Path(self.config.output_base_dir).mkdir(exist_ok=True)
        state = {
            'current_frame_idx': self.current_frame_idx,
            'generation_count': self.generation_count,
            'current_output_dir': str(self.current_output_dir) if self.current_output_dir else None
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def print_vram_usage(self, step_name: str):
        """Print current VRAM usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{step_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            print(f"{step_name}: CUDA not available")
    
    @contextmanager
    def memory_context(self, step_name: str = ""):
        """Memory management context."""
        self.print_vram_usage(f"Before {step_name}")
        try:
            yield
        finally:
            self.mm.soft_empty_cache()
            self.print_vram_usage(f"After {step_name}")
    
    def load_models(self, force_reload: bool = False):
        """Load all models needed for generation."""
        if self.models_loaded and not force_reload:
            print("✓ Models already loaded, skipping")
            return
        
        print("Loading models...")
        
        # Load T5 text encoder
        print("  Loading T5 text encoder...")
        with self.memory_context("T5 loading"):
            text_loader = self.wan.LoadWanVideoT5TextEncoder()
            t5_model, = text_loader.loadmodel(
                model_name=self.config.t5_name,
                precision="bf16",
                load_device="offload_device",
                quantization="disabled",
            )
        
        # Encode text (this is typically the same for all generations)
        print("  Encoding text...")
        with self.memory_context("Text encoding"):
            text_encode = self.wan.WanVideoTextEncode()
            self.text_embeds, = text_encode.process(
                t5=t5_model,
                positive_prompt="bellowing flames",  # Make configurable if needed
                negative_prompt="",
                force_offload=True,
            )
        
        # Unload T5 to save memory
        del t5_model
        self.mm.soft_empty_cache()
        
        # Load LoRAs
        print("  Loading LoRAs...")
        with self.memory_context("LoRA loading"):
            base_lora_node = self.wan.WanVideoLoraSelect()
            base_lora, = base_lora_node.getlorapath(
                lora=self.config.base_lora,
                strength=self.config.base_lora_strength,
                low_mem_load=False,
            )
            
            extra_lora_node = self.wan.WanVideoLoraSelect()
            extra_lora, = extra_lora_node.getlorapath(
                lora=self.config.extra_lora,
                strength=self.config.extra_lora_strength,
                prev_lora=base_lora,
                low_mem_load=False,
            )
            
            self.lora_config = extra_lora
        
        # Block swap setup
        print("  Setting up block swap...")
        with self.memory_context("Block swap"):
            blockswap_node = self.wan.WanVideoBlockSwap()
            blockswap_args, = blockswap_node.setargs(
                blocks_to_swap=self.config.blocks_to_swap,
                offload_img_emb=False,
                offload_txt_emb=False,
                use_non_blocking=True,
                vace_blocks_to_swap=self.config.vace_blocks_to_swap,
            )
        
        # VACE model
        print("  Setting up VACE...")
        with self.memory_context("VACE setup"):
            vace_select_node = self.wan.WanVideoVACEModelSelect()
            vace_model_path, = vace_select_node.getvacepath(
                vace_model=self.config.vace_name,
            )
        
        # Load main model
        print("  Loading main WanVideo model...")
        with self.memory_context("Main model loading"):
            model_loader = self.wan.WanVideoModelLoader()
            self.video_model, = model_loader.loadmodel(
                model=self.config.model_name,
                base_precision="bf16",
                quantization="fp8_e4m3fn",
                load_device="offload_device",
                attention_mode="sdpa",
                block_swap_args=blockswap_args,
                lora=self.lora_config,
                vace_model=vace_model_path,
            )
        
        # Load VAE
        print("  Loading VAE...")
        with self.memory_context("VAE loading"):
            vae_loader = self.wan.WanVideoVAELoader()
            self.vae_model, = vae_loader.loadmodel(
                model_name=self.config.vae_name,
                precision="bf16",
            )
        
        # Setup helper arguments
        print("  Setting up helper arguments...")
        with self.memory_context("Helper args"):
            # FETA args
            feta_node = self.wan.WanVideoEnhanceAVideo()
            feta_args, = feta_node.setargs(
                weight=self.config.feta_weight,
                start_percent=0.0,
                end_percent=1.0
            )
            
            # SLG args
            slg_node = self.wan.WanVideoSLG()
            slg_args, = slg_node.process(
                blocks=self.config.slg_blocks,
                start_percent=self.config.slg_start_percent,
                end_percent=1.0
            )
            
            # Experimental args
            exp_node = self.wan.WanVideoExperimentalArgs()
            exp_args, = exp_node.process(
                video_attention_split_steps="",
                cfg_zero_star=self.config.cfg_zero_star,
                use_zero_init=False,
                zero_star_steps=0,
                use_fresca=False,
                fresca_scale_low=1.0,
                fresca_scale_high=1.2,
                fresca_freq_cutoff=20,
            )
            
            self.helper_args = {
                'feta_args': feta_args,
                'slg_args': slg_args,
                'experimental_args': exp_args
            }
        
        self.models_loaded = True
        print("✓ All models loaded successfully")
    
    def unload_models(self):
        """Unload all models to free memory."""
        if not self.models_loaded:
            return
        
        print("Unloading models...")
        
        if self.video_model is not None:
            del self.video_model
            self.video_model = None
        
        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
        
        if self.text_embeds is not None:
            del self.text_embeds
            self.text_embeds = None
        
        self.lora_config = None
        self.helper_args = None
        
        # Use ComfyUI's model unloading if available
        if hasattr(self.mm, 'unload_all_models'):
            self.mm.unload_all_models()
        else:
            self.mm.soft_empty_cache()
        
        self.models_loaded = False
        print("✓ Models unloaded")
    
    def generate_initial(self, video_path: str, prompt: str = "bellowing flames", 
                        output_name: str = "initial") -> Path:
        """Generate initial video frames."""
        print("\n" + "="*60)
        print("INITIAL VIDEO GENERATION")
        print("="*60)
        
        self.print_vram_usage("Initial state")
        
        # Reset state for new generation
        self.current_frame_idx = 0
        self.generation_count = 0
        
        # Create output directory
        output_dir = Path(self.config.output_base_dir) / f"{output_name}_{self.generation_count:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_output_dir = output_dir
        
        # Load models
        self.load_models()
        
        # Process video and generate
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Load and process video frames
        print("Processing input video...")
        with self.memory_context("Video processing"):
            video_frames, video_masks = self._load_video_frames(
                video_path, start_frame=0, num_frames=self.config.initial_frames
            )
            
            # Save debug frames
            self._save_debug_frames(video_frames, video_masks, output_dir / "debug_input")
        
        # VACE encoding
        print("VACE encoding...")
        with self.memory_context("VACE encoding"):
            vace_encode = self.wan.WanVideoVACEEncode()
            image_embeds, = vace_encode.process(
                vae=self.vae_model,
                width=self.config.width,
                height=self.config.height,
                num_frames=self.config.initial_frames,
                strength=1.0,
                vace_start_percent=0.0,
                vace_end_percent=1.0,
                input_frames=video_frames,
                input_masks=video_masks,
            )
        
        # Clean up input data
        del video_frames, video_masks
        self.mm.soft_empty_cache()
        
        # Sampling
        print("Sampling...")
        with self.memory_context("Sampling"):
            latents = self._sample_latents(image_embeds, self.config.base_seed)
        
        # Decode
        print("Decoding...")
        with self.memory_context("Decoding"):
            images = self._decode_latents(latents)
        
        # Save frames
        print("Saving frames...")
        saved_files = self._save_frames(images, output_dir)
        
        # Create video
        video_file = self._create_video(output_dir, f"{output_name}_initial.mp4")
        
        # Update state
        self.current_frame_idx = self.config.initial_frames
        self.generation_count += 1
        self._save_state()
        
        print(f"\n✅ Initial generation complete!")
        print(f"📁 Output: {output_dir}")
        print(f"🎬 Video: {video_file}")
        print(f"📊 Frames generated: {len(saved_files)}")
        
        return output_dir
    
    def extend_video(self, video_path: Optional[str] = None, num_extensions: int = 1,
                    extension_frames: Optional[int] = None, 
                    context_frames: Optional[int] = None) -> List[Path]:
        """Extend video with one or more extension passes."""
        if self.current_output_dir is None:
            raise ValueError("No initial generation found. Run generate_initial() first.")
        
        extension_frames = extension_frames or self.config.extension_frames
        context_frames = context_frames or self.config.context_frames
        
        print(f"\n" + "="*60)
        print(f"VIDEO EXTENSION ({num_extensions} passes)")
        print("="*60)
        
        # Load models if not already loaded
        self.load_models()
        
        generated_dirs = []
        
        for ext_num in range(num_extensions):
            print(f"\n--- Extension {ext_num + 1}/{num_extensions} ---")
            
            # Generate extension
            ext_dir = self._generate_extension(
                video_path=video_path,
                extension_frames=extension_frames,
                context_frames=context_frames,
                extension_num=ext_num
            )
            
            generated_dirs.append(ext_dir)
            
            # Update state for next extension
            self.current_output_dir = ext_dir
            self.current_frame_idx += extension_frames
            self.generation_count += 1
            self._save_state()
        
        print(f"\n✅ All extensions complete!")
        print(f"📊 Total extensions: {len(generated_dirs)}")
        for i, dir_path in enumerate(generated_dirs):
            print(f"   {i+1}. {dir_path}")
        
        return generated_dirs
    
    def _generate_extension(self, video_path: Optional[str], extension_frames: int,
                           context_frames: int, extension_num: int) -> Path:
        """Generate a single extension."""
        # Create output directory
        output_dir = Path(self.config.output_base_dir) / f"extension_{self.generation_count:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extension output: {output_dir}")
        
        # Load previous frames
        print("Loading previous frames...")
        previous_frames = self._load_previous_frames(self.current_output_dir, context_frames)
        
        # Load new video chunk if video path provided
        if video_path:
            print("Loading new video chunk...")
            new_video_frames, new_video_masks = self._load_video_frames(
                Path(video_path), 
                start_frame=self.current_frame_idx,
                num_frames=extension_frames
            )
        else:
            # Generate blank frames for extension without new video input
            print("Generating blank frames for extension...")
            new_video_frames = torch.zeros(
                (extension_frames, self.config.height, self.config.width, 3),
                dtype=torch.float32
            )
            new_video_masks = torch.zeros(
                (extension_frames, self.config.height, self.config.width),
                dtype=torch.float32
            )
        
        # Prepare extension inputs
        print("Preparing extension inputs...")
        input_frames, input_masks = self._prepare_extension_inputs(
            previous_frames, new_video_frames, new_video_masks
        )
        
        # Save debug frames
        self._save_debug_frames(input_frames, input_masks, output_dir / "debug_extension")
        
        # Clean up intermediate data
        del previous_frames, new_video_frames, new_video_masks
        self.mm.soft_empty_cache()
        
        # VACE encoding
        print("VACE encoding extension...")
        with self.memory_context("Extension VACE encoding"):
            vace_encode = self.wan.WanVideoVACEEncode()
            image_embeds, = vace_encode.process(
                vae=self.vae_model,
                width=self.config.width,
                height=self.config.height,
                num_frames=context_frames + extension_frames,
                strength=1.0,
                vace_start_percent=0.0,
                vace_end_percent=1.0,
                input_frames=input_frames,
                input_masks=input_masks,
            )
        
        # Clean up input data
        del input_frames, input_masks
        self.mm.soft_empty_cache()
        
        # Sampling with varied seed
        print("Sampling extension...")
        extension_seed = self.config.base_seed + self.current_frame_idx
        with self.memory_context("Extension sampling"):
            latents = self._sample_latents(image_embeds, extension_seed)
        
        # Decode
        print("Decoding extension...")
        with self.memory_context("Extension decoding"):
            images = self._decode_latents(latents)
        
        # Save frames
        print("Saving extension frames...")
        saved_files = self._save_frames(images, output_dir)
        
        # Create video
        video_file = self._create_video(output_dir, f"extension_{self.generation_count:03d}.mp4")
        
        print(f"✅ Extension complete: {len(saved_files)} frames")
        print(f"📁 Output: {output_dir}")
        print(f"🎬 Video: {video_file}")
        
        return output_dir
    
    def _load_video_frames(self, video_path: Path, start_frame: int, num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load video frames and create masks."""
        print(f"Loading video frames: {start_frame} to {start_frame + num_frames}")
        
        try:
            import imageio.v2 as iio
            reader = iio.get_reader(str(video_path), "ffmpeg")
            frames_list = []
            
            for idx, frame in enumerate(reader):
                if idx < start_frame:
                    continue
                if len(frames_list) >= num_frames:
                    break
                frame_rgb = frame[..., :3]
                frames_list.append(frame_rgb)
            reader.close()
            
        except Exception:
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                
                # Skip to start frame
                for _ in range(start_frame):
                    ret, _ = cap.read()
                    if not ret:
                        break
                
                frames_list = []
                for _ in range(num_frames):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame_rgb)
                cap.release()
                
            except ImportError:
                raise RuntimeError("Either imageio or opencv-python must be installed")
        
        if not frames_list:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Pad if necessary
        if len(frames_list) < num_frames:
            last_frame = frames_list[-1]
            for _ in range(num_frames - len(frames_list)):
                frames_list.append(last_frame.copy())
        
        # Resize to target size
        resized_frames = []
        for frame in frames_list:
            try:
                import cv2
                resized = cv2.resize(frame, (self.config.width, self.config.height), interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized)
            except ImportError:
                # Fallback to PIL
                pil_frame = Image.fromarray(frame)
                resized_pil = pil_frame.resize((self.config.width, self.config.height), Image.BICUBIC)
                resized_frames.append(np.array(resized_pil))
        
        # Convert to tensors
        frames_np = np.stack(resized_frames).astype(np.float32) / 255.0
        masks_np = np.mean(frames_np, axis=-1)  # Intensity masks
        
        # Create grey composite
        grey_val = ((self.config.grey_color) & 0xFF) / 255.0
        grey_frames = np.full_like(frames_np, grey_val)
        mask_expanded = masks_np[..., None]
        frames_np = grey_frames * mask_expanded
        
        frames_tensor = torch.from_numpy(frames_np)
        masks_tensor = torch.from_numpy(masks_np)
        
        return frames_tensor, masks_tensor
    
    def _load_previous_frames(self, output_dir: Path, num_frames: int) -> torch.Tensor:
        """Load the last N frames from previous generation."""
        frame_files = sorted(list(output_dir.glob("frame_*.png")))
        if len(frame_files) < num_frames:
            raise ValueError(f"Need at least {num_frames} frames, found {len(frame_files)}")
        
        last_frame_files = frame_files[-num_frames:]
        
        frames = []
        for frame_file in last_frame_files:
            img = Image.open(frame_file)
            arr = np.array(img).astype(np.float32) / 255.0
            frames.append(arr)
        
        return torch.from_numpy(np.stack(frames))
    
    def _prepare_extension_inputs(self, previous_frames: torch.Tensor, 
                                 new_video_frames: torch.Tensor, 
                                 new_video_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare extension inputs and masks."""
        # Create grey composite for new frames
        grey_val = ((self.config.grey_color) & 0xFF) / 255.0
        height, width = new_video_frames.shape[1:3]
        
        grey_frames = torch.full((len(new_video_frames), height, width, 3), grey_val, dtype=torch.float32)
        mask_expanded = new_video_masks.unsqueeze(-1).expand(-1, -1, -1, 3)
        new_grey_composite = grey_frames * mask_expanded
        
        # Combine frames: [previous] + [new grey composite]
        input_frames = torch.cat([previous_frames, new_grey_composite], dim=0)
        
        # Create masks: [black for previous] + [real masks for new]
        black_masks = torch.zeros((len(previous_frames), height, width), dtype=torch.float32)
        input_masks = torch.cat([black_masks, new_video_masks], dim=0)
        
        return input_frames, input_masks
    
    def _sample_latents(self, image_embeds: torch.Tensor, seed: int) -> torch.Tensor:
        """Sample latents using the WanVideo sampler."""
        sampler = self.wan.WanVideoSampler()
        latents, = sampler.process(
            model=self.video_model,
            image_embeds=image_embeds,
            steps=self.config.steps,
            cfg=self.config.cfg,
            shift=self.config.shift,
            seed=seed,
            force_offload=True,
            scheduler=self.config.scheduler,
            riflex_freq_index=0,
            denoise_strength=1.0,
            batched_cfg=False,
            rope_function="comfy",
            text_embeds=self.text_embeds,
            feta_args=self.helper_args['feta_args'],
            slg_args=self.helper_args['slg_args'],
            experimental_args=self.helper_args['experimental_args'],
        )
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE."""
        vae_decode = self.wan.WanVideoDecode()
        images, = vae_decode.decode(
            vae=self.vae_model,
            samples=latents,
            enable_vae_tiling=False,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
        )
        return images
    
    def _save_frames(self, images: torch.Tensor, output_dir: Path) -> List[str]:
        """Save frames to disk."""
        saved_files = []
        
        for idx, frame in enumerate(images):
            np_img = (frame.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(np_img)
            fname = output_dir / f"frame_{idx:04d}.png"
            img.save(fname)
            saved_files.append(str(fname))
        
        return saved_files
    
    def _save_debug_frames(self, frames: torch.Tensor, masks: torch.Tensor, debug_dir: Path):
        """Save debug frames for inspection."""
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a few representative frames
        indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        
        for i in indices:
            if i < len(frames):
                # Save frame
                frame_arr = (frames[i].numpy() * 255).astype(np.uint8)
                Image.fromarray(frame_arr).save(debug_dir / f"frame_{i:04d}.png")
                
                # Save mask
                mask_arr = (masks[i].numpy() * 255).astype(np.uint8)
                Image.fromarray(mask_arr, mode="L").save(debug_dir / f"mask_{i:04d}.png")
    
    def _create_video(self, output_dir: Path, video_name: str) -> Path:
        """Create video from frames using ffmpeg."""
        video_path = output_dir / video_name
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.config.frame_rate),
            "-i", "frame_%04d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            video_name,
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, cwd=output_dir)
            print(f"✅ Video created: {video_path}")
        except FileNotFoundError:
            print("⚠️ ffmpeg not found. Install ffmpeg to create videos.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ ffmpeg failed: {e}")
        
        return video_path
    
    def get_status(self) -> dict:
        """Get current generator status."""
        return {
            'current_frame_idx': self.current_frame_idx,
            'generation_count': self.generation_count,
            'current_output_dir': str(self.current_output_dir) if self.current_output_dir else None,
            'models_loaded': self.models_loaded,
            'config': asdict(self.config)
        }
    
    def reset(self):
        """Reset generator state."""
        self.current_frame_idx = 0
        self.generation_count = 0
        self.current_output_dir = None
        self.unload_models()
        
        # Remove state file
        if self.state_file.exists():
            self.state_file.unlink()
        
        print("✅ Generator reset")

def main():
    """Example usage of the WanVideoGenerator."""
    # Create generator
    generator = WanVideoGenerator()
    
    # Generate initial video
    initial_dir = generator.generate_initial("something_about_us_video.mp4")
    
    # Extend with 3 more generations
    extension_dirs = generator.extend_video(
        video_path="something_about_us_video.mp4",
        num_extensions=3
    )
    
    # Print final status
    status = generator.get_status()
    print(f"\n📊 Final Status:")
    print(f"   Total generations: {status['generation_count']}")
    print(f"   Current frame index: {status['current_frame_idx']}")
    print(f"   Latest output: {status['current_output_dir']}")
    
    # Optionally unload models to free memory
    generator.unload_models()

if __name__ == "__main__":
    main() 