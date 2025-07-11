#!/usr/bin/env python3
"""run_wan_video_simple.py
Simple runner that fixes import issues by adding current directory to Python path.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path so we can import wan_video_generator
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now try to import
try:
    from wan_video_generator import WanVideoGenerator, GenerationConfig
    print("✅ Successfully imported WanVideoGenerator")
except ImportError as e:
    print(f"❌ Failed to import WanVideoGenerator: {e}")
    print("Current directory:", current_dir)
    print("Python path:", sys.path)
    sys.exit(1)

def main():
    """Simple main function to test the generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WanVideo Generator - Simple Runner")
    parser.add_argument("--video", "-v", help="Input video path")
    parser.add_argument("--extensions", "-e", type=int, default=3, help="Number of extensions")
    parser.add_argument("--initial-only", action="store_true", help="Only generate initial frames")
    parser.add_argument("--test-import", action="store_true", help="Just test imports")
    
    args = parser.parse_args()
    
    if args.test_import:
        print("✅ Import test successful!")
        print("Generator class:", WanVideoGenerator)
        print("Config class:", GenerationConfig)
        return
    
    if not args.video:
        print("❌ Error: --video argument is required")
        parser.print_help()
        return
    
    # Create generator
    try:
        generator = WanVideoGenerator()
        print("✅ Generator created successfully")
        
        # Generate initial
        print(f"🎬 Generating initial frames from {args.video}")
        initial_dir = generator.generate_initial(args.video)
        print(f"✅ Initial generation complete: {initial_dir}")
        
        if not args.initial_only:
            # Generate extensions
            print(f"🔄 Generating {args.extensions} extensions")
            extension_dirs = generator.extend_video(
                video_path=args.video,
                num_extensions=args.extensions
            )
            print(f"✅ Extensions complete: {len(extension_dirs)} generated")
        
        # Show status
        status = generator.get_status()
        print(f"\n📊 Final Status:")
        print(f"   Total frames: {status['current_frame_idx']}")
        print(f"   Total generations: {status['generation_count']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 