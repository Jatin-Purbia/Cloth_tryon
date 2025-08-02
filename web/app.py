#!/usr/bin/env python3
"""
Web interface for Virtual Try-On System.
Provides real-time try-on capabilities with modern UI.
"""

import os
import sys
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.try_on import VirtualTryOnInference, MultiGarmentTryOnInference


class VirtualTryOnWebApp:
    """
    Proprietary web interface for virtual try-on.
    
    Patentable Features:
    - Real-time try-on with progressive feedback
    - Multi-garment interface
    - Quality assessment display
    - Performance optimization for web deployment
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        
        # Initialize inference engines
        self.single_inference = VirtualTryOnInference(
            model_path=model_path,
            device=device,
            use_real_time=True
        )
        
        self.multi_inference = MultiGarmentTryOnInference(
            model_path=model_path,
            device=device,
            max_garments=3
        )
        
        # Performance tracking
        self.total_requests = 0
        self.avg_inference_time = 0.0
        
    def single_garment_try_on(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_mask: np.ndarray,
        use_fast_mode: bool = True
    ) -> tuple:
        """
        Perform single garment try-on.
        
        Args:
            person_image: Person image
            garment_image: Garment image
            garment_mask: Garment mask
            use_fast_mode: Whether to use fast mode
        
        Returns:
            Tuple of (result_image, performance_info)
        """
        try:
            # Perform try-on
            result = self.single_inference.try_on(
                person_image=person_image,
                garment_image=garment_image,
                garment_mask=garment_mask,
                use_fast_mode=use_fast_mode
            )
            
            # Update performance stats
            self.total_requests += 1
            self.avg_inference_time = (
                (self.avg_inference_time * (self.total_requests - 1) + result['inference_time']) 
                / self.total_requests
            )
            
            # Prepare performance info
            performance_info = f"""
            **Performance Statistics:**
            - Current inference time: {result['inference_time']:.3f}s
            - Average inference time: {self.avg_inference_time:.3f}s
            - Total requests: {self.total_requests}
            - Quality score: {result.get('quality_score', 'N/A')}
            """
            
            return result['result_image'], performance_info
            
        except Exception as e:
            error_msg = f"Error during try-on: {str(e)}"
            return None, error_msg
    
    def multi_garment_try_on(
        self,
        person_image: np.ndarray,
        garment_images: list,
        garment_masks: list,
        garment_types: list
    ) -> tuple:
        """
        Perform multi-garment try-on.
        
        Args:
            person_image: Person image
            garment_images: List of garment images
            garment_masks: List of garment masks
            garment_types: List of garment types
        
        Returns:
            Tuple of (result_image, performance_info)
        """
        try:
            # Perform multi-garment try-on
            result = self.multi_inference.try_on_multiple_garments(
                person_image=person_image,
                garment_images=garment_images,
                garment_masks=garment_masks,
                garment_types=garment_types
            )
            
            # Update performance stats
            self.total_requests += 1
            
            # Prepare performance info
            performance_info = f"""
            **Multi-Garment Try-On Completed:**
            - Total requests: {self.total_requests}
            - Garments processed: {len(garment_images)}
            - Garment types: {', '.join(garment_types)}
            """
            
            return result['final_result'], performance_info
            
        except Exception as e:
            error_msg = f"Error during multi-garment try-on: {str(e)}"
            return None, error_msg
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Virtual Try-On System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🧥 Virtual Try-On System</h1>
                <p>State-of-the-art virtual clothes try-on with proprietary innovations</p>
            </div>
            """)
            
            # Main interface
            with gr.Tabs():
                
                # Single Garment Tab
                with gr.TabItem("Single Garment Try-On"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input Images")
                            
                            person_input = gr.Image(
                                label="Person Image",
                                type="numpy",
                                height=300
                            )
                            
                            garment_input = gr.Image(
                                label="Garment Image",
                                type="numpy",
                                height=300
                            )
                            
                            mask_input = gr.Image(
                                label="Garment Mask",
                                type="numpy",
                                height=300
                            )
                            
                            fast_mode = gr.Checkbox(
                                label="Fast Mode (Real-time)",
                                value=True,
                                info="Enable for faster inference with slight quality trade-off"
                            )
                            
                            try_on_btn = gr.Button(
                                "Try On Garment",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Try-On Result")
                            
                            result_image = gr.Image(
                                label="Result",
                                height=400
                            )
                            
                            performance_info = gr.Markdown(
                                label="Performance Information"
                            )
                    
                    # Connect button
                    try_on_btn.click(
                        fn=self.single_garment_try_on,
                        inputs=[person_input, garment_input, mask_input, fast_mode],
                        outputs=[result_image, performance_info]
                    )
                
                # Multi-Garment Tab
                with gr.TabItem("Multi-Garment Try-On"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input Images")
                            
                            multi_person_input = gr.Image(
                                label="Person Image",
                                type="numpy",
                                height=300
                            )
                            
                            # Multiple garment inputs
                            garment_inputs = []
                            mask_inputs = []
                            type_inputs = []
                            
                            for i in range(3):
                                with gr.Group(f"Garment {i+1}"):
                                    garment_inputs.append(
                                        gr.Image(
                                            label=f"Garment {i+1} Image",
                                            type="numpy",
                                            height=200
                                        )
                                    )
                                    mask_inputs.append(
                                        gr.Image(
                                            label=f"Garment {i+1} Mask",
                                            type="numpy",
                                            height=200
                                        )
                                    )
                                    type_inputs.append(
                                        gr.Dropdown(
                                            choices=["shirt", "pants", "jacket", "dress", "empty"],
                                            label=f"Garment {i+1} Type",
                                            value="empty"
                                        )
                                    )
                            
                            multi_try_on_btn = gr.Button(
                                "Try On Multiple Garments",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Multi-Garment Result")
                            
                            multi_result_image = gr.Image(
                                label="Result",
                                height=400
                            )
                            
                            multi_performance_info = gr.Markdown(
                                label="Performance Information"
                            )
                    
                    # Connect button
                    multi_try_on_btn.click(
                        fn=self.multi_garment_try_on,
                        inputs=[multi_person_input] + garment_inputs + mask_inputs + type_inputs,
                        outputs=[multi_result_image, multi_performance_info]
                    )
                
                # Batch Processing Tab
                with gr.TabItem("Batch Processing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Batch Input")
                            
                            batch_person_files = gr.File(
                                label="Person Images",
                                file_count="multiple",
                                file_types=["image"]
                            )
                            
                            batch_garment_files = gr.File(
                                label="Garment Images",
                                file_count="multiple",
                                file_types=["image"]
                            )
                            
                            batch_mask_files = gr.File(
                                label="Garment Masks",
                                file_count="multiple",
                                file_types=["image"]
                            )
                            
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Batch Size",
                                info="Number of images to process simultaneously"
                            )
                            
                            batch_process_btn = gr.Button(
                                "Process Batch",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Batch Results")
                            
                            batch_results = gr.Gallery(
                                label="Results",
                                height=400,
                                columns=2
                            )
                            
                            batch_performance_info = gr.Markdown(
                                label="Batch Performance Information"
                            )
                    
                    # Connect button
                    batch_process_btn.click(
                        fn=self.batch_process,
                        inputs=[batch_person_files, batch_garment_files, batch_mask_files, batch_size],
                        outputs=[batch_results, batch_performance_info]
                    )
                
                # Settings Tab
                with gr.TabItem("Settings & Info"):
                    gr.Markdown("### System Information")
                    
                    gr.Markdown(f"""
                    **Model Configuration:**
                    - Device: {self.device}
                    - Model Path: {self.model_path}
                    - Total Requests: {self.total_requests}
                    - Average Inference Time: {self.avg_inference_time:.3f}s
                    
                    **Patentable Features:**
                    - Multi-stage pose refinement
                    - Texture-aware garment warping
                    - Occlusion-aware composition
                    - Real-time optimization
                    - Quality assessment
                    
                    **Performance Optimizations:**
                    - Model compression for web deployment
                    - Adaptive computation based on input complexity
                    - Progressive refinement for user feedback
                    - Batch processing capabilities
                    """)
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>© 2024 Virtual Try-On System - Proprietary Technology</p>
                <p>Built with state-of-the-art deep learning and patentable innovations</p>
            </div>
            """)
        
        return interface
    
    def batch_process(self, person_files, garment_files, mask_files, batch_size):
        """Process batch of images."""
        try:
            # Load images
            person_images = []
            garment_images = []
            mask_images = []
            
            for person_file, garment_file, mask_file in zip(person_files, garment_files, mask_files):
                # Load images (simplified - in practice you'd handle file loading properly)
                person_img = np.array(Image.open(person_file.name))
                garment_img = np.array(Image.open(garment_file.name))
                mask_img = np.array(Image.open(mask_file.name))
                
                person_images.append(person_img)
                garment_images.append(garment_img)
                mask_images.append(mask_img)
            
            # Process batch
            results = self.single_inference.batch_try_on(
                person_images=person_images,
                garment_images=garment_images,
                garment_masks=mask_images,
                batch_size=batch_size
            )
            
            # Prepare results
            result_images = [result['result_image'] for result in results]
            performance_info = f"Processed {len(results)} images in batch"
            
            return result_images, performance_info
            
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            return [], error_msg


def main():
    """Main function to run the web app."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On Web App')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the web app on')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link')
    
    args = parser.parse_args()
    
    # Create web app
    app = VirtualTryOnWebApp(
        model_path=args.model_path,
        device=args.device
    )
    
    # Create interface
    interface = app.create_interface()
    
    # Launch app
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main() 