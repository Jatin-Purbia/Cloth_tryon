import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow for Google Colab

def segment_person(image_path):
    # Load pre-trained DeepLabV3+ model with ResNet-101 backbone
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare image for the model
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image_rgb).unsqueeze(0)
    
    # Generate prediction
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Process output
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Extract person class (class 15 is for person in PASCAL VOC)
    # In COCO dataset used by DeepLabV3+, person is class 15
    person_mask = (output_predictions == 15).astype(np.uint8)
    
    # Create RGB person mask (3 channels)
    person_mask_rgb = np.stack([person_mask, person_mask, person_mask], axis=2)
    
    # Extract person from the image by applying the mask
    person_only = image_rgb * person_mask_rgb
    
    # Create white background for transparency
    white_background = np.ones_like(image_rgb) * 255
    background = white_background * (1 - person_mask_rgb)
    
    # Combine person with white background
    result = person_only + background
    
    return person_only, person_mask * 255, result, image_rgb

# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print('Usage: python person_segmentation.py <path_to_image>')
        sys.exit(1)
    
    image_path = sys.argv[1]
    person_only, mask, result_with_background, original = segment_person(image_path)
    
    # Convert back to BGR for OpenCV display (if not in Colab)
    person_only_bgr = cv2.cvtColor(person_only.astype(np.uint8), cv2.COLOR_RGB2BGR)
    result_bgr = cv2.cvtColor(result_with_background.astype(np.uint8), cv2.COLOR_RGB2BGR)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    # For Colab usage
    print('Original Image:')
    cv2_imshow(original)
    
    print('Person Mask:')
    cv2_imshow(mask)
    
    print('Person Only (with transparent background):')
    cv2_imshow(person_only)
    
    print('Person with White Background:')
    cv2_imshow(result_with_background)
    
    # For non-Colab usage (commented out)
    # cv2.imshow('Original', original_bgr)
    # cv2.imshow('Person Mask', mask)
    # cv2.imshow('Person Only', person_only_bgr)
    # cv2.imshow('Person with White Background', result_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Optionally save the output
    cv2.imwrite('person_mask.png', mask)
    cv2.imwrite('person_only.png', cv2.cvtColor(person_only.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite('person_with_background.png', cv2.cvtColor(result_with_background.astype(np.uint8), cv2.COLOR_RGB2BGR))
