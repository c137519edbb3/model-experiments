import torch
from PIL import Image
import os
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pathlib import Path

class CLIPLPInference:
    def __init__(self, model_path, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the base CLIP model and preprocessing
        self.base_model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        
        # Load LP++ weights
        self.lp_weights = torch.load(model_path, map_location=self.device)
        
        # Apply weights to the model
        if isinstance(self.lp_weights, dict):
            if 'model_state_dict' in self.lp_weights:
                self.lp_weights = self.lp_weights['model_state_dict']
            self.base_model.load_state_dict(self.lp_weights, strict=False)
        
        self.base_model.eval()
        
        # Define standard CLIP preprocessing
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])

    @torch.no_grad()
    def process_image(self, image_path):
        """
        Process a single image and return the features.
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get image features
        features = self.base_model.encode_image(image)
        
        # If the model has LP-specific forward method
        if hasattr(self.base_model, 'forward_lp'):
            logits = self.base_model.forward_lp(features)
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            
            return {
                'features': features.cpu(),
                'logits': logits.cpu(),
                'probabilities': probs.cpu(),
                'predicted_class': predicted_class.cpu().item()
            }
        
        # If no LP-specific forward method, return features only
        return {
            'features': features.cpu()
        }

    @torch.no_grad()
    def batch_process(self, image_paths, batch_size=32):
        """
        Process multiple images in batches
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [Image.open(path).convert('RGB') for path in batch_paths]
            batch_tensors = torch.stack([self.transform(img) for img in batch_images]).to(self.device)
            
            features = self.base_model.encode_image(batch_tensors)
            
            if hasattr(self.base_model, 'forward_lp'):
                logits = self.base_model.forward_lp(features)
                probs = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
                
                batch_results = [{
                    'features': feat.cpu(),
                    'logits': log.cpu(),
                    'probabilities': prob.cpu(),
                    'predicted_class': pred.cpu().item()
                } for feat, log, prob, pred in zip(features, logits, probs, predicted_classes)]
            else:
                batch_results = [{'features': feat.cpu()} for feat in features]
                
            all_results.extend(batch_results)
            
        return all_results

def main():
    # Example usage
    model_path = 'outputs/best_lp_model_16shots.pt'
    inferencer = CLIPLPInference(model_path)
    
    # Single image inference
    image_path = 'path/to/test/image.jpg'
    result = inferencer.process_image(image_path)
    
    if 'predicted_class' in result:
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Image Features Shape: {result['features'].shape}")
    
    # Batch processing example
    image_paths = ['path1.jpg', 'path2.jpg', 'path3.jpg']
    batch_results = inferencer.batch_process(image_paths)

if __name__ == '__main__':
    main()