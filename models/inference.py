import argparse
import torch
from PIL import Image
import torch
from PIL import Image
import os
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pathlib import Path

class CLIPLPInference:
    def __init__(self, model_path, num_classes, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load the base CLIP model and preprocessing
        self.base_model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        
        # Add classification head
        self.classifier = torch.nn.Linear(512, num_classes).to(self.device)
        
        # Load LP++ weights
        self.lp_weights = torch.load(model_path, map_location=self.device)
        
        # Apply weights to the model and classifier
        if isinstance(self.lp_weights, dict):
            if 'model_state_dict' in self.lp_weights:
                # Separate CLIP and classifier weights
                clip_weights = {k: v for k, v in self.lp_weights['model_state_dict'].items() 
                              if not k.startswith('classifier')}
                classifier_weights = {k.replace('classifier.', ''): v 
                                   for k, v in self.lp_weights['model_state_dict'].items() 
                                   if k.startswith('classifier')}
                
                self.base_model.load_state_dict(clip_weights, strict=False)
                self.classifier.load_state_dict(classifier_weights)
        
        self.base_model.eval()
        self.classifier.eval()
        
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
        Process a single image and return the features and classification results.
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get image features
        features = self.base_model.encode_image(image)
        
        # Get classification results
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        
        return {
            # 'features': features.cpu(),
            # 'logits': logits.cpu(),
            'probabilities': probs.cpu(),
            'predicted_class': predicted_class.cpu().item()
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
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)
            
            batch_results = [{
                'features': feat.cpu(),
                'logits': log.cpu(),
                'probabilities': prob.cpu(),
                'predicted_class': pred.cpu().item()
            } for feat, log, prob, pred in zip(features, logits, probs, predicted_classes)]
                
            all_results.extend(batch_results)
            
        return all_results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLIP LP++ Inference Script")
    parser.add_argument("--frame", type=str, nargs='+', help="Paths to frame for inference")
    args = parser.parse_args()

    # Initialize inference class
    inferencer = CLIPLPInference('outputs/best_lp_model_16shots.pt', 4)

    # Perform inference on each image path provided
    for image_path in args.frame:
        predicted_class = inferencer.process_image(image_path)
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()