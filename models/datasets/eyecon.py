import os
from .utils import Datum, DatasetBase

template = ['a photo of {}.']  # Template for CLIP text prompts

class Eyecon(DatasetBase):
    dataset_dir = 'dataset'  # Your main dataset directory name

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template
        
        # Define class names and their corresponding labels
        self.class_names = {
            'student being punished': 0,
            'students fighting': 1,
            'teacher checking notebook': 2,
            'teacher using mobile phones': 3
        }
        
        # Generate training data
        train_items = self._read_data()
        
        # Since we don't have validation and test sets, we'll use the same data
        # This is not ideal for real evaluation, but matches your current setup
        val_items = train_items
        test_items = train_items
        
        # Initialize with generated data
        super().__init__(train_x=train_items, val=val_items, test=test_items)
    
    def _read_data(self):
        """Read and organize the Eyecon dataset."""
        items = []
        
        # Iterate through each class folder
        for class_name, label in self.class_names.items():
            class_dir = os.path.join(self.dataset_dir, class_name)
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Create Datum objects for each image
            for image_file in image_files:
                impath = os.path.join(class_dir, image_file)
                
                # Create a Datum instance for this image
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=class_name,
                    domain=-1  # We don't use domains in this dataset
                )
                items.append(item)
        
        return items

    def generate_fewshot_dataset_(self, num_shots, split):
        """Generate a few-shot dataset for the specified split."""
        print('num_shots is', num_shots)
        
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        else:
            raise ValueError(f"Unsupported split: {split}")
        
        return few_shot_data

    @staticmethod
    def read_split(filepath, image_dir):
        """
        This is a dummy method to maintain compatibility with the original codebase.
        In your case, all data is in the training set.
        """
        raise NotImplementedError(
            "read_split is not implemented for Eyecon dataset as it uses a single training set"
        )