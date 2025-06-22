# data_loader.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, VALIDATION_SPLIT

class MaterialDataLoader:
    """
    Handles loading and preprocessing of image data for material recognition.
    """
    def __init__(self, data_directory=DATA_DIR, img_height=IMG_HEIGHT,
                 img_width=IMG_WIDTH, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
        """
        Initializes the data loader with specified parameters.

        Args:
            data_directory (str): Path to the root directory containing material subfolders.
            img_height (int): Target height for resizing images.
            img_width (int): Target width for resizing images.
            batch_size (int): Number of samples per batch.
            validation_split (float): Fraction of data to reserve for validation.
        """
        self.data_directory = data_directory
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.class_names = []
        self.num_classes = 0

    def load_data(self):
        """
        Loads training and validation data using ImageDataGenerator.

        Returns:
            tuple: (train_generator, validation_generator, class_names)
        Raises:
            FileNotFoundError: If the data directory does not exist or contains no subdirectories.
        """
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

        print(f"Loading data from: {self.data_directory}")

        # Data augmentation and normalization for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=self.validation_split
        )

        # Only normalization for validation
        validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=self.validation_split)


        train_generator = train_datagen.flow_from_directory(
            self.data_directory,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.data_directory,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False # No need to shuffle validation data
        )

        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)

        if self.num_classes == 0:
            raise ValueError(f"No classes found in {self.data_directory}. Ensure subdirectories exist.")

        print(f"Found {train_generator.samples} training images belonging to {self.num_classes} classes.")
        print(f"Found {validation_generator.samples} validation images belonging to {self.num_classes} classes.")
        print(f"Classes: {self.class_names}")

        return train_generator, validation_generator, self.class_names

# Example Usage (for testing data_loader independently)
if __name__ == "__main__":
    print("Testing MaterialDataLoader...")
    try:
        # Make sure you have a 'data' directory with subfolders for this test
        # Example: data/metal, data/plastic
        # If not, create dummy files to test
        import os
        dummy_data_dir = os.path.join(os.path.dirname(__file__), 'data_test_dummy')
        os.makedirs(os.path.join(dummy_data_dir, 'class_a'), exist_ok=True)
        os.makedirs(os.path.join(dummy_data_dir, 'class_b'), exist_ok=True)
        with open(os.path.join(dummy_data_dir, 'class_a', 'img1.txt'), 'w') as f: f.write('dummy')
        with open(os.path.join(dummy_data_dir, 'class_a', 'img2.txt'), 'w') as f: f.write('dummy')
        with open(os.path.join(dummy_data_dir, 'class_b', 'img1.txt'), 'w') as f: f.write('dummy')


        loader = MaterialDataLoader(data_directory=dummy_data_dir)
        train_gen, val_gen, classes = loader.load_data()
        print(f"Loaded classes: {classes}")
        print("Data loader test successful.")

        # Clean up dummy data
        os.remove(os.path.join(dummy_data_dir, 'class_a', 'img1.txt'))
        os.remove(os.path.join(dummy_data_dir, 'class_a', 'img2.txt'))
        os.remove(os.path.join(dummy_data_dir, 'class_b', 'img1.txt'))
        os.rmdir(os.path.join(dummy_data_dir, 'class_a'))
        os.rmdir(os.path.join(dummy_data_dir, 'class_b'))
        os.rmdir(dummy_data_dir)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you have a 'data' directory set up correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")