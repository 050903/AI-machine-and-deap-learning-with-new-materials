# model_trainer.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, EPOCHS, LEARNING_RATE, MODEL_PATH

class MaterialModelTrainer:
    """
    Manages the creation, compilation, training, and saving of the material recognition model.
    """
    def __init__(self, num_classes, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 img_channels=IMG_CHANNELS, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                 model_save_path=MODEL_PATH):
        """
        Initializes the model trainer.

        Args:
            num_classes (int): The number of distinct material classes.
            img_height (int): Height of input images.
            img_width (int): Width of input images.
            img_channels (int): Number of color channels in input images (e.g., 3 for RGB).
            epochs (int): Number of training epochs.
            learning_rate (float): Initial learning rate for the optimizer.
            model_save_path (str): Path to save the trained model.
        """
        self.num_classes = num_classes
        self.input_shape = (img_height, img_width, img_channels)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.model = None
        self.history = None

    def build_model(self):
        """
        Builds the transfer learning model using MobileNetV2 as a base.
        """
        print("Building model...")
        # Load the pre-trained MobileNetV2 model, excluding the top (classification) layer
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        # Add new custom classification layers on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Global Average Pooling to reduce dimensions
        x = Dense(1024, activation='relu')(x) # A fully connected layer
        predictions = Dense(self.num_classes, activation='softmax')(x) # Output layer

        self.model = Model(inputs=base_model.input, outputs=predictions)
        print("Model built successfully.")
        self.model.summary()

    def compile_model(self):
        """
        Compiles the model with Adam optimizer and categorical crossentropy loss.
        """
        print("Compiling model...")
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print("Model compiled.")

    def train_model(self, train_generator, validation_generator):
        """
        Trains the compiled model using provided data generators.

        Args:
            train_generator: Keras ImageDataGenerator for training data.
            validation_generator: Keras ImageDataGenerator for validation data.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        if train_generator.samples == 0:
            raise ValueError("Training generator has no samples. Please check your data directory.")

        print("Starting model training...")

        # Callbacks for better training control
        checkpoint_cb = ModelCheckpoint(
            self.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        early_stopping_cb = EarlyStopping(
            monitor='val_loss',
            patience=5, # Stop if validation loss doesn't improve for 5 epochs
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr_cb = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2, # Reduce learning rate by 20%
            patience=3, # If validation loss doesn't improve for 3 epochs
            min_lr=1e-6,
            verbose=1
        )

        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
        )
        print("Model training finished.")
        print(f"Best model saved to {self.model_save_path}")
        # Always save the final model after training (in case callback did not save)
        self.model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path} (final model after training)")

    def plot_training_history(self):
        """
        Plots the training and validation accuracy and loss history.
        """
        if self.history is None:
            print("No training history to plot.")
            return

        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

# Example Usage (for testing model_trainer independently)
if __name__ == "__main__":
    print("Testing MaterialModelTrainer...")
    try:
        from data_loader import MaterialDataLoader
        # Create a dummy data directory for testing
        dummy_data_dir = os.path.join(os.path.dirname(__file__), 'data_test_dummy')
        os.makedirs(os.path.join(dummy_data_dir, 'class_a'), exist_ok=True)
        os.makedirs(os.path.join(dummy_data_dir, 'class_b'), exist_ok=True)
        # Create dummy image files (small text files will suffice for flow_from_directory to detect classes)
        for i in range(10):
            with open(os.path.join(dummy_data_dir, 'class_a', f'img_a_{i}.jpg'), 'w') as f: f.write('dummy')
            with open(os.path.join(dummy_data_dir, 'class_b', f'img_b_{i}.jpg'), 'w') as f: f.write('dummy')

        data_loader = MaterialDataLoader(data_directory=dummy_data_dir, batch_size=2)
        train_gen, val_gen, class_names = data_loader.load_data()

        trainer = MaterialModelTrainer(num_classes=len(class_names), epochs=2) # Short epochs for test
        trainer.build_model()
        trainer.compile_model()
        trainer.train_model(train_gen, val_gen)
        trainer.plot_training_history()

        print("Model trainer test successful.")

        # Clean up dummy data
        for i in range(10):
            os.remove(os.path.join(dummy_data_dir, 'class_a', f'img_a_{i}.jpg'))
            os.remove(os.path.join(dummy_data_dir, 'class_b', f'img_b_{i}.jpg'))
        os.rmdir(os.path.join(dummy_data_dir, 'class_a'))
        os.rmdir(os.path.join(dummy_data_dir, 'class_b'))
        os.rmdir(dummy_data_dir)

    except Exception as e:
        print(f"An error occurred during model trainer test: {e}")