# train_model.py (Final Version - English Comments)

import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB1

# =================================================================================================
# 1. Configuration
# =================================================================================================
class Config:
    """Holds all configuration parameters for training."""
    BASE_DIR = './Dataset' # Assumes 'Dataset' folder is in the same directory
    WORKING_DIR = r'./'

    # Image and Model Parameters
    IMG_HEIGHT, IMG_WIDTH = 256, 256 # Reduced size to prevent memory errors
    IMG_SHAPE, IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, 3), (IMG_HEIGHT, IMG_WIDTH)
    SAVE_MODEL_NAME = 'shoulder_xray_model.h5'

    # Training Hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 4 # Reduced batch size to prevent memory errors
    LEARNING_RATE = 0.001

# =================================================================================================
# 2. Data Preparation Functions
# =================================================================================================
def create_dataframes(base_dir):
    """Scans train, validation, and test directories to create pandas DataFrames."""
    print("Creating DataFrames...")
    data = {}
    for category in ['train', 'val', 'test']:
        category_path = os.path.join(base_dir, category)
        filepaths, labels = [], []
        for klass in os.listdir(category_path):
            classpath = os.path.join(category_path, klass)
            for f in os.listdir(classpath):
                filepaths.append(os.path.join(classpath, f))
                labels.append(klass)
        data[f"{category}_df"] = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    print(f"DataFrames created: train({len(data['train_df'])}), valid({len(data['val_df'])}), test({len(data['test_df'])})")
    return data['train_df'], data['val_df'], data['test_df']

def balance_and_augment_data(train_df, working_dir, img_size):
    """Balances the training data by downsampling majority classes and augmenting minority classes."""
    print("\nBalancing and augmenting data...")
    # --- Downsampling ---
    max_size = train_df['labels'].value_counts().iloc[1] # Use second largest class size as target
    groups = train_df.groupby('labels')
    sample_list = [groups.get_group(label).sample(min(len(groups.get_group(label)), max_size), random_state=123) for label in train_df['labels'].unique()]
    balanced_df = pd.concat(sample_list, axis=0).reset_index(drop=True)
    
    # --- Augmentation ---
    aug_dir = os.path.join(working_dir, 'aug')
    shutil.rmtree(aug_dir, ignore_errors=True) # Clean up old augmentation folder
    os.makedirs(aug_dir)
    
    target_count = balanced_df['labels'].value_counts().max()
    gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)
    
    for label in balanced_df['labels'].unique():
        group = balanced_df[balanced_df['labels'] == label]
        if len(group) < target_count:
            delta = target_count - len(group)
            target_dir = os.path.join(aug_dir, label)
            os.makedirs(target_dir, exist_ok=True)
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size, class_mode=None, batch_size=1, shuffle=False, save_to_dir=target_dir, save_prefix='aug-', save_format='jpg')
            for i in range(delta):
                next(aug_gen)
            
    # --- Combine original and augmented data ---
    aug_paths, aug_labels = [], []
    for klass in os.listdir(aug_dir):
        for f in os.listdir(os.path.join(aug_dir, klass)):
            aug_paths.append(os.path.join(aug_dir, klass, f))
            aug_labels.append(klass)
    if aug_paths:
        final_train_df = pd.concat([balanced_df, pd.DataFrame({'filepaths': aug_paths, 'labels': aug_labels})], axis=0)
    else:
        final_train_df = balanced_df
        
    return final_train_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

def create_data_generators(train_df, valid_df, img_size, batch_size):
    """Creates ImageDataGenerators for training and validation."""
    print("\nSetting up data generators...")
    # No rescaling, to match the original notebook's logic
    trgen = ImageDataGenerator(horizontal_flip=True)
    vgen = ImageDataGenerator()
    
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = vgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    return train_gen, valid_gen

# =================================================================================================
# 3. Main Training Execution
# =================================================================================================
if __name__ == "__main__":
    train_df, valid_df, test_df = create_dataframes(Config.BASE_DIR)
    
    if train_df is not None:
        train_df = balance_and_augment_data(train_df, Config.WORKING_DIR, Config.IMG_SIZE)
        train_gen, valid_gen = create_data_generators(train_df, valid_df, Config.IMG_SIZE, Config.BATCH_SIZE)
        class_count = len(train_gen.class_indices)

        # --- Build the Model ---
        print("\nBuilding model...")
        base_model = EfficientNetB1(include_top=False, weights="imagenet", input_shape=Config.IMG_SHAPE, pooling='max')
        x = base_model.output
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.45)(x)
        output = Dense(class_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        
        model.compile(Adamax(learning_rate=Config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model built and compiled.")

        # --- Train the Model ---
        print(f"\nStarting training for {Config.EPOCHS} epochs...")
        model.fit(train_gen, epochs=Config.EPOCHS, validation_data=valid_gen, shuffle=False)
        
        # --- Save the Final Model ---
        model.save(os.path.join(Config.WORKING_DIR, Config.SAVE_MODEL_NAME))
        print(f"\nTRAINING COMPLETE! Model saved successfully at: {Config.SAVE_MODEL_NAME}")