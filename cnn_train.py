# cnn_temperature_predictor.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    Dense, Dropout, BatchNormalization, Input,
    Flatten, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Traditional ML untuk comparison
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set random seeds untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNTemperaturePredictor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.images = []
        self.temperatures = []
        self.cnn_model = None
        self.hybrid_model = None
        self.scaler = StandardScaler()
        self.history = {}
        self.results = {}
        
    def load_data_for_cnn(self, image_folder, temperature_file, augment_data=True):
        """
        Load dan preprocess data untuk CNN
        """
        # Baca temperature data
        temp_data = {}
        with open(temperature_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    full_path = parts[0].strip()
                    temp = float(parts[1].strip())
                    filename = os.path.basename(full_path)
                    temp_data[filename] = temp
        
        print(f"📊 Temperature data loaded: {len(temp_data)} entries")
        
        images = []
        temperatures = []
        
        # Load original images
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename in temp_data:
                    image_path = os.path.join(image_folder, filename)
                    try:
                        # Load dan preprocess image
                        image = self._preprocess_image(image_path)
                        images.append(image)
                        temperatures.append(temp_data[filename])
                        print(f"✅ Loaded: {filename} -> {temp_data[filename]}°C")
                        
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
        
        self.images = np.array(images)
        self.temperatures = np.array(temperatures)
        
        print(f"\n📈 LOADED DATA SUMMARY:")
        print(f"  Images: {len(self.images)} samples")
        print(f"  Image shape: {self.images[0].shape}")
        print(f"  Temperature range: {np.min(self.temperatures):.1f}°C - {np.max(self.temperatures):.1f}°C")
        
        # Data augmentation menggunakan Keras ImageDataGenerator
        if augment_data:
            self._augment_data()
        
        return len(self.images) > 0
    
    def _preprocess_image(self, image_path):
        """
        Preprocess image untuk CNN input
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image_resized = cv2.resize(image_rgb, self.image_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized
    
    def _augment_data(self):
        """
        Augment data menggunakan Keras ImageDataGenerator
        """
        print("\n🔄 Augmenting data...")
        
        # Create data generator
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        original_count = len(self.images)
        augmented_images = []
        augmented_temps = []
        
        # Generate augmented images
        for i, (image, temp) in enumerate(zip(self.images, self.temperatures)):
            # Add original
            augmented_images.append(image)
            augmented_temps.append(temp)
            
            # Generate 5 augmented versions
            image_batch = np.expand_dims(image, 0)
            aug_iter = datagen.flow(image_batch, batch_size=1)
            
            for j in range(5):
                aug_image = next(aug_iter)[0]
                augmented_images.append(aug_image)
                augmented_temps.append(temp)
        
        self.images = np.array(augmented_images)
        self.temperatures = np.array(augmented_temps)
        
        print(f"  Original: {original_count} images")
        print(f"  Augmented: {len(self.images)} images ({len(self.images)//original_count}x increase)")
    
    def create_cnn_model(self, model_type='custom'):
        """
        Create CNN model untuk temperature prediction
        """
        if model_type == 'custom':
            return self._create_custom_cnn()
        elif model_type == 'transfer':
            return self._create_transfer_learning_model()
        elif model_type == 'hybrid':
            return self._create_hybrid_model()
    
    def _create_custom_cnn(self):
        """
        Custom CNN architecture untuk temperature prediction
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        return model
    
    def _create_transfer_learning_model(self):
        """
        Transfer learning menggunakan pre-trained model
        """
        # Load pre-trained model (tanpa top layer)
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _create_hybrid_model(self):
        """
        Hybrid model: CNN features + traditional features
        """
        # CNN branch
        image_input = Input(shape=(*self.image_size, 3))
        x = Conv2D(64, (3, 3), activation='relu')(image_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        cnn_features = Dense(128, activation='relu')(x)
        
        # Traditional features branch (RGB statistics, etc.)
        traditional_input = Input(shape=(85,))  # Same as previous model
        trad_features = Dense(64, activation='relu')(traditional_input)
        trad_features = Dense(32, activation='relu')(trad_features)
        
        # Combine branches
        combined = Concatenate()([cnn_features, trad_features])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(64, activation='relu')(combined)
        output = Dense(1, activation='linear')(combined)
        
        model = Model(inputs=[image_input, traditional_input], outputs=output)
        return model
    
    def train_cnn_model(self, model_type='custom', epochs=100, batch_size=16):
        """
        Train CNN model
        """
        print(f"\n🚀 Training {model_type} CNN model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.temperatures, test_size=0.2, random_state=42
        )
        
        print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Create model
        model = self.create_cnn_model(model_type)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"🏗️  Model architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_{model_type}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and results
        self.cnn_model = model
        self.history[model_type] = history
        
        # Evaluate
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.results[f'cnn_{model_type}'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'model': model,
            'history': history
        }
        
        print(f"\n📊 {model_type.upper()} CNN RESULTS:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}°C")
        print(f"  Test MAE:  {test_mae:.4f}°C")
        
        return model, history
    
    def extract_traditional_features(self, image):
        """
        Extract traditional features untuk hybrid model
        """
        # Resize untuk consistency dengan previous model
        image_resized = cv2.resize(image, (64, 64))
        features = []
        
        # Multiple patches
        patches = self._extract_multiple_patches(image_resized)
        
        for patch in patches:
            # RGB statistics
            rgb_mean = np.mean(patch, axis=(0, 1))
            rgb_std = np.std(patch, axis=(0, 1))
            rgb_max = np.max(patch, axis=(0, 1))
            rgb_min = np.min(patch, axis=(0, 1))
            
            features.extend(rgb_mean)
            features.extend(rgb_std)
            features.extend(rgb_max)
            features.extend(rgb_min)
            
            # HSV features
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            hsv_mean = np.mean(patch_hsv, axis=(0, 1))
            features.extend(hsv_mean)
            
            # Brightness and contrast
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray_patch)
            contrast = np.std(gray_patch)
            features.extend([brightness, contrast])
        
        return np.array(features)
    
    def _extract_multiple_patches(self, image):
        """
        Extract multiple patches (same as previous model)
        """
        height, width = image.shape[:2]
        patch_size = 16  # Smaller for 64x64 image
        half_patch = patch_size // 2
        
        patches = []
        
        # 5 locations
        locations = [
            (width // 2, height // 2),      # Center
            (width // 2, height // 4),      # Top
            (width // 2, 3 * height // 4),  # Bottom
            (width // 4, height // 2),      # Left
            (3 * width // 4, height // 2)   # Right
        ]
        
        for x, y in locations:
            start_y = max(0, y - half_patch)
            end_y = min(height, y + half_patch)
            start_x = max(0, x - half_patch)
            end_x = min(width, x + half_patch)
            
            patch = image[start_y:end_y, start_x:end_x]
            
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
        
        return patches
    
    def compare_with_traditional_ml(self):
        """
        Compare CNN dengan traditional ML
        """
        print("\n🔄 Comparing with traditional ML...")
        
        # Extract traditional features from CNN images
        traditional_features = []
        for img in self.images:
            # Convert normalized image back to 0-255 range
            img_255 = (img * 255).astype(np.uint8)
            features = self.extract_traditional_features(img_255)
            traditional_features.append(features)
        
        traditional_features = np.array(traditional_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            traditional_features, self.temperatures, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = rf_model.predict(X_train_scaled)
        test_pred = rf_model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        self.results['traditional_rf'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'model': rf_model
        }
        
        print(f"📊 TRADITIONAL RF RESULTS:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}°C")
        print(f"  Test MAE:  {test_mae:.4f}°C")
    
    def plot_training_history(self):
        """
        Plot training history untuk CNN models
        """
        if not self.history:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, len(self.history), figsize=(15, 8))
        if len(self.history) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, history) in enumerate(self.history.items()):
            # Loss plot
            axes[0, i].plot(history.history['loss'], label='Train Loss')
            axes[0, i].plot(history.history['val_loss'], label='Val Loss')
            axes[0, i].set_title(f'{model_name.upper()} - Loss')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('MSE Loss')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # MAE plot
            axes[1, i].plot(history.history['mae'], label='Train MAE')
            axes[1, i].plot(history.history['val_mae'], label='Val MAE')
            axes[1, i].set_title(f'{model_name.upper()} - MAE')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('MAE')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_all_models(self):
        """
        Compare semua model yang sudah ditraining
        """
        if not self.results:
            print("❌ No models trained yet")
            return
        
        print("\n" + "="*80)
        print("                    COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # Buat comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train R²': f"{results['train_r2']:.4f}",
                'Test R²': f"{results['test_r2']:.4f}",
                'Test RMSE': f"{results['test_rmse']:.4f}°C",
                'Test MAE': f"{results['test_mae']:.4f}°C"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_r2'])
        best_score = self.results[best_model_name]['test_r2']
        
        print(f"\n🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()} (R² = {best_score:.4f})")
        
        # Performance improvement analysis
        if 'traditional_rf' in self.results:
            traditional_score = self.results['traditional_rf']['test_r2']
            improvement = ((best_score - traditional_score) / traditional_score) * 100
            print(f"📈 CNN improvement over traditional: {improvement:+.1f}%")
    
    def predict_new_image(self, image_path, model_name='cnn_custom'):
        """
        Prediksi suhu untuk gambar baru menggunakan CNN
        """
        if model_name not in self.results:
            print(f"❌ Model {model_name} not available")
            print(f"Available models: {list(self.results.keys())}")
            return None
        
        try:
            # Preprocess image
            image = self._preprocess_image(image_path)
            image_batch = np.expand_dims(image, 0)
            
            # Predict
            model = self.results[model_name]['model']
            prediction = model.predict(image_batch, verbose=0)[0][0]
            
            print(f"🌡️  Predicted temperature: {prediction:.2f}°C")
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            return None
    
    def save_models(self, save_dir="saved_cnn_models"):
        """
        Save trained CNN models
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        saved_files = []
        
        for model_name, results in self.results.items():
            if 'model' in results:
                model_path = os.path.join(save_dir, f"{model_name}_model.h5")
                if hasattr(results['model'], 'save'):
                    # Keras model
                    results['model'].save(model_path)
                else:
                    # Sklearn model
                    import joblib
                    model_path = model_path.replace('.h5', '.pkl')
                    joblib.dump(results['model'], model_path)
                
                saved_files.append(model_path)
                print(f"✅ Saved {model_name} to: {model_path}")
        
        print(f"\n💾 Saved {len(saved_files)} models to {save_dir}/")
        return saved_files
    
    def generate_report(self):
        """
        Generate comprehensive CNN analysis report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cnn_temperature_analysis_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("              CNN WATER TEMPERATURE PREDICTION ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples: {len(self.images)}\n")
            f.write(f"Image shape: {self.images[0].shape}\n")
            f.write(f"Temperature range: {np.min(self.temperatures):.2f}°C - {np.max(self.temperatures):.2f}°C\n")
            f.write(f"Temperature mean: {np.mean(self.temperatures):.2f}°C\n")
            f.write(f"Temperature std: {np.std(self.temperatures):.2f}°C\n\n")
            
            # Model results
            if self.results:
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 50 + "\n")
                for model_name, results in self.results.items():
                    f.write(f"{model_name.upper()}:\n")
                    f.write(f"  Train R²: {results['train_r2']:.4f}\n")
                    f.write(f"  Test R²: {results['test_r2']:.4f}\n")
                    f.write(f"  Test RMSE: {results['test_rmse']:.4f}°C\n")
                    f.write(f"  Test MAE: {results['test_mae']:.4f}°C\n\n")
                
                # Best model
                best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
                f.write(f"BEST MODEL: {best_model}\n")
                f.write(f"Best Test R²: {self.results[best_model]['test_r2']:.4f}\n")
        
        print(f"\n📄 Report saved to: {report_filename}")
        return report_filename

# Main execution
if __name__ == "__main__":
    print("🤖 CNN WATER TEMPERATURE PREDICTION")
    print("=" * 50)
    
    # Initialize predictor
    predictor = CNNTemperaturePredictor(image_size=(224, 224))
    
    # Load data
    print("\n📂 Loading data for CNN...")
    success = predictor.load_data_for_cnn("imgs", "temperatures.txt", augment_data=True)
    
    if success:
        # Train different CNN models
        print("\n🚀 Training CNN models...")
        
        # 1. Custom CNN
        predictor.train_cnn_model('custom', epochs=50, batch_size=8)
        
        # 2. Transfer Learning
        try:
            predictor.train_cnn_model('transfer', epochs=30, batch_size=8)
        except Exception as e:
            print(f"⚠️ Transfer learning failed: {e}")
        
        # 3. Compare with traditional ML
        predictor.compare_with_traditional_ml()
        
        # 4. Plot training history
        predictor.plot_training_history()
        
        # 5. Compare all models
        predictor.compare_all_models()
        
        # 6. Save models
        predictor.save_models()
        
        # 7. Generate report
        report_file = predictor.generate_report()
        
        print(f"\n✅ CNN analysis completed!")
        print(f"📄 Check '{report_file}' for detailed results.")
        
        # Test prediction
        sample_images = [f for f in os.listdir("imgs") if f.endswith('.jpg')]
        if sample_images:
            print(f"\n🧪 Testing CNN prediction:")
            pred_temp = predictor.predict_new_image(f"imgs/{sample_images[0]}")
            
    else:
        print("❌ Failed to load data")