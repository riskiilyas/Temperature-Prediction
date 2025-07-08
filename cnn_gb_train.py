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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Traditional ML
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class CNNGradientBoostingPredictor:
    def __init__(self, image_size=(224, 224), n_folds=5):
        self.image_size = image_size
        self.n_folds = n_folds
        self.images = []
        self.temperatures = []
        self.traditional_features = []
        
        # Models
        self.cnn_models = {}
        self.gb_models = {}
        self.ensemble_models = {}
        self.scalers = {}
        
        # Results
        self.kfold_results = {}
        self.final_results = {}
        self.training_histories = {}
        
    def load_data_for_hybrid(self, image_folder, temperature_file, augment_data=True):
        """
        Load data untuk hybrid CNN + GB model
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
        traditional_features = []
        
        # Load dan process images
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename in temp_data:
                    image_path = os.path.join(image_folder, filename)
                    try:
                        # Load CNN image
                        cnn_image = self._preprocess_image_for_cnn(image_path)
                        images.append(cnn_image)
                        
                        # Extract traditional features
                        trad_features = self._extract_traditional_features(image_path)
                        traditional_features.append(trad_features)
                        
                        temperatures.append(temp_data[filename])
                        print(f"✅ Processed: {filename} -> {temp_data[filename]}°C")
                        
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
        
        self.images = np.array(images)
        self.temperatures = np.array(temperatures)
        self.traditional_features = np.array(traditional_features)
        
        print(f"\n📈 LOADED DATA SUMMARY:")
        print(f"  Images: {len(self.images)} samples")
        print(f"  Image shape: {self.images[0].shape}")
        print(f"  Traditional features: {self.traditional_features.shape[1]} features")
        print(f"  Temperature range: {np.min(self.temperatures):.1f}°C - {np.max(self.temperatures):.1f}°C")
        
        # Data augmentation
        if augment_data and len(self.images) > 0:
            self._augment_data()
        
        return len(self.images) > 0
    
    def _preprocess_image_for_cnn(self, image_path):
        """
        Preprocess image untuk CNN input
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.image_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        return image_normalized
    
    def _extract_traditional_features(self, image_path):
        """
        Extract traditional features untuk GB (sama seperti model sebelumnya)
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        features = []
        patches = self._extract_multiple_patches(image_rgb, patch_size=64)
        
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
    
    def _extract_multiple_patches(self, image_rgb, patch_size=64):
        """
        Extract multiple patches dari posisi berbeda
        """
        height, width = image_rgb.shape[:2]
        half_patch = patch_size // 2
        
        patches = []
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
            
            patch = image_rgb[start_y:end_y, start_x:end_x]
            
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
        
        return patches
    
    def _augment_data(self):
        """
        Data augmentation untuk kedua jenis data
        """
        print("\n🔄 Augmenting data...")
        
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
        augmented_trad_features = []
        augmented_temps = []
        
        for i, (image, trad_feat, temp) in enumerate(zip(self.images, self.traditional_features, self.temperatures)):
            # Add original
            augmented_images.append(image)
            augmented_trad_features.append(trad_feat)
            augmented_temps.append(temp)
            
            # Generate augmented versions
            image_batch = np.expand_dims(image, 0)
            aug_iter = datagen.flow(image_batch, batch_size=1)
            
            for j in range(3):  # Generate 3 augmented versions
                aug_image = next(aug_iter)[0]
                augmented_images.append(aug_image)
                augmented_trad_features.append(trad_feat)  # Traditional features tetap sama
                augmented_temps.append(temp)
        
        self.images = np.array(augmented_images)
        self.traditional_features = np.array(augmented_trad_features)
        self.temperatures = np.array(augmented_temps)
        
        print(f"  Original: {original_count} samples")
        print(f"  Augmented: {len(self.images)} samples ({len(self.images)//original_count}x increase)")
    
    def create_cnn_feature_extractor(self):
        """
        Create CNN model untuk feature extraction
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
            
            # Feature extraction layer
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            
            # Dense layers for features
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(256, activation='relu', name='feature_layer'),  # Feature extraction layer
            Dropout(0.3),
            Dense(1, activation='linear')  # Final prediction
        ])
        
        return model
    
    def create_hybrid_model(self):
        """
        Create hybrid model yang menggabungkan CNN features dengan traditional features
        """
        # CNN branch untuk image features
        image_input = Input(shape=(*self.image_size, 3), name='image_input')
        x = Conv2D(64, (3, 3), activation='relu')(image_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        cnn_features = Dense(128, activation='relu', name='cnn_features')(x)
        
        # Traditional features branch
        traditional_input = Input(shape=(self.traditional_features.shape[1],), name='traditional_input')
        trad_dense = Dense(64, activation='relu')(traditional_input)
        trad_dense = Dense(32, activation='relu')(trad_dense)
        trad_features = Dense(16, activation='relu', name='trad_features')(trad_dense)
        
        # Combine both branches
        combined = Concatenate(name='combined_features')([cnn_features, trad_features])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        output = Dense(1, activation='linear', name='temperature_output')(combined)
        
        model = Model(inputs=[image_input, traditional_input], outputs=output)
        return model
    
    def train_with_kfold_cv(self):
        """
        Train models menggunakan K-Fold Cross Validation
        """
        print("\n" + "="*80)
        print("                CNN + GRADIENT BOOSTING K-FOLD TRAINING")
        print("="*80)
        
        if len(self.images) < self.n_folds:
            print(f"❌ Data terlalu sedikit untuk {self.n_folds}-fold CV")
            return None
        
        print(f"📊 Using {self.n_folds}-Fold Cross Validation")
        print(f"📊 Total samples: {len(self.images)}")
        
        # Setup K-Fold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train different model combinations
        self._train_cnn_kfold(kf)
        self._train_gb_kfold(kf)
        self._train_hybrid_kfold(kf)
        self._train_ensemble_kfold(kf)
        
        # Train final models on full dataset
        self._train_final_models()
        
        # Compare results
        self._compare_kfold_results()
        
        return self.kfold_results
    
    def _train_cnn_kfold(self, kf):
        """
        Train CNN dengan K-fold cross validation
        """
        print(f"\n🔧 Training CNN with {self.n_folds}-Fold CV...")
        
        fold_scores = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
        
        print(f"  Fold progress: ", end="")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.images)):
            print(f"{fold+1}", end="", flush=True)
            
            # Split data
            X_train, X_val = self.images[train_idx], self.images[val_idx]
            y_train, y_val = self.temperatures[train_idx], self.temperatures[val_idx]
            
            # Create and compile model
            model = self.create_cnn_feature_extractor()
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train with early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=8,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_val, verbose=0).flatten()
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            fold_scores['r2_scores'].append(r2)
            fold_scores['rmse_scores'].append(rmse)
            fold_scores['mae_scores'].append(mae)
            
            print(".", end="", flush=True)
        
        print(" ✅")
        
        # Store results
        self.kfold_results['cnn'] = {
            'cv_r2_mean': np.mean(fold_scores['r2_scores']),
            'cv_r2_std': np.std(fold_scores['r2_scores']),
            'cv_rmse_mean': np.mean(fold_scores['rmse_scores']),
            'cv_rmse_std': np.std(fold_scores['rmse_scores']),
            'cv_mae_mean': np.mean(fold_scores['mae_scores']),
            'cv_mae_std': np.std(fold_scores['mae_scores']),
            'fold_r2_scores': fold_scores['r2_scores']
        }
        
        print(f"  📊 CNN CV R² (mean±std): {self.kfold_results['cnn']['cv_r2_mean']:.4f}±{self.kfold_results['cnn']['cv_r2_std']:.4f}")
    
    def _train_gb_kfold(self, kf):
        """
        Train Gradient Boosting dengan K-fold cross validation
        """
        print(f"\n🔧 Training Gradient Boosting with {self.n_folds}-Fold CV...")
        
        fold_scores = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
        best_params_list = []
        
        print(f"  Fold progress: ", end="")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.traditional_features)):
            print(f"{fold+1}", end="", flush=True)
            
            # Split data
            X_train, X_val = self.traditional_features[train_idx], self.traditional_features[val_idx]
            y_train, y_val = self.temperatures[train_idx], self.temperatures[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            
            gb = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)
            
            # Predict and evaluate
            y_pred = best_model.predict(X_val_scaled)
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            fold_scores['r2_scores'].append(r2)
            fold_scores['rmse_scores'].append(rmse)
            fold_scores['mae_scores'].append(mae)
            
            print(".", end="", flush=True)
        
        print(" ✅")
        
        # Find most common parameters
        best_params = self._get_most_common_params(best_params_list)
        
        # Store results
        self.kfold_results['gradient_boosting'] = {
            'cv_r2_mean': np.mean(fold_scores['r2_scores']),
            'cv_r2_std': np.std(fold_scores['r2_scores']),
            'cv_rmse_mean': np.mean(fold_scores['rmse_scores']),
            'cv_rmse_std': np.std(fold_scores['rmse_scores']),
            'cv_mae_mean': np.mean(fold_scores['mae_scores']),
            'cv_mae_std': np.std(fold_scores['mae_scores']),
            'best_params': best_params,
            'fold_r2_scores': fold_scores['r2_scores']
        }
        
        print(f"  📊 GB CV R² (mean±std): {self.kfold_results['gradient_boosting']['cv_r2_mean']:.4f}±{self.kfold_results['gradient_boosting']['cv_r2_std']:.4f}")
        print(f"  🔧 Best params: {best_params}")
    
    def _train_hybrid_kfold(self, kf):
        """
        Train Hybrid CNN+Traditional model dengan K-fold
        """
        print(f"\n🔧 Training Hybrid CNN+Traditional with {self.n_folds}-Fold CV...")
        
        fold_scores = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
        
        print(f"  Fold progress: ", end="")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.images)):
            print(f"{fold+1}", end="", flush=True)
            
            # Split data
            X_img_train, X_img_val = self.images[train_idx], self.images[val_idx]
            X_trad_train, X_trad_val = self.traditional_features[train_idx], self.traditional_features[val_idx]
            y_train, y_val = self.temperatures[train_idx], self.temperatures[val_idx]
            
            # Scale traditional features
            scaler = StandardScaler()
            X_trad_train_scaled = scaler.fit_transform(X_trad_train)
            X_trad_val_scaled = scaler.transform(X_trad_val)
            
            # Create and compile hybrid model
            model = self.create_hybrid_model()
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=0)
            ]
            
            history = model.fit(
                [X_img_train, X_trad_train_scaled], y_train,
                validation_data=([X_img_val, X_trad_val_scaled], y_val),
                epochs=60,
                batch_size=8,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict and evaluate
            y_pred = model.predict([X_img_val, X_trad_val_scaled], verbose=0).flatten()
            
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            fold_scores['r2_scores'].append(r2)
            fold_scores['rmse_scores'].append(rmse)
            fold_scores['mae_scores'].append(mae)
            
            print(".", end="", flush=True)
        
        print(" ✅")
        
        # Store results
        self.kfold_results['hybrid_cnn_traditional'] = {
            'cv_r2_mean': np.mean(fold_scores['r2_scores']),
            'cv_r2_std': np.std(fold_scores['r2_scores']),
            'cv_rmse_mean': np.mean(fold_scores['rmse_scores']),
            'cv_rmse_std': np.std(fold_scores['rmse_scores']),
            'cv_mae_mean': np.mean(fold_scores['mae_scores']),
            'cv_mae_std': np.std(fold_scores['mae_scores']),
            'fold_r2_scores': fold_scores['r2_scores']
        }
        
        print(f"  📊 Hybrid CV R² (mean±std): {self.kfold_results['hybrid_cnn_traditional']['cv_r2_mean']:.4f}±{self.kfold_results['hybrid_cnn_traditional']['cv_r2_std']:.4f}")
    
    def _train_ensemble_kfold(self, kf):
        """
        Train Ensemble CNN + GB dengan K-fold
        """
        print(f"\n🔧 Training Ensemble CNN+GB with {self.n_folds}-Fold CV...")
        
        fold_scores = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
        
        print(f"  Fold progress: ", end="")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.images)):
            print(f"{fold+1}", end="", flush=True)
            
            # Split data
            X_img_train, X_img_val = self.images[train_idx], self.images[val_idx]
            X_trad_train, X_trad_val = self.traditional_features[train_idx], self.traditional_features[val_idx]
            y_train, y_val = self.temperatures[train_idx], self.temperatures[val_idx]
            
            # Train CNN
            cnn_model = self.create_cnn_feature_extractor()
            cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            cnn_callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            ]
            
            cnn_model.fit(
                X_img_train, y_train,
                validation_data=(X_img_val, y_val),
                epochs=40,
                batch_size=8,
                callbacks=cnn_callbacks,
                verbose=0
            )
            
            # Get CNN predictions
            cnn_pred_train = cnn_model.predict(X_img_train, verbose=0).flatten()
            cnn_pred_val = cnn_model.predict(X_img_val, verbose=0).flatten()
            
            # Train GB
            scaler = StandardScaler()
            X_trad_train_scaled = scaler.fit_transform(X_trad_train)
            X_trad_val_scaled = scaler.transform(X_trad_val)
            
            gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
            gb_model.fit(X_trad_train_scaled, y_train)
            
            # Get GB predictions
            gb_pred_val = gb_model.predict(X_trad_val_scaled)
            
            # Ensemble: weighted average
            ensemble_pred = 0.6 * cnn_pred_val + 0.4 * gb_pred_val
            
            # Evaluate ensemble
            r2 = r2_score(y_val, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            mae = mean_absolute_error(y_val, ensemble_pred)
            
            fold_scores['r2_scores'].append(r2)
            fold_scores['rmse_scores'].append(rmse)
            fold_scores['mae_scores'].append(mae)
            
            print(".", end="", flush=True)
        
        print(" ✅")
        
        # Store results
        self.kfold_results['ensemble_cnn_gb'] = {
            'cv_r2_mean': np.mean(fold_scores['r2_scores']),
            'cv_r2_std': np.std(fold_scores['r2_scores']),
            'cv_rmse_mean': np.mean(fold_scores['rmse_scores']),
            'cv_rmse_std': np.std(fold_scores['rmse_scores']),
            'cv_mae_mean': np.mean(fold_scores['mae_scores']),
            'cv_mae_std': np.std(fold_scores['mae_scores']),
            'fold_r2_scores': fold_scores['r2_scores']
        }
        
        print(f"  📊 Ensemble CV R² (mean±std): {self.kfold_results['ensemble_cnn_gb']['cv_r2_mean']:.4f}±{self.kfold_results['ensemble_cnn_gb']['cv_r2_std']:.4f}")
    
    def _get_most_common_params(self, params_list):
        """
        Get most common hyperparameters across folds
        """
        if not params_list:
            return {}
        
        param_counts = {}
        for params in params_list:
            param_str = str(sorted(params.items()))
            param_counts[param_str] = param_counts.get(param_str, 0) + 1
        
        most_common_str = max(param_counts.keys(), key=lambda x: param_counts[x])
        return dict(eval(most_common_str))
    
    def _train_final_models(self):
        """
        Train final models on full dataset
        """
        print(f"\n🎯 Training final models on full dataset...")
        
        # Final CNN model
        print("  🔧 Training final CNN model...")
        final_cnn = self.create_cnn_feature_extractor()
        final_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.images, self.temperatures, test_size=0.15, random_state=42
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
        ]
        
        history = final_cnn.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        self.cnn_models['final'] = final_cnn
        self.training_histories['cnn'] = history
        
        # Final GB model
        print("  🔧 Training final GB model...")
        X_trad_train, X_trad_val, y_train_gb, y_val_gb = train_test_split(
            self.traditional_features, self.temperatures, test_size=0.15, random_state=42
        )
        
        scaler = StandardScaler()
        X_trad_train_scaled = scaler.fit_transform(X_trad_train)
        X_trad_val_scaled = scaler.transform(X_trad_val)
        
        best_gb_params = self.kfold_results['gradient_boosting']['best_params']
        final_gb = GradientBoostingRegressor(random_state=42, **best_gb_params)
        final_gb.fit(X_trad_train_scaled, y_train_gb)
        
        self.gb_models['final'] = final_gb
        self.scalers['gb'] = scaler
        
        # Final Hybrid model
        print("  🔧 Training final Hybrid model...")
        final_hybrid = self.create_hybrid_model()
        final_hybrid.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        scaler_hybrid = StandardScaler()
        X_trad_train_scaled_hybrid = scaler_hybrid.fit_transform(X_trad_train)
        X_trad_val_scaled_hybrid = scaler_hybrid.transform(X_trad_val)
        
        hybrid_callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
        ]
        
        hybrid_history = final_hybrid.fit(
            [X_train, X_trad_train_scaled_hybrid], y_train,
            validation_data=([X_val, X_trad_val_scaled_hybrid], y_val),
            epochs=100,
            batch_size=8,
            callbacks=hybrid_callbacks,
            verbose=1
        )
        
        self.cnn_models['hybrid'] = final_hybrid
        self.scalers['hybrid'] = scaler_hybrid
        self.training_histories['hybrid'] = hybrid_history
        
        print("    ✅ All final models trained")
    
    def _compare_kfold_results(self):
        """
        Compare K-fold validation results
        """
        print("\n" + "="*80)
        print("                K-FOLD VALIDATION RESULTS COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.kfold_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'CV R² (mean)': f"{results['cv_r2_mean']:.4f}",
                'CV R² (std)': f"{results['cv_r2_std']:.4f}",
                'CV RMSE (mean)': f"{results['cv_rmse_mean']:.4f}°C",
                'CV RMSE (std)': f"{results['cv_rmse_std']:.4f}°C",
                'CV MAE (mean)': f"{results['cv_mae_mean']:.4f}°C",
                'CV MAE (std)': f"{results['cv_mae_std']:.4f}°C"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n📊 K-FOLD VALIDATION COMPARISON:")
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.kfold_results.keys(), 
                             key=lambda x: self.kfold_results[x]['cv_r2_mean'])
        best_score = self.kfold_results[best_model_name]['cv_r2_mean']
        best_std = self.kfold_results[best_model_name]['cv_r2_std']
        
        print(f"\n🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
        print(f"   📊 CV R² = {best_score:.4f} ± {best_std:.4f}")
        
        # Model stability analysis
        self._analyze_model_stability()
        
        # Performance comparison analysis
        self._analyze_model_performance()
    
    def _analyze_model_stability(self):
        """
        Analyze model stability across folds
        """
        print(f"\n🔍 MODEL STABILITY ANALYSIS (across {self.n_folds} folds):")
        print("-" * 60)
        
        for model_name, results in self.kfold_results.items():
            fold_scores = results['fold_r2_scores']
            stability_score = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
            
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  📊 R² range: [{min(fold_scores):.4f}, {max(fold_scores):.4f}]")
            print(f"  📊 Stability score: {stability_score:.4f}")
            
            if stability_score > 0.95:
                print("  ✅ Very stable across folds")
            elif stability_score > 0.90:
                print("  ✅ Stable across folds") 
            elif stability_score > 0.85:
                print("  ⚠️  Moderately stable")
            else:
                print("  ❌ Unstable across folds")
            print()
    
    def _analyze_model_performance(self):
        """
        Analyze relative performance between models
        """
        print(f"\n💡 MODEL PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        # Get scores
        cnn_score = self.kfold_results['cnn']['cv_r2_mean']
        gb_score = self.kfold_results['gradient_boosting']['cv_r2_mean']
        hybrid_score = self.kfold_results['hybrid_cnn_traditional']['cv_r2_mean']
        ensemble_score = self.kfold_results['ensemble_cnn_gb']['cv_r2_mean']
        
        print(f"📈 CNN vs GB improvement: {((cnn_score - gb_score) / gb_score * 100):+.1f}%")
        print(f"📈 Hybrid vs CNN improvement: {((hybrid_score - cnn_score) / cnn_score * 100):+.1f}%")
        print(f"📈 Hybrid vs GB improvement: {((hybrid_score - gb_score) / gb_score * 100):+.1f}%")
        print(f"📈 Ensemble vs best single model: {((ensemble_score - max(cnn_score, gb_score, hybrid_score)) / max(cnn_score, gb_score, hybrid_score) * 100):+.1f}%")
        
        # Best approach recommendation
        best_score = max(cnn_score, gb_score, hybrid_score, ensemble_score)
        if best_score > 0.8:
            print("\n🎉 EXCELLENT! Model performance is very good for deployment.")
        elif best_score > 0.6:
            print("\n👍 GOOD! Model performance is satisfactory.")
        elif best_score > 0.4:
            print("\n⚠️  MODERATE! Consider more data or feature engineering.")
        else:
            print("\n❌ POOR! Needs significant improvement.")
    
    def predict_new_image(self, image_path, model_type='ensemble'):
        """
        Predict temperature for new image
        """
        if model_type not in ['cnn', 'gb', 'hybrid', 'ensemble']:
            print(f"❌ Invalid model type. Choose from: cnn, gb, hybrid, ensemble")
            return None
        
        try:
            if model_type == 'cnn':
                # CNN prediction
                cnn_image = self._preprocess_image_for_cnn(image_path)
                cnn_image_batch = np.expand_dims(cnn_image, 0)
                prediction = self.cnn_models['final'].predict(cnn_image_batch, verbose=0)[0][0]
                
            elif model_type == 'gb':
                # GB prediction
                trad_features = self._extract_traditional_features(image_path)
                trad_features_scaled = self.scalers['gb'].transform(trad_features.reshape(1, -1))
                prediction = self.gb_models['final'].predict(trad_features_scaled)[0]
                
            elif model_type == 'hybrid':
                # Hybrid prediction
                cnn_image = self._preprocess_image_for_cnn(image_path)
                cnn_image_batch = np.expand_dims(cnn_image, 0)
                trad_features = self._extract_traditional_features(image_path)
                trad_features_scaled = self.scalers['hybrid'].transform(trad_features.reshape(1, -1))
                prediction = self.cnn_models['hybrid'].predict([cnn_image_batch, trad_features_scaled], verbose=0)[0][0]
                
            elif model_type == 'ensemble':
                # Ensemble prediction
                cnn_image = self._preprocess_image_for_cnn(image_path)
                cnn_image_batch = np.expand_dims(cnn_image, 0)
                cnn_pred = self.cnn_models['final'].predict(cnn_image_batch, verbose=0)[0][0]
                
                trad_features = self._extract_traditional_features(image_path)
                trad_features_scaled = self.scalers['gb'].transform(trad_features.reshape(1, -1))
                gb_pred = self.gb_models['final'].predict(trad_features_scaled)[0]
                
                # Weighted ensemble
                prediction = 0.6 * cnn_pred + 0.4 * gb_pred
            
            print(f"🌡️  Predicted temperature ({model_type}): {prediction:.2f}°C")
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            return None
    
    def save_models(self, save_dir="saved_hybrid_models"):
        """
        Save all trained models
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        saved_files = []
        
        # Save CNN models
        for model_name, model in self.cnn_models.items():
            model_path = os.path.join(save_dir, f"cnn_{model_name}_model.h5")
            model.save(model_path)
            saved_files.append(model_path)
            print(f"✅ Saved CNN {model_name} to: {model_path}")
        
        # Save GB models
        for model_name, model in self.gb_models.items():
            model_path = os.path.join(save_dir, f"gb_{model_name}_model.pkl")
            joblib.dump(model, model_path)
            saved_files.append(model_path)
            print(f"✅ Saved GB {model_name} to: {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"{scaler_name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            saved_files.append(scaler_path)
            print(f"✅ Saved {scaler_name} scaler to: {scaler_path}")
        
        # Save results and metadata
        results_info = {
            'kfold_results': self.kfold_results,
            'n_folds': self.n_folds,
            'image_size': self.image_size,
            'feature_count': self.traditional_features.shape[1] if len(self.traditional_features) > 0 else 0,
            'temperature_stats': {
                'mean': float(np.mean(self.temperatures)),
                'std': float(np.std(self.temperatures)),
                'min': float(np.min(self.temperatures)),
                'max': float(np.max(self.temperatures))
            }
        }
        
        info_path = os.path.join(save_dir, "model_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(results_info, f)
        saved_files.append(info_path)
        
        print(f"\n📦 All models saved! Files: {len(saved_files)}")
        return saved_files
    
    def visualize_results(self):
        """
        Create comprehensive visualizations
        """
        if not self.kfold_results:
            print("❌ No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CNN + Gradient Boosting K-Fold Results', fontsize=16, fontweight='bold')
        
        # 1. R² comparison
        models = list(self.kfold_results.keys())
        r2_means = [self.kfold_results[model]['cv_r2_mean'] for model in models]
        r2_stds = [self.kfold_results[model]['cv_r2_std'] for model in models]
        
        axes[0,0].bar(range(len(models)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
        axes[0,0].set_title('Cross-Validation R² Scores')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_xticks(range(len(models)))
        axes[0,0].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. RMSE comparison
        rmse_means = [self.kfold_results[model]['cv_rmse_mean'] for model in models]
        rmse_stds = [self.kfold_results[model]['cv_rmse_std'] for model in models]
        
        axes[0,1].bar(range(len(models)), rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Cross-Validation RMSE')
        axes[0,1].set_ylabel('RMSE (°C)')
        axes[0,1].set_xticks(range(len(models)))
        axes[0,1].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Box plot of R² across folds
        fold_data = [self.kfold_results[model]['fold_r2_scores'] for model in models]
        box_plot = axes[0,2].boxplot(fold_data, labels=[m.replace('_', '\n') for m in models])
        axes[0,2].set_title('R² Distribution Across Folds')
        axes[0,2].set_ylabel('R² Score')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Model stability
        stability_scores = []
        for model in models:
            results = self.kfold_results[model]
            stability = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
            stability_scores.append(stability)
        
        colors = ['green' if s > 0.9 else 'orange' if s > 0.85 else 'red' for s in stability_scores]
        axes[1,0].bar(range(len(models)), stability_scores, alpha=0.7, color=colors)
        axes[1,0].set_title('Model Stability')
        axes[1,0].set_ylabel('Stability Score')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        axes[1,0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Training history (if available)
        if self.training_histories:
            for i, (model_name, history) in enumerate(self.training_histories.items()):
                if hasattr(history, 'history'):
                    axes[1,1].plot(history.history['loss'], label=f'{model_name} Train')
                    axes[1,1].plot(history.history['val_loss'], label=f'{model_name} Val')
            axes[1,1].set_title('Training History')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Performance improvement matrix
        performance_matrix = np.zeros((len(models), len(models)))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    score1 = self.kfold_results[model1]['cv_r2_mean']
                    score2 = self.kfold_results[model2]['cv_r2_mean']
                    improvement = ((score1 - score2) / score2 * 100) if score2 > 0 else 0
                    performance_matrix[i, j] = improvement
        
        im = axes[1,2].imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        axes[1,2].set_title('Performance Improvement Matrix (%)')
        axes[1,2].set_xticks(range(len(models)))
        axes[1,2].set_yticks(range(len(models)))
        axes[1,2].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        axes[1,2].set_yticklabels([m.replace('_', '\n') for m in models])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,2])
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"cnn_gb_kfold_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_filename}")
        
        plt.show()
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive analysis report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cnn_gb_kfold_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("="*90 + "\n")
            f.write("           CNN + GRADIENT BOOSTING K-FOLD VALIDATION REPORT\n")
            f.write("="*90 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples (after augmentation): {len(self.images)}\n")
            f.write(f"Image shape: {self.images[0].shape}\n")
            f.write(f"Traditional features: {self.traditional_features.shape[1]}\n")
            f.write(f"Temperature range: {np.min(self.temperatures):.2f}°C - {np.max(self.temperatures):.2f}°C\n")
            f.write(f"Temperature mean: {np.mean(self.temperatures):.2f}°C\n")
            f.write(f"Temperature std: {np.std(self.temperatures):.2f}°C\n\n")
            
            # K-fold setup
            f.write("K-FOLD VALIDATION SETUP:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of folds: {self.n_folds}\n")
            f.write(f"Validation strategy: KFold with shuffle\n\n")
            
            # Model architectures
            f.write("MODEL ARCHITECTURES:\n")
            f.write("-" * 50 + "\n")
            f.write("1. CNN: Custom architecture with Conv2D + BatchNorm + Dropout\n")
            f.write("2. Gradient Boosting: Traditional features with hyperparameter tuning\n")
            f.write("3. Hybrid: CNN features + Traditional features combined\n")
            f.write("4. Ensemble: Weighted combination of CNN + GB predictions\n\n")
            
            # Results
            if self.kfold_results:
                f.write("K-FOLD VALIDATION RESULTS:\n")
                f.write("-" * 50 + "\n")
                for model_name, results in self.kfold_results.items():
                    f.write(f"{model_name.upper().replace('_', ' ')}:\n")
                    f.write(f"  CV R² (mean±std): {results['cv_r2_mean']:.4f}±{results['cv_r2_std']:.4f}\n")
                    f.write(f"  CV RMSE (mean±std): {results['cv_rmse_mean']:.4f}±{results['cv_rmse_std']:.4f}°C\n")
                    f.write(f"  CV MAE (mean±std): {results['cv_mae_mean']:.4f}±{results['cv_mae_std']:.4f}°C\n")
                    
                    # Stability analysis
                    stability = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
                    f.write(f"  Model stability: {stability:.4f}\n")
                    f.write(f"  Fold R² scores: {results['fold_r2_scores']}\n")
                    f.write("\n")
                
                # Best model
                best_model = max(self.kfold_results.keys(), key=lambda x: self.kfold_results[x]['cv_r2_mean'])
                f.write(f"BEST MODEL: {best_model.replace('_', ' ').title()}\n")
                f.write(f"Best CV R²: {self.kfold_results[best_model]['cv_r2_mean']:.4f}±{self.kfold_results[best_model]['cv_r2_std']:.4f}\n")
                
                # Performance analysis
                f.write("\nPERFORMANCE ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                cnn_score = self.kfold_results['cnn']['cv_r2_mean']
                gb_score = self.kfold_results['gradient_boosting']['cv_r2_mean']
                hybrid_score = self.kfold_results['hybrid_cnn_traditional']['cv_r2_mean']
                ensemble_score = self.kfold_results['ensemble_cnn_gb']['cv_r2_mean']
                
                f.write(f"CNN vs GB improvement: {((cnn_score - gb_score) / gb_score * 100):+.1f}%\n")
                f.write(f"Hybrid vs CNN improvement: {((hybrid_score - cnn_score) / cnn_score * 100):+.1f}%\n")
                f.write(f"Ensemble vs best single: {((ensemble_score - max(cnn_score, gb_score, hybrid_score)) / max(cnn_score, gb_score, hybrid_score) * 100):+.1f}%\n")
                
                # Recommendations
                best_score = max(cnn_score, gb_score, hybrid_score, ensemble_score)
                f.write("\nRECOMMENDATIONS:\n")
                f.write("-" * 50 + "\n")
                if best_score > 0.8:
                    f.write("✅ Excellent performance! Models are ready for deployment.\n")
                    f.write("✅ Consider ensemble approach for best results.\n")
                elif best_score > 0.6:
                    f.write("✅ Good performance. Fine-tuning recommended.\n")
                    f.write("⚠️  Consider collecting more diverse data.\n")
                elif best_score > 0.4:
                    f.write("⚠️  Moderate performance. Significant improvements needed.\n")
                    f.write("🔧 Try different architectures or feature engineering.\n")
                else:
                    f.write("❌ Poor performance. Reconsider approach.\n")
                    f.write("🔧 More data, different features, or alternative methods needed.\n")
        
        print(f"\n📄 Comprehensive report saved to: {report_filename}")
        return report_filename

# Main execution
if __name__ == "__main__":
    print("🌊🤖 CNN + GRADIENT BOOSTING WATER TEMPERATURE PREDICTION")
    print("=" * 80)
    
    # Initialize predictor
    predictor = CNNGradientBoostingPredictor(image_size=(224, 224), n_folds=5)
    
    # Load data
    print("\n📂 Loading data for hybrid CNN+GB training...")
    success = predictor.load_data_for_hybrid("imgs", "temperatures.txt", augment_data=True)
    
    if success:
        # Train models with K-fold CV
        print("\n🚀 Training models with K-Fold Cross Validation...")
        kfold_results = predictor.train_with_kfold_cv()
        
        if kfold_results:
            # Save all models
            print("\n💾 Saving all trained models...")
            saved_files = predictor.save_models()
            
            # Create visualizations
            print("\n📊 Creating comprehensive visualizations...")
            predictor.visualize_results()
            
            # Generate report
            report_file = predictor.generate_comprehensive_report()
            
            # Find best model
            best_model_name = max(kfold_results.keys(), 
                                 key=lambda x: kfold_results[x]['cv_r2_mean'])
            best_score = kfold_results[best_model_name]['cv_r2_mean']
            
            print(f"\n✅ CNN + GB K-fold analysis completed!")
            print(f"📄 Report: {report_file}")
            print(f"💾 Models saved in: saved_hybrid_models/")
            print(f"🏆 Best model: {best_model_name.replace('_', ' ').title()} (CV R² = {best_score:.4f})")
            
            # Test predictions with different models
            sample_images = [f for f in os.listdir("imgs") if f.endswith('.jpg')]
            if sample_images:
                print(f"\n🧪 Testing predictions with different models:")
                test_image = f"imgs/{sample_images[0]}"
                
                print("📋 Predictions for same image:")
                predictor.predict_new_image(test_image, 'cnn')
                predictor.predict_new_image(test_image, 'gb')
                predictor.predict_new_image(test_image, 'hybrid')
                predictor.predict_new_image(test_image, 'ensemble')
                
        else:
            print("❌ K-fold training failed")
    else:
        print("❌ Data loading failed")