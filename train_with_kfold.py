import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

class WaterTemperaturePredictorWithKFold:
    def __init__(self, n_folds=5):
        self.rgb_features = []
        self.temperatures = []
        self.feature_names = []
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.n_folds = n_folds
        self.kfold_results = {}

    def save_models(self, save_dir="saved_models"):
        """
        Save trained models and scalers
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        saved_files = []
        
        for model_name, model in self.models.items():
            # Save model
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            saved_files.append(model_path)
            print(f"✅ Saved {model_name} model to: {model_path}")
            
            # Save scaler if exists
            if model_name in self.scalers:
                scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
                joblib.dump(self.scalers[model_name], scaler_path)
                saved_files.append(scaler_path)
                print(f"✅ Saved {model_name} scaler to: {scaler_path}")
        
        # Save feature extraction parameters and K-fold results
        feature_info = {
            'feature_count': self.rgb_features.shape[1] if len(self.rgb_features) > 0 else 0,
            'temperature_stats': {
                'mean': float(np.mean(self.temperatures)),
                'std': float(np.std(self.temperatures)),
                'min': float(np.min(self.temperatures)),
                'max': float(np.max(self.temperatures))
            },
            'kfold_results': self.kfold_results,
            'n_folds': self.n_folds
        }
        
        info_path = os.path.join(save_dir, "model_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(feature_info, f)
        saved_files.append(info_path)
        
        print(f"\n📦 All models saved! Files: {len(saved_files)}")
        return saved_files

    def load_model(self, model_name, model_dir="saved_models"):
        """
        Load a specific trained model
        """
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Load model
        self.models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded {model_name} model")
        
        # Load scaler if exists
        if os.path.exists(scaler_path):
            self.scalers[model_name] = joblib.load(scaler_path)
            print(f"✅ Loaded {model_name} scaler")
        
        return True

    def get_best_model(self):
        """
        Get the best performing model based on K-fold validation
        """
        if not self.kfold_results:
            print("❌ No K-fold results available")
            return None
        
        # Find best model based on mean CV score
        best_model_name = max(self.kfold_results.keys(), 
                            key=lambda x: self.kfold_results[x]['cv_r2_mean'])
        best_model = self.models[best_model_name]
        best_scaler = self.scalers.get(best_model_name, None)
        
        print(f"🏆 Best model: {best_model_name} (CV R² = {self.kfold_results[best_model_name]['cv_r2_mean']:.4f})")
        
        return {
            'name': best_model_name,
            'model': best_model,
            'scaler': best_scaler,
            'kfold_performance': self.kfold_results[best_model_name]
        }

    def extract_enhanced_features(self, image_path, patch_size=64):
        """
        Ekstrak fitur yang lebih kaya dari gambar
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        features = []
        
        # 1. Multiple patches (center, top, bottom, left, right)
        patches = self._extract_multiple_patches(image_rgb, patch_size)
        
        for i, patch in enumerate(patches):
            # RGB statistics untuk setiap patch
            rgb_mean = np.mean(patch, axis=(0, 1))
            rgb_std = np.std(patch, axis=(0, 1))
            rgb_max = np.max(patch, axis=(0, 1))
            rgb_min = np.min(patch, axis=(0, 1))
            
            features.extend(rgb_mean)      # 3 features
            features.extend(rgb_std)       # 3 features  
            features.extend(rgb_max)       # 3 features
            features.extend(rgb_min)       # 3 features
            
            # 2. HSV features
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            hsv_mean = np.mean(patch_hsv, axis=(0, 1))
            features.extend(hsv_mean)      # 3 features
            
            # 3. Brightness and contrast
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray_patch)
            contrast = np.std(gray_patch)
            features.extend([brightness, contrast])  # 2 features
        
        # Total: 5 patches × (3+3+3+3+3+2) = 5 × 17 = 85 features
        
        return np.array(features)
    
    def _extract_multiple_patches(self, image_rgb, patch_size):
        """
        Ekstrak multiple patches dari posisi berbeda
        """
        height, width = image_rgb.shape[:2]
        half_patch = patch_size // 2
        
        patches = []
        
        # Center patch
        center_x, center_y = width // 2, height // 2
        patches.append(self._safe_extract_patch(image_rgb, center_x, center_y, half_patch))
        
        # Top patch
        top_x, top_y = width // 2, height // 4
        patches.append(self._safe_extract_patch(image_rgb, top_x, top_y, half_patch))
        
        # Bottom patch  
        bottom_x, bottom_y = width // 2, 3 * height // 4
        patches.append(self._safe_extract_patch(image_rgb, bottom_x, bottom_y, half_patch))
        
        # Left patch
        left_x, left_y = width // 4, height // 2
        patches.append(self._safe_extract_patch(image_rgb, left_x, left_y, half_patch))
        
        # Right patch
        right_x, right_y = 3 * width // 4, height // 2
        patches.append(self._safe_extract_patch(image_rgb, right_x, right_y, half_patch))
        
        return patches
    
    def _safe_extract_patch(self, image_rgb, center_x, center_y, half_patch):
        """
        Safely extract patch with boundary checking
        """
        height, width = image_rgb.shape[:2]
        
        start_y = max(0, center_y - half_patch)
        end_y = min(height, center_y + half_patch)
        start_x = max(0, center_x - half_patch)
        end_x = min(width, center_x + half_patch)
        
        patch = image_rgb[start_y:end_y, start_x:end_x]
        
        # Resize to fixed size if needed
        if patch.shape[0] < half_patch * 2 or patch.shape[1] < half_patch * 2:
            patch = cv2.resize(patch, (half_patch * 2, half_patch * 2))
            
        return patch
    
    def augment_image(self, image_path):
        """
        Augment image dengan rotasi dan flip
        Returns: list of augmented images (as numpy arrays)
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented_images = [image_rgb]  # Original
        
        # 3 rotations: 90°, 180°, 270°
        for angle in [90, 180, 270]:
            if angle == 90:
                rotated = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image_rgb, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            augmented_images.append(rotated)
        
        # 2 flips: horizontal dan vertical
        flip_h = cv2.flip(image_rgb, 1)  # Horizontal flip
        flip_v = cv2.flip(image_rgb, 0)  # Vertical flip
        augmented_images.extend([flip_h, flip_v])
        
        return augmented_images  # Total: 6 images (1 + 3 + 2)
    
    def load_data_with_augmentation(self, image_folder, temperature_file):
        """
        Load data dengan augmentation
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
        
        original_count = 0
        augmented_count = 0
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename in temp_data:
                    image_path = os.path.join(image_folder, filename)
                    temp = temp_data[filename]
                    
                    try:
                        # Augment images
                        augmented_images = self.augment_image(image_path)
                        
                        for i, aug_image in enumerate(augmented_images):
                            # Save augmented image temporarily untuk extract features
                            temp_path = f"temp_aug_{i}.jpg"
                            cv2.imwrite(temp_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                            
                            # Extract features
                            features = self.extract_enhanced_features(temp_path)
                            self.rgb_features.append(features)
                            self.temperatures.append(temp)
                            
                            if i == 0:
                                original_count += 1
                            else:
                                augmented_count += 1
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                        print(f"✅ {filename}: {len(augmented_images)} variants processed -> {temp}°C")
                        
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
        
        self.rgb_features = np.array(self.rgb_features)
        self.temperatures = np.array(self.temperatures)
        
        print(f"\n📈 DATA AUGMENTATION SUMMARY:")
        print(f"  Original images: {original_count}")
        print(f"  Augmented images: {augmented_count}")
        print(f"  Total samples: {len(self.rgb_features)} (6x increase)")
        print(f"  Features per sample: {self.rgb_features.shape[1]}")
        
        return len(self.rgb_features) > 0
    
    def train_with_kfold_cv(self):
        """
        Train models menggunakan K-Fold Cross Validation
        """
        print("\n" + "="*80)
        print("                K-FOLD CROSS VALIDATION TRAINING")
        print("="*80)
        
        if len(self.rgb_features) < self.n_folds:
            print(f"❌ Data terlalu sedikit untuk {self.n_folds}-fold CV")
            return None
        
        X = self.rgb_features
        y = self.temperatures
        
        print(f"📊 Using {self.n_folds}-Fold Cross Validation")
        print(f"📊 Total samples: {len(X)}")
        print(f"📊 Features per sample: {X.shape[1]}")
        
        # Setup K-Fold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Models to train
        models_config = {
            'ridge': {
                'model': Ridge(),
                'param_grid': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
                'needs_scaling': True
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'needs_scaling': False
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'needs_scaling': False
            }
        }
        
        # Train each model with K-Fold CV
        for model_name, config in models_config.items():
            print(f"\n🔧 Training {model_name.replace('_', ' ').title()} with {self.n_folds}-Fold CV...")
            self._train_model_kfold(model_name, config, X, y, kf)
        
        # Final model training on full dataset
        self._train_final_models(X, y)
        
        # Compare results
        self._compare_kfold_results()
        
        return self.kfold_results
    
    def _train_model_kfold(self, model_name, config, X, y, kf):
        """
        Train a specific model using K-Fold Cross Validation
        """
        model = config['model']
        param_grid = config['param_grid']
        needs_scaling = config['needs_scaling']
        
        fold_scores = {
            'r2_scores': [],
            'rmse_scores': [],
            'mae_scores': []
        }
        
        best_params_list = []
        
        print(f"  Fold progress: ", end="")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"{fold+1}", end="", flush=True)
            
            # Split data for this fold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale if needed
            if needs_scaling:
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)
            
            # Hyperparameter tuning using inner CV
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv, 
                scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train_fold, y_train_fold)
            
            # Get best model for this fold
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)
            
            # Predict on validation set
            y_pred = best_model.predict(X_val_fold)
            
            # Calculate metrics
            r2 = r2_score(y_val_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            fold_scores['r2_scores'].append(r2)
            fold_scores['rmse_scores'].append(rmse)
            fold_scores['mae_scores'].append(mae)
            
            print(".", end="", flush=True)
        
        print(" ✅")
        
        # Calculate statistics across folds
        cv_r2_mean = np.mean(fold_scores['r2_scores'])
        cv_r2_std = np.std(fold_scores['r2_scores'])
        cv_rmse_mean = np.mean(fold_scores['rmse_scores'])
        cv_rmse_std = np.std(fold_scores['rmse_scores'])
        cv_mae_mean = np.mean(fold_scores['mae_scores'])
        cv_mae_std = np.std(fold_scores['mae_scores'])
        
        # Find most common hyperparameters
        best_params = self._get_most_common_params(best_params_list)
        
        # Store results
        self.kfold_results[model_name] = {
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std,
            'cv_mae_mean': cv_mae_mean,
            'cv_mae_std': cv_mae_std,
            'best_params': best_params,
            'needs_scaling': needs_scaling,
            'fold_r2_scores': fold_scores['r2_scores']
        }
        
        print(f"  📊 CV R² (mean±std): {cv_r2_mean:.4f}±{cv_r2_std:.4f}")
        print(f"  📊 CV RMSE (mean±std): {cv_rmse_mean:.4f}±{cv_rmse_std:.4f}°C")
        print(f"  📊 CV MAE (mean±std): {cv_mae_mean:.4f}±{cv_mae_std:.4f}°C")
        print(f"  🔧 Best params: {best_params}")
    
    def _get_most_common_params(self, params_list):
        """
        Get the most commonly selected hyperparameters across folds
        """
        if not params_list:
            return {}
        
        # Count frequency of each parameter combination
        param_counts = {}
        for params in params_list:
            param_str = str(sorted(params.items()))
            param_counts[param_str] = param_counts.get(param_str, 0) + 1
        
        # Get most frequent
        most_common_str = max(param_counts.keys(), key=lambda x: param_counts[x])
        
        # Convert back to dict
        most_common_params = dict(eval(most_common_str))
        
        return most_common_params
    
    def _train_final_models(self, X, y):
        """
        Train final models on full dataset using best hyperparameters from K-fold
        """
        print(f"\n🎯 Training final models on full dataset...")
        
        for model_name, kfold_result in self.kfold_results.items():
            print(f"  🔧 Training final {model_name} model...")
            
            best_params = kfold_result['best_params']
            needs_scaling = kfold_result['needs_scaling']
            
            # Prepare data
            X_final = X.copy()
            if needs_scaling:
                scaler = StandardScaler()
                X_final = scaler.fit_transform(X_final)
                self.scalers[model_name] = scaler
            
            # Create and train final model
            if model_name == 'ridge':
                final_model = Ridge(**best_params)
            elif model_name == 'random_forest':
                final_model = RandomForestRegressor(random_state=42, **best_params)
            elif model_name == 'gradient_boosting':
                final_model = GradientBoostingRegressor(random_state=42, **best_params)
            
            final_model.fit(X_final, y)
            self.models[model_name] = final_model
            
            print(f"    ✅ {model_name} final model trained")
    
    def _compare_kfold_results(self):
        """
        Compare K-Fold validation results across models
        """
        print("\n" + "="*80)
        print("                K-FOLD CROSS VALIDATION RESULTS")
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
    
    def _analyze_model_stability(self):
        """
        Analyze model stability across K-fold iterations
        """
        print(f"\n🔍 MODEL STABILITY ANALYSIS (across {self.n_folds} folds):")
        print("-" * 60)
        
        for model_name, results in self.kfold_results.items():
            fold_scores = results['fold_r2_scores']
            stability_score = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
            
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  📊 R² range: [{min(fold_scores):.4f}, {max(fold_scores):.4f}]")
            print(f"  📊 Stability score: {stability_score:.4f} (higher = more stable)")
            
            if stability_score > 0.95:
                print("  ✅ Very stable across folds")
            elif stability_score > 0.90:
                print("  ✅ Stable across folds") 
            elif stability_score > 0.85:
                print("  ⚠️  Moderately stable")
            else:
                print("  ❌ Unstable across folds")
            print()
    
    def predict_new_image(self, image_path, model_name='ridge'):
        """
        Prediksi suhu untuk gambar baru menggunakan model terbaik
        """
        if model_name not in self.models:
            print(f"❌ Model {model_name} belum ditraining")
            return None
        
        try:
            # Extract features
            features = self.extract_enhanced_features(image_path)
            features = features.reshape(1, -1)
            
            # Scale features if needed
            if model_name in self.scalers:
                features = self.scalers[model_name].transform(features)
            
            # Predict
            prediction = self.models[model_name].predict(features)[0]
            
            print(f"🌡️  Predicted temperature: {prediction:.2f}°C")
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            return None
    
    def generate_kfold_report(self):
        """
        Generate comprehensive K-fold validation report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"kfold_validation_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*90 + "\n")
            f.write("           WATER TEMPERATURE PREDICTION - K-FOLD VALIDATION REPORT\n")
            f.write("="*90 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples (after augmentation): {len(self.rgb_features)}\n")
            f.write(f"Feature dimensions: {self.rgb_features.shape[1]}\n")
            f.write(f"Temperature range: {np.min(self.temperatures):.2f}°C - {np.max(self.temperatures):.2f}°C\n")
            f.write(f"Temperature mean: {np.mean(self.temperatures):.2f}°C\n")
            f.write(f"Temperature std: {np.std(self.temperatures):.2f}°C\n\n")
            
            # K-fold setup
            f.write("K-FOLD VALIDATION SETUP:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of folds: {self.n_folds}\n")
            f.write(f"Validation strategy: KFold with shuffle\n")
            f.write(f"Random state: 42\n\n")
            
            # Model results
            if self.kfold_results:
                f.write("K-FOLD VALIDATION RESULTS:\n")
                f.write("-" * 50 + "\n")
                for model_name, results in self.kfold_results.items():
                    f.write(f"{model_name.upper().replace('_', ' ')}:\n")
                    f.write(f"  CV R² (mean±std): {results['cv_r2_mean']:.4f}±{results['cv_r2_std']:.4f}\n")
                    f.write(f"  CV RMSE (mean±std): {results['cv_rmse_mean']:.4f}±{results['cv_rmse_std']:.4f}°C\n")
                    f.write(f"  CV MAE (mean±std): {results['cv_mae_mean']:.4f}±{results['cv_mae_std']:.4f}°C\n")
                    f.write(f"  Best hyperparameters: {results['best_params']}\n")
                    f.write(f"  Fold R² scores: {results['fold_r2_scores']}\n")
                    
                    # Calculate stability
                    stability_score = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
                    f.write(f"  Model stability: {stability_score:.4f}\n")
                    f.write("\n")
                
                # Best model
                best_model = max(self.kfold_results.keys(), key=lambda x: self.kfold_results[x]['cv_r2_mean'])
                f.write(f"BEST MODEL: {best_model.replace('_', ' ').title()}\n")
                f.write(f"Best CV R²: {self.kfold_results[best_model]['cv_r2_mean']:.4f}±{self.kfold_results[best_model]['cv_r2_std']:.4f}\n")
                f.write(f"Best CV RMSE: {self.kfold_results[best_model]['cv_rmse_mean']:.4f}°C\n")
                
                # Recommendations
                f.write("\nRECOMMENDATIONS:\n")
                f.write("-" * 50 + "\n")
                if self.kfold_results[best_model]['cv_r2_mean'] > 0.8:
                    f.write("Excellent model performance! Ready for deployment.\n")
                elif self.kfold_results[best_model]['cv_r2_mean'] > 0.6:
                    f.write("Good model performance. Consider fine-tuning for better results.\n")
                elif self.kfold_results[best_model]['cv_r2_mean'] > 0.4:
                    f.write("Moderate performance. Consider more feature engineering.\n")
                else:
                    f.write("Poor performance. Consider different approach or more data.\n")
        
        print(f"\n📄 K-fold validation report saved to: {report_filename}")
        return report_filename

    def visualize_kfold_results(self):
        """
        Create visualizations for K-fold validation results
        """
        if not self.kfold_results:
            print("❌ No K-fold results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Fold Cross Validation Results', fontsize=16, fontweight='bold')
        
        # 1. R² scores comparison
        models = list(self.kfold_results.keys())
        r2_means = [self.kfold_results[model]['cv_r2_mean'] for model in models]
        r2_stds = [self.kfold_results[model]['cv_r2_std'] for model in models]
        
        axes[0,0].bar(models, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Cross-Validation R² Scores')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. RMSE comparison
        rmse_means = [self.kfold_results[model]['cv_rmse_mean'] for model in models]
        rmse_stds = [self.kfold_results[model]['cv_rmse_std'] for model in models]
        
        axes[0,1].bar(models, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Cross-Validation RMSE')
        axes[0,1].set_ylabel('RMSE (°C)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Box plot of R² scores across folds
        fold_data = []
        fold_labels = []
        for model in models:
            fold_data.append(self.kfold_results[model]['fold_r2_scores'])
            fold_labels.append(model)
        
        axes[1,0].boxplot(fold_data, labels=fold_labels)
        axes[1,0].set_title('R² Score Distribution Across Folds')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Model stability visualization
        stability_scores = []
        for model in models:
            results = self.kfold_results[model]
            stability = 1 - (results['cv_r2_std'] / results['cv_r2_mean']) if results['cv_r2_mean'] > 0 else 0
            stability_scores.append(stability)
        
        colors = ['green' if s > 0.9 else 'orange' if s > 0.85 else 'red' for s in stability_scores]
        axes[1,1].bar(models, stability_scores, alpha=0.7, color=colors)
        axes[1,1].set_title('Model Stability Across Folds')
        axes[1,1].set_ylabel('Stability Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Good stability')
        axes[1,1].axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Moderate stability')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"kfold_validation_plots_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 K-fold validation plots saved to: {plot_filename}")
        
        plt.show()

# Main execution
if __name__ == "__main__":
    predictor = WaterTemperaturePredictorWithKFold(n_folds=5)
    
    print("🌊 WATER TEMPERATURE PREDICTION WITH K-FOLD CROSS VALIDATION")
    print("=" * 80)
    
    # Load data with augmentation
    print("\n📂 Loading data with augmentation...")
    success = predictor.load_data_with_augmentation("imgs", "temperatures.txt")
    
    if success:
        # Train models with K-fold CV
        print("\n🚀 Training models with K-Fold Cross Validation...")
        kfold_results = predictor.train_with_kfold_cv()
        
        if kfold_results:
            # Save all trained models
            print("\n💾 Saving trained models...")
            saved_files = predictor.save_models()
            
            # Get best model info
            best_model_info = predictor.get_best_model()
            
            # Generate comprehensive report
            report_file = predictor.generate_kfold_report()
            
            # Create visualizations
            print("\n📊 Creating K-fold validation visualizations...")
            predictor.visualize_kfold_results()
            
            print(f"\n✅ K-fold validation analysis completed!")
            print(f"📄 Report: {report_file}")
            print(f"💾 Models saved in: saved_models/")
            print(f"🏆 Best model: {best_model_info['name']} (CV R² = {best_model_info['kfold_performance']['cv_r2_mean']:.4f})")
            
            # Test prediction with best model
            sample_images = [f for f in os.listdir("imgs") if f.endswith('.jpg')]
            if sample_images:
                print(f"\n🧪 Testing prediction with best model:")
                pred_temp = predictor.predict_new_image(f"imgs/{sample_images[0]}", 
                                                    model_name=best_model_info['name'])
        else:
            print("❌ K-fold validation failed")
    else:
        print("❌ Data loading failed")