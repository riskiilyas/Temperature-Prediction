import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

class WaterTemperaturePredictorV2:
    def __init__(self):
        self.rgb_features = []
        self.temperatures = []
        self.feature_names = []
        self.models = {}
        self.scalers = {}
        self.results = {}

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
        
        # Save feature extraction parameters
        feature_info = {
            'feature_count': self.rgb_features.shape[1] if len(self.rgb_features) > 0 else 0,
            'temperature_stats': {
                'mean': float(np.mean(self.temperatures)),
                'std': float(np.std(self.temperatures)),
                'min': float(np.min(self.temperatures)),
                'max': float(np.max(self.temperatures))
            }
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
        Get the best performing model
        """
        if not self.results:
            print("❌ No models trained yet")
            return None
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_model_name]
        best_scaler = self.scalers.get(best_model_name, None)
        
        print(f"🏆 Best model: {best_model_name} (R² = {self.results[best_model_name]['test_r2']:.4f})")
        
        return {
            'name': best_model_name,
            'model': best_model,
            'scaler': best_scaler,
            'performance': self.results[best_model_name]
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
    
    def train_advanced_models(self):
        """
        Train model yang lebih advanced dengan hyperparameter tuning
        """
        print("\n" + "="*70)
        print("                 ADVANCED MODEL TRAINING")
        print("="*70)
        
        if len(self.rgb_features) < 20:
            print("❌ Data masih terlalu sedikit untuk advanced training")
            return None
        
        X = self.rgb_features
        y = self.temperatures
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Data split: {len(X_train)} training, {len(X_test)} testing samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model 1: Ridge Regression with hyperparameter tuning
        self._train_ridge_regression(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        
        # Model 2: Random Forest with tuning
        self._train_tuned_random_forest(X_train, X_test, y_train, y_test)
        
        # Model 3: Gradient Boosting
        self._train_gradient_boosting(X_train, X_test, y_train, y_test)
        
        # Model comparison
        self._compare_advanced_models()
        
        return self.results
    
    def _train_ridge_regression(self, X_train, X_test, y_train, y_test, scaler):
        """
        Train Ridge Regression dengan hyperparameter tuning
        """
        print("\n🔧 Training Ridge Regression with hyperparameter tuning...")
        
        # Hyperparameter tuning
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        ridge = Ridge()
        
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        
        self.models['ridge'] = best_model
        self.scalers['ridge'] = scaler
        self.results['ridge'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_alpha': grid_search.best_params_['alpha']
        }
        
        print(f"  Best alpha: {grid_search.best_params_['alpha']}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}°C")
        print(f"  Test MAE: {test_mae:.4f}°C")
        print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _train_tuned_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Train Random Forest dengan hyperparameter tuning
        """
        print("\n🔧 Training Random Forest with hyperparameter tuning...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = best_model.feature_importances_
        
        self.models['random_forest'] = best_model
        self.results['random_forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}°C")
        print(f"  Test MAE: {test_mae:.4f}°C")
        print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """
        Train Gradient Boosting Regressor
        """
        print("\n🔧 Training Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        
        self.models['gradient_boosting'] = best_model
        self.results['gradient_boosting'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}°C")
        print(f"  Test MAE: {test_mae:.4f}°C")
        print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def _compare_advanced_models(self):
        """
        Bandingkan performa model advanced
        """
        print("\n" + "="*70)
        print("                 ADVANCED MODEL COMPARISON")
        print("="*70)
        
        # Buat comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test R²': f"{results['test_r2']:.4f}",
                'Test RMSE': f"{results['test_rmse']:.4f}°C",
                'Test MAE': f"{results['test_mae']:.4f}°C",
                'CV R² (mean)': f"{results['cv_mean']:.4f}",
                'CV Std': f"{results['cv_std']:.4f}"
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
        self._analyze_improvement()
    
    def _analyze_improvement(self):
        """
        Analisis peningkatan performa dibanding baseline
        """
        print("\n💡 PERFORMANCE IMPROVEMENT ANALYSIS:")
        
        best_r2 = max(self.results[model]['test_r2'] for model in self.results.keys())
        
        if best_r2 > 0.7:
            print("🎉 EXCELLENT! Model performance significantly improved!")
            print("   ✅ Data augmentation was very effective")
            print("   ✅ Enhanced feature extraction worked well")
            print("   ✅ Model is ready for deployment")
        elif best_r2 > 0.5:
            print("👍 GOOD! Model performance improved substantially")
            print("   ✅ Data augmentation helped")
            print("   ⚠️  Consider more feature engineering")
        elif best_r2 > 0.2:
            print("⚠️  MODERATE improvement")
            print("   ✅ Data augmentation provided some benefit")
            print("   🔧 Need more sophisticated features or more data")
        else:
            print("😞 LIMITED improvement")
            print("   ❌ RGB features may not be sufficient")
            print("   🔧 Consider different approach (CNN, texture analysis)")
    
    def predict_new_image(self, image_path, model_name='ridge'):
        """
        Prediksi suhu untuk gambar baru
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
    
    def generate_advanced_report(self):
        """
        Generate comprehensive advanced analysis report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"advanced_water_temp_analysis_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("           ADVANCED WATER TEMPERATURE PREDICTION ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset summary
            f.write("ENHANCED DATASET SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total samples (after augmentation): {len(self.rgb_features)}\n")
            f.write(f"Feature dimensions: {self.rgb_features.shape[1]}\n")
            f.write(f"Temperature range: {np.min(self.temperatures):.2f}°C - {np.max(self.temperatures):.2f}°C\n")
            f.write(f"Temperature mean: {np.mean(self.temperatures):.2f}°C\n")
            f.write(f"Temperature std: {np.std(self.temperatures):.2f}°C\n\n")
            
            # Feature engineering summary
            f.write("FEATURE ENGINEERING:\n")
            f.write("-" * 50 + "\n")
            f.write("Multiple patch extraction (5 locations)\n")
            f.write("RGB statistics (mean, std, max, min)\n")
            f.write("HSV color space features\n")
            f.write("Brightness and contrast features\n")
            f.write("Data augmentation (6x increase)\n\n")
            
            # Model results
            if self.results:
                f.write("ADVANCED MODEL PERFORMANCE:\n")
                f.write("-" * 50 + "\n")
                for model_name, results in self.results.items():
                    f.write(f"{model_name.upper().replace('_', ' ')}:\n")
                    f.write(f"  Test R²: {results['test_r2']:.4f}\n")
                    f.write(f"  Test RMSE: {results['test_rmse']:.4f}°C\n")
                    f.write(f"  Test MAE: {results['test_mae']:.4f}°C\n")
                    f.write(f"  CV R² (mean±std): {results['cv_mean']:.4f}±{results['cv_std']:.4f}\n")
                    if 'best_params' in results:
                        f.write(f"  Best hyperparameters: {results['best_params']}\n")
                    f.write("\n")
                
                # Best model
                best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
                f.write(f"BEST MODEL: {best_model.replace('_', ' ').title()}\n")
                f.write(f"Best Test R²: {self.results[best_model]['test_r2']:.4f}\n")
        
        print(f"\n📄 Advanced report saved to: {report_filename}")
        return report_filename

# Main execution
if __name__ == "__main__":
    predictor = WaterTemperaturePredictorV2()
    
    print("🌊 ADVANCED WATER TEMPERATURE PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Load data with augmentation
    print("\n📂 Loading data with augmentation...")
    success = predictor.load_data_with_augmentation("imgs", "temperatures.txt")
    
    if success:
        # Train advanced models
        print("\n🚀 Training advanced models...")
        results = predictor.train_advanced_models()
        
        # Di akhir main execution, tambahkan:
        if results:
            # Save all trained models
            print("\n💾 Saving trained models...")
            saved_files = predictor.save_models()
            
            # Get best model info
            best_model_info = predictor.get_best_model()
            
            # Generate comprehensive report
            report_file = predictor.generate_advanced_report()
            
            print(f"\n✅ Advanced analysis completed!")
            print(f"📄 Report: {report_file}")
            print(f"💾 Models saved in: saved_models/")
            print(f"🏆 Best model: {best_model_info['name']} (R² = {best_model_info['performance']['test_r2']:.4f})")
            
            # Test prediction
            sample_images = [f for f in os.listdir("imgs") if f.endswith('.jpg')]
            if sample_images:
                print(f"\n🧪 Testing prediction with best model:")
                pred_temp = predictor.predict_new_image(f"imgs/{sample_images[0]}", 
                                                    model_name=best_model_info['name'])
        else:
            print("❌ Model training failed")
    else:
        print("❌ Data loading failed")