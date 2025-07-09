# model_tester.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import joblib
import pickle

class ModelTester:
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        self.results = []
        
        self._load_available_models()
    
    def _load_available_models(self):
        """Load semua model yang tersedia"""
        if not os.path.exists(self.models_dir):
            print(f"❌ Model directory tidak ditemukan: {self.models_dir}")
            return
        
        print(f"📦 Loading models from: {self.models_dir}")
        
        # Cari file model
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            
            try:
                # Load model
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                
                # Load scaler jika ada
                scaler_file = f"{model_name}_scaler.pkl"
                scaler_path = os.path.join(self.models_dir, scaler_file)
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                
                print(f"✅ Loaded: {model_name}")
                
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        
        # Load model info
        info_path = os.path.join(self.models_dir, "model_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.model_info = pickle.load(f)
        
        print(f"✅ Total models loaded: {len(self.models)}")
    
    def extract_enhanced_features(self, image_path, patch_size=64):
        """Extract features sama seperti training"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = []
        
        # Multiple patches
        patches = self._extract_multiple_patches(image_rgb, patch_size)
        
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
    
    def _extract_multiple_patches(self, image_rgb, patch_size):
        """Extract multiple patches"""
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
    
    def predict_single_image(self, image_path, show_details=True):
        """Test single image dengan semua model"""
        if not os.path.exists(image_path):
            print(f"❌ Image tidak ditemukan: {image_path}")
            return None
        
        if show_details:
            print(f"\n🔍 Testing image: {os.path.basename(image_path)}")
            print("=" * 60)
        
        # Extract features
        try:
            features = self.extract_enhanced_features(image_path)
            features = features.reshape(1, -1)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Scale features jika perlu
                model_features = features.copy()
                if model_name in self.scalers:
                    model_features = self.scalers[model_name].transform(model_features)
                
                # Predict
                prediction = model.predict(model_features)[0]
                predictions[model_name] = prediction
                
                if show_details:
                    print(f"{model_name:<20} -> {prediction:6.2f}°C")
                
            except Exception as e:
                if show_details:
                    print(f"{model_name:<20} -> ERROR: {e}")
                predictions[model_name] = None
        
        if show_details and len(predictions) > 1:
            valid_preds = [p for p in predictions.values() if p is not None]
            if len(valid_preds) > 1:
                print("-" * 60)
                print(f"Mean prediction:     {np.mean(valid_preds):6.2f}°C")
                print(f"Std deviation:       {np.std(valid_preds):6.2f}°C")
                print(f"Range:               {np.max(valid_preds) - np.min(valid_preds):6.2f}°C")
        
        return predictions
    
    def test_folder(self, test_folder, save_results=True):
        """Test semua gambar dalam folder"""
        if not os.path.exists(test_folder):
            print(f"❌ Test folder tidak ditemukan: {test_folder}")
            return None
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(test_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ Tidak ada gambar ditemukan di: {test_folder}")
            return None
        
        print(f"\n🧪 Testing {len(image_files)} images from: {test_folder}")
        print("=" * 80)
        
        all_results = []
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(test_folder, filename)
            print(f"[{i:2d}/{len(image_files)}] {filename:<30}", end=" ")
            
            predictions = self.predict_single_image(image_path, show_details=False)
            
            if predictions:
                result = {'filename': filename}
                result.update(predictions)
                all_results.append(result)
                
                # Show predictions in one line
                pred_str = " | ".join([f"{name}: {pred:5.1f}°C" for name, pred in predictions.items() if pred is not None])
                print(f"-> {pred_str}")
            else:
                print("-> ERROR")
        
        if save_results and all_results:
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"test_results_{timestamp}.csv"
            
            df = pd.DataFrame(all_results)
            df.to_csv(csv_filename, index=False)
            print(f"\n💾 Results saved to: {csv_filename}")
            
            # Statistics
            self._show_statistics(df)
        
        return all_results
    
    def _show_statistics(self, df):
        """Show prediction statistics"""
        print(f"\n📊 PREDICTION STATISTICS:")
        print("-" * 50)
        
        model_cols = [col for col in df.columns if col != 'filename']
        
        for model_name in model_cols:
            valid_preds = df[model_name].dropna()
            if len(valid_preds) > 0:
                print(f"{model_name}:")
                print(f"  Mean: {valid_preds.mean():6.2f}°C")
                print(f"  Std:  {valid_preds.std():6.2f}°C")
                print(f"  Min:  {valid_preds.min():6.2f}°C")
                print(f"  Max:  {valid_preds.max():6.2f}°C")
                print(f"  Count: {len(valid_preds)}")
                print()
    
    def compare_with_actual(self, test_folder, actual_temps_file):
        """Compare predictions dengan actual temperatures"""
        # Load actual temperatures
        actual_temps = {}
        if os.path.exists(actual_temps_file):
            with open(actual_temps_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = os.path.basename(parts[0].strip())
                        temp = float(parts[1].strip())
                        actual_temps[filename] = temp
        
        print(f"📊 Loaded {len(actual_temps)} actual temperatures")
        
        # Test predictions
        results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(image_extensions) and filename in actual_temps:
                image_path = os.path.join(test_folder, filename)
                predictions = self.predict_single_image(image_path, show_details=False)
                
                if predictions:
                    result = {
                        'filename': filename,
                        'actual_temp': actual_temps[filename]
                    }
                    result.update(predictions)
                    results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            
            # Calculate errors
            model_cols = [col for col in df.columns if col not in ['filename', 'actual_temp']]
            
            print(f"\n📊 ACCURACY COMPARISON:")
            print("-" * 60)
            print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
            print("-" * 60)
            
            for model_name in model_cols:
                valid_idx = df[model_name].notna()
                if valid_idx.sum() > 0:
                    actual = df.loc[valid_idx, 'actual_temp']
                    predicted = df.loc[valid_idx, model_name]
                    
                    mae = np.mean(np.abs(actual - predicted))
                    rmse = np.sqrt(np.mean((actual - predicted)**2))
                    
                    # R² calculation
                    ss_res = np.sum((actual - predicted)**2)
                    ss_tot = np.sum((actual - np.mean(actual))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    print(f"{model_name:<20} {mae:<8.2f} {rmse:<8.2f} {r2:<8.3f}")
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"accuracy_comparison_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"\n💾 Detailed results saved to: {csv_filename}")
            
            return df
        
        return None
    
    def visualize_predictions(self, image_path, save_plot=True):
        """Visualize predictions untuk satu gambar"""
        predictions = self.predict_single_image(image_path, show_details=False)
        
        if not predictions:
            print("❌ No predictions available")
            return
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title(f'Test Image\n{os.path.basename(image_path)}')
        axes[0].axis('off')
        
        # Add patch locations
        height, width = image_rgb.shape[:2]
        patch_size = 64
        half_patch = patch_size // 2
        
        locations = [
            (width // 2, height // 2, 'Center'),
            (width // 2, height // 4, 'Top'),
            (width // 2, 3 * height // 4, 'Bottom'),
            (width // 4, height // 2, 'Left'),
            (3 * width // 4, height // 2, 'Right')
        ]
        
        for x, y, label in locations:
            rect = plt.Rectangle((x - half_patch, y - half_patch), 
                               patch_size, patch_size, 
                               linewidth=2, edgecolor='yellow', facecolor='none')
            axes[0].add_patch(rect)
        
        # Predictions bar chart
        model_names = []
        pred_values = []
        colors = []
        
        for model_name, pred_value in predictions.items():
            if pred_value is not None:
                model_names.append(model_name.replace('_', '\n'))
                pred_values.append(pred_value)
                
                # Color coding
                if pred_value < 25:
                    colors.append('lightblue')
                elif pred_value < 30:
                    colors.append('lightgreen')
                elif pred_value < 35:
                    colors.append('orange')
                else:
                    colors.append('lightcoral')
        
        if pred_values:
            bars = axes[1].bar(model_names, pred_values, color=colors, alpha=0.7)
            axes[1].set_title('Temperature Predictions')
            axes[1].set_ylabel('Temperature (°C)')
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, pred_values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}°C', ha='center', va='bottom', fontweight='bold')
            
            # Statistics
            if len(pred_values) > 1:
                mean_pred = np.mean(pred_values)
                std_pred = np.std(pred_values)
                axes[1].axhline(y=mean_pred, color='red', linestyle='--', alpha=0.7, 
                              label=f'Mean: {mean_pred:.1f}°C')
                axes[1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"prediction_test_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {plot_filename}")
        
        plt.show()
    
    def interactive_tester(self):
        """Interactive testing mode"""
        print("\n🧪 INTERACTIVE MODEL TESTER")
        print("=" * 50)
        print(f"Available models: {list(self.models.keys())}")
        
        while True:
            print("\nOPTIONS:")
            print("1. Test single image")
            print("2. Test folder of images")
            print("3. Compare with actual temperatures")
            print("4. Visualize predictions")
            print("5. Show model info")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                self.predict_single_image(image_path)
            
            elif choice == '2':
                folder_path = input("Enter folder path: ").strip()
                self.test_folder(folder_path)
            
            elif choice == '3':
                folder_path = input("Enter test folder path: ").strip()
                temps_file = input("Enter actual temperatures file: ").strip()
                self.compare_with_actual(folder_path, temps_file)
            
            elif choice == '4':
                image_path = input("Enter image path: ").strip()
                self.visualize_predictions(image_path)
            
            elif choice == '5':
                self.show_model_info()
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option")
    
    def show_model_info(self):
        """Show information about loaded models"""
        print("\n📋 MODEL INFORMATION:")
        print("=" * 50)
        print(f"Models directory: {self.models_dir}")
        print(f"Total models loaded: {len(self.models)}")
        print(f"Available models: {list(self.models.keys())}")
        
        if self.model_info:
            print(f"\nDataset info:")
            if 'temperature_stats' in self.model_info:
                stats = self.model_info['temperature_stats']
                print(f"  Temperature range: {stats['min']:.1f}°C - {stats['max']:.1f}°C")
                print(f"  Temperature mean: {stats['mean']:.1f}°C")
            if 'feature_count' in self.model_info:
                print(f"  Feature count: {self.model_info['feature_count']}")

if __name__ == "__main__":
    # Initialize tester
    tester = ModelTester(models_dir="saved_hybrid_models")
    
    if len(tester.models) == 0:
        print("❌ No models found! Please train models first.")
    else:
        # Run interactive tester
        tester.interactive_tester()