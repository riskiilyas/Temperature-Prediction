import numpy as np
import cv2
import os
import joblib
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import argparse

class WaterTemperatureTester:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        self.available_models = []
        
        # Load available models
        self._load_available_models()
    
    def _load_available_models(self):
        """
        Scan dan load semua model yang tersedia
        """
        if not os.path.exists(self.model_dir):
            print(f"❌ Model directory tidak ditemukan: {self.model_dir}")
            return
        
        # Cari file model
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.pkl')]
        
        if not model_files:
            print(f"❌ Tidak ada model ditemukan di: {self.model_dir}")
            return
        
        print(f"📦 Loading models from: {self.model_dir}")
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            
            try:
                # Load model
                model_path = os.path.join(self.model_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                
                # Load scaler jika ada
                scaler_file = f"{model_name}_scaler.pkl"
                scaler_path = os.path.join(self.model_dir, scaler_file)
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                
                self.available_models.append(model_name)
                print(f"✅ Loaded: {model_name}")
                
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        
        # Load model info
        info_path = os.path.join(self.model_dir, "model_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.model_info = pickle.load(f)
        
        print(f"✅ Loaded {len(self.available_models)} models: {self.available_models}")
    
    def extract_enhanced_features(self, image_path, patch_size=64):
        """
        Ekstrak fitur yang sama seperti saat training
        (HARUS SAMA dengan training code!)
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
    
    def predict_single_image(self, image_path, model_name=None, show_details=True):
        """
        Prediksi suhu untuk satu gambar
        """
        if not self.available_models:
            print("❌ Tidak ada model yang tersedia")
            return None
        
        # Pilih model
        if model_name is None:
            model_name = self.available_models[0]  # Default ke model pertama
        
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' tidak tersedia")
            print(f"Available models: {self.available_models}")
            return None
        
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"❌ Image file tidak ditemukan: {image_path}")
                return None
            
            if show_details:
                print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
                print(f"🤖 Using model: {model_name}")
            
            # Extract features
            features = self.extract_enhanced_features(image_path)
            features = features.reshape(1, -1)
            
            if show_details:
                print(f"📊 Extracted {features.shape[1]} features")
            
            # Scale features if needed
            if model_name in self.scalers:
                features = self.scalers[model_name].transform(features)
                if show_details:
                    print("⚙️ Features scaled")
            
            # Predict
            prediction = self.models[model_name].predict(features)[0]
            
            if show_details:
                print(f"🌡️ Predicted temperature: {prediction:.2f}°C")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
            return None
    
    def predict_multiple_images(self, image_folder, model_name=None, save_results=True):
        """
        Prediksi suhu untuk multiple images dalam folder
        """
        if not os.path.exists(image_folder):
            print(f"❌ Folder tidak ditemukan: {image_folder}")
            return None
        
        # Cari semua image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ Tidak ada image ditemukan di: {image_folder}")
            return None
        
        print(f"📂 Found {len(image_files)} images in {image_folder}")
        
        # Pilih model
        if model_name is None:
            model_name = self.available_models[0]
        
        results = []
        successful_predictions = 0
        
        print(f"🔄 Processing images with model: {model_name}")
        print("-" * 60)
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(image_folder, filename)
            
            print(f"[{i:2d}/{len(image_files)}] {filename:<30}", end=" ")
            
            prediction = self.predict_single_image(image_path, model_name, show_details=False)
            
            if prediction is not None:
                print(f"-> {prediction:6.2f}°C ✅")
                results.append({
                    'filename': filename,
                    'predicted_temp': prediction,
                    'status': 'success'
                })
                successful_predictions += 1
            else:
                print("-> ERROR ❌")
                results.append({
                    'filename': filename,
                    'predicted_temp': None,
                    'status': 'error'
                })
        
        print("-" * 60)
        print(f"✅ Successfully processed: {successful_predictions}/{len(image_files)} images")
        
        # Save results to file
        if save_results and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"prediction_results_{timestamp}.txt"
            
            with open(results_file, 'w') as f:
                f.write("WATER TEMPERATURE PREDICTION RESULTS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model used: {model_name}\n")
                f.write(f"Total images: {len(image_files)}\n")
                f.write(f"Successful predictions: {successful_predictions}\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                for result in results:
                    if result['status'] == 'success':
                        f.write(f"{result['filename']:<30} {result['predicted_temp']:6.2f}°C\n")
                    else:
                        f.write(f"{result['filename']:<30} ERROR\n")
                
                # Statistics
                successful_temps = [r['predicted_temp'] for r in results if r['status'] == 'success']
                if successful_temps:
                    f.write(f"\nSTATISTICS:\n")
                    f.write(f"Mean temperature: {np.mean(successful_temps):.2f}°C\n")
                    f.write(f"Min temperature:  {np.min(successful_temps):.2f}°C\n")
                    f.write(f"Max temperature:  {np.max(successful_temps):.2f}°C\n")
                    f.write(f"Std deviation:    {np.std(successful_temps):.2f}°C\n")
            
            print(f"📄 Results saved to: {results_file}")
        
        return results
    
    def compare_models(self, image_path):
        """
        Bandingkan prediksi dari semua model untuk satu gambar
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file tidak ditemukan: {image_path}")
            return None
        
        if len(self.available_models) < 2:
            print("❌ Minimal 2 model diperlukan untuk comparison")
            return None
        
        print(f"🔍 Comparing all models for: {os.path.basename(image_path)}")
        print("=" * 60)
        
        results = {}
        
        for model_name in self.available_models:
            prediction = self.predict_single_image(image_path, model_name, show_details=False)
            results[model_name] = prediction
            
            if prediction is not None:
                print(f"{model_name:<20} -> {prediction:6.2f}°C")
            else:
                print(f"{model_name:<20} -> ERROR")
        
        # Statistics
        valid_predictions = [temp for temp in results.values() if temp is not None]
        
        if len(valid_predictions) > 1:
            print("-" * 60)
            print(f"Mean prediction:     {np.mean(valid_predictions):6.2f}°C")
            print(f"Std deviation:       {np.std(valid_predictions):6.2f}°C")
            print(f"Min prediction:      {np.min(valid_predictions):6.2f}°C")
            print(f"Max prediction:      {np.max(valid_predictions):6.2f}°C")
            print(f"Prediction range:    {np.max(valid_predictions) - np.min(valid_predictions):6.2f}°C")
        
        return results
    
    def visualize_prediction(self, image_path, model_name=None):
        """
        Visualisasi gambar dengan prediksi suhu
        """
        prediction = self.predict_single_image(image_path, model_name)
        
        if prediction is None:
            return
        
        # Load dan display image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f'Original Image\n{os.path.basename(image_path)}')
        plt.axis('off')
        
        # Image with prediction overlay
        plt.subplot(1, 2, 2)
        plt.imshow(image_rgb)
        
        # Add prediction text overlay
        plt.text(0.02, 0.98, f'Predicted Temperature:\n{prediction:.2f}°C', 
                transform=plt.gca().transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw patch locations
        height, width = image_rgb.shape[:2]
        patch_size = 64
        half_patch = patch_size // 2
        
        # Patch locations
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
            plt.gca().add_patch(rect)
            plt.text(x, y - half_patch - 10, label, ha='center', va='bottom', 
                    color='yellow', fontweight='bold', fontsize=10)
        
        plt.title(f'Temperature Analysis\nFeature extraction regions')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"prediction_viz_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_file}")
        
        plt.show()
    
    def list_models(self):
        """
        List semua model yang tersedia
        """
        print("📋 AVAILABLE MODELS:")
        print("=" * 40)
        
        if not self.available_models:
            print("❌ No models found")
            return
        
        for i, model_name in enumerate(self.available_models, 1):
            model_file = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            scaler_file = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            
            print(f"{i}. {model_name}")
            print(f"   Model: {model_file}")
            if os.path.exists(scaler_file):
                print(f"   Scaler: {scaler_file}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Water Temperature Prediction Tester')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--model', type=str, help='Model name to use (default: auto-select)')
    parser.add_argument('--models_dir', type=str, default='saved_models', help='Directory containing saved models')
    parser.add_argument('--compare', action='store_true', help='Compare all models on single image')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = WaterTemperatureTester(model_dir=args.models_dir)
    
    if args.list:
        tester.list_models()
        return
    
    if args.image:
        if args.compare:
            tester.compare_models(args.image)
        elif args.visualize:
            tester.visualize_prediction(args.image, args.model)
        else:
            tester.predict_single_image(args.image, args.model)
    
    elif args.folder:
        tester.predict_multiple_images(args.folder, args.model)
    
    else:
        # Interactive mode
        print("🌊 WATER TEMPERATURE PREDICTION TESTER")
        print("=" * 50)
        
        tester.list_models()
        
        while True:
            print("\nOPTIONS:")
            print("1. Test single image")
            print("2. Test folder of images")
            print("3. Compare models on single image")
            print("4. Visualize prediction")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                model_name = input("Enter model name (or press Enter for auto): ").strip()
                if not model_name:
                    model_name = None
                tester.predict_single_image(image_path, model_name)
            
            elif choice == '2':
                folder_path = input("Enter folder path: ").strip()
                model_name = input("Enter model name (or press Enter for auto): ").strip()
                if not model_name:
                    model_name = None
                tester.predict_multiple_images(folder_path, model_name)
            
            elif choice == '3':
                image_path = input("Enter image path: ").strip()
                tester.compare_models(image_path)
            
            elif choice == '4':
                image_path = "test_imgs/" + input("Enter image : ").strip() + ".jpg"
                model_name = input("Enter model name (or press Enter for auto): ").strip()
                if not model_name:
                    model_name = None
                tester.visualize_prediction(image_path, model_name)
            
            elif choice == '5':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option")

if __name__ == "__main__":
    main()