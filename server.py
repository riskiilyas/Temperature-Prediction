# server_api_detailed.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import joblib
import base64
from PIL import Image
import io
import traceback
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

class DetailedWaterTempPredictionAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = None
        self.feature_names = self._create_feature_names()
        self.load_model()
    
    def _create_feature_names(self):
        """Create descriptive names for all features"""
        feature_names = []
        patches = ['center', 'top', 'bottom', 'left', 'right']
        
        for patch in patches:
            # RGB statistics (12 features per patch)
            for channel in ['R', 'G', 'B']:
                feature_names.extend([
                    f'{patch}_rgb_{channel}_mean',
                    f'{patch}_rgb_{channel}_std',
                    f'{patch}_rgb_{channel}_max',
                    f'{patch}_rgb_{channel}_min'
                ])
            
            # HSV features (3 features per patch)
            for channel in ['H', 'S', 'V']:
                feature_names.append(f'{patch}_hsv_{channel}_mean')
            
            # Brightness and contrast (2 features per patch)
            feature_names.extend([
                f'{patch}_brightness',
                f'{patch}_contrast'
            ])
        
        return feature_names
    
    def load_model(self):
        """Load the trained Gradient Boosting model"""
        try:
            model_path = r"C:\Users\ASUS\Desktop\Prediction_Regression\saved_models copy 2\gradient_boosting_model.pkl"
            scaler_path = r"C:\Users\ASUS\Desktop\Prediction_Regression\saved_models copy 2\gradient_boosting_scaler.pkl"
            info_path = r"C:\Users\ASUS\Desktop\Prediction_Regression\saved_models copy 2\model_info.pkl"
            
            # Load model
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
            
            # Load scaler if exists
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded from {scaler_path}")
            
            # Load model info if exists
            if os.path.exists(info_path):
                import pickle
                with open(info_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                print(f"✅ Model info loaded")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
    
    def extract_enhanced_features_detailed(self, image, patch_size=64):
        """
        Extract features with detailed breakdown and visualization
        """
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        height, width = image_rgb.shape[:2]
        
        # Store all patches and their info
        patches_info = []
        all_features = []
        feature_breakdown = {}
        
        # Extract multiple patches
        patches_data = self._extract_multiple_patches_detailed(image_rgb, patch_size)
        
        for i, (patch_name, patch, patch_coords) in enumerate(patches_data):
            patch_features = []
            patch_feature_details = {}
            
            # RGB statistics
            rgb_stats = {}
            for j, channel in enumerate(['R', 'G', 'B']):
                channel_data = patch[:, :, j]
                stats = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'max': float(np.max(channel_data)),
                    'min': float(np.min(channel_data))
                }
                rgb_stats[channel] = stats
                patch_features.extend([stats['mean'], stats['std'], stats['max'], stats['min']])
            
            # HSV features
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            hsv_stats = {}
            for j, channel in enumerate(['H', 'S', 'V']):
                mean_val = float(np.mean(patch_hsv[:, :, j]))
                hsv_stats[channel] = mean_val
                patch_features.append(mean_val)
            
            # Brightness and contrast
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray_patch))
            contrast = float(np.std(gray_patch))
            
            patch_features.extend([brightness, contrast])
            
            # Store patch information
            patch_info = {
                'name': patch_name,
                'coordinates': patch_coords,
                'size': patch.shape,
                'rgb_statistics': rgb_stats,
                'hsv_statistics': hsv_stats,
                'brightness': brightness,
                'contrast': contrast,
                'feature_count': len(patch_features)
            }
            
            patches_info.append(patch_info)
            all_features.extend(patch_features)
            
            # Add to feature breakdown
            start_idx = len(feature_breakdown)
            for k, feature_name in enumerate(self.feature_names[start_idx:start_idx + len(patch_features)]):
                feature_breakdown[feature_name] = {
                    'value': patch_features[k],
                    'patch': patch_name,
                    'index': start_idx + k
                }
        
        return np.array(all_features), patches_info, feature_breakdown
    
    def _extract_multiple_patches_detailed(self, image_rgb, patch_size):
        """Extract multiple patches with detailed coordinate information"""
        height, width = image_rgb.shape[:2]
        half_patch = patch_size // 2
        
        patches_data = []
        
        # Define patch positions and names
        positions = [
            ('center', width // 2, height // 2),
            ('top', width // 2, height // 4),
            ('bottom', width // 2, 3 * height // 4),
            ('left', width // 4, height // 2),
            ('right', 3 * width // 4, height // 2)
        ]
        
        for name, x, y in positions:
            patch, coords = self._safe_extract_patch_detailed(image_rgb, x, y, half_patch)
            patches_data.append((name, patch, coords))
        
        return patches_data
    
    def _safe_extract_patch_detailed(self, image_rgb, center_x, center_y, half_patch):
        """Safely extract patch with detailed coordinate tracking"""
        height, width = image_rgb.shape[:2]
        
        start_y = max(0, center_y - half_patch)
        end_y = min(height, center_y + half_patch)
        start_x = max(0, center_x - half_patch)
        end_x = min(width, center_x + half_patch)
        
        patch = image_rgb[start_y:end_y, start_x:end_x]
        
        # Store coordinates info
        coords = {
            'center': (center_x, center_y),
            'top_left': (start_x, start_y),
            'bottom_right': (end_x, end_y),
            'actual_size': patch.shape,
            'requested_size': (half_patch * 2, half_patch * 2),
            'was_resized': False
        }
        
        # Resize if needed
        if patch.shape[0] < half_patch * 2 or patch.shape[1] < half_patch * 2:
            patch = cv2.resize(patch, (half_patch * 2, half_patch * 2))
            coords['was_resized'] = True
            coords['resized_to'] = patch.shape
            
        return patch, coords
    
    def create_visualization_image(self, original_image, patches_info, prediction_result):
        """Create visualization image with patches marked and info"""
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Colors for different patches
        colors = {
            'center': (0, 255, 0),    # Green
            'top': (255, 0, 0),       # Blue
            'bottom': (0, 0, 255),    # Red
            'left': (255, 255, 0),    # Cyan
            'right': (255, 0, 255)    # Magenta
        }
        
        # Draw patches
        for patch_info in patches_info:
            name = patch_info['name']
            coords = patch_info['coordinates']
            color = colors.get(name, (255, 255, 255))
            
            # Draw rectangle
            top_left = coords['top_left']
            bottom_right = coords['bottom_right']
            cv2.rectangle(vis_image, top_left, bottom_right, color, 2)
            
            # Add label
            label = f"{name}"
            cv2.putText(vis_image, label, 
                       (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add prediction result
        temp_text = f"Predicted: {prediction_result:.2f}C"
        cv2.putText(vis_image, temp_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(vis_image, timestamp, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def analyze_feature_importance(self, features):
        """Analyze feature importance for this specific prediction"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        feature_importance = self.model.feature_importances_
        
        # Get top important features for this prediction
        importance_analysis = []
        for i, (name, importance) in enumerate(zip(self.feature_names, feature_importance)):
            importance_analysis.append({
                'feature_name': name,
                'feature_value': float(features[i]) if i < len(features) else 0.0,
                'importance_score': float(importance),
                'rank': 0  # Will be filled after sorting
            })
        
        # Sort by importance and add rank
        importance_analysis.sort(key=lambda x: x['importance_score'], reverse=True)
        for i, item in enumerate(importance_analysis):
            item['rank'] = i + 1
        
        return {
            'top_10_features': importance_analysis[:10],
            'all_features': importance_analysis
        }
    
    def get_model_confidence(self, features):
        """
        Calculate prediction confidence for Gradient Boosting model
        """
        try:
            if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
                print(f"🔍 Calculating confidence for GB model with {len(self.model.estimators_)} estimators...")
                
                # Method 1: Individual tree predictions variance
                predictions = []
                features_reshaped = features.reshape(1, -1)
                
                # Get predictions from individual trees
                for i, estimator_stage in enumerate(self.model.estimators_):
                    if hasattr(estimator_stage, '__len__'):  # Multiple estimators per stage
                        for estimator in estimator_stage:
                            try:
                                pred = estimator.predict(features_reshaped)
                                predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
                            except:
                                continue
                    else:  # Single estimator per stage
                        try:
                            pred = estimator_stage.predict(features_reshaped)
                            predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
                        except:
                            continue
                
                if len(predictions) > 1:
                    pred_array = np.array(predictions)
                    pred_std = float(np.std(pred_array))
                    pred_mean = float(np.mean(pred_array))
                    
                    # Method 2: Use staged predictions (more reliable for GB)
                    try:
                        staged_preds = list(self.model.staged_predict(features_reshaped))
                        staged_preds = [float(pred[0]) for pred in staged_preds]
                        
                        if len(staged_preds) > 10:  # Use last portion for stability
                            recent_preds = staged_preds[-10:]
                            recent_std = float(np.std(recent_preds))
                            recent_mean = float(np.mean(recent_preds))
                            
                            # Calculate confidence based on convergence
                            # Lower std in recent predictions = higher confidence
                            if recent_mean != 0:
                                confidence_convergence = max(0.0, min(1.0, 1.0 - (recent_std / abs(recent_mean))))
                            else:
                                confidence_convergence = 0.5
                            
                            # Method 3: Learning rate based confidence
                            learning_rate = getattr(self.model, 'learning_rate', 0.1)
                            n_estimators = getattr(self.model, 'n_estimators', 100)
                            
                            # Higher learning rate and more estimators generally = more confidence
                            structure_confidence = min(1.0, (n_estimators / 100.0) * (learning_rate * 10))
                            
                            # Method 4: Feature importance based confidence
                            feature_importance = getattr(self.model, 'feature_importances_', None)
                            if feature_importance is not None:
                                # Higher concentrated importance = more confidence
                                importance_entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-10))
                                max_entropy = np.log(len(feature_importance))
                                importance_confidence = 1.0 - (importance_entropy / max_entropy)
                            else:
                                importance_confidence = 0.5
                            
                            # Combined confidence score
                            final_confidence = (
                                0.4 * confidence_convergence +
                                0.3 * structure_confidence +
                                0.3 * importance_confidence
                            )
                            
                            return {
                                'confidence_score': float(final_confidence),
                                'confidence_breakdown': {
                                    'convergence_confidence': float(confidence_convergence),
                                    'structure_confidence': float(structure_confidence),
                                    'importance_confidence': float(importance_confidence)
                                },
                                'prediction_stats': {
                                    'recent_predictions_std': float(recent_std),
                                    'recent_predictions_mean': float(recent_mean),
                                    'total_staged_predictions': len(staged_preds),
                                    'individual_tree_predictions_count': len(predictions)
                                },
                                'model_stats': {
                                    'learning_rate': float(learning_rate),
                                    'n_estimators': int(n_estimators),
                                    'max_depth': getattr(self.model, 'max_depth', None),
                                    'feature_importance_entropy': float(importance_entropy) if feature_importance is not None else None
                                },
                                'staged_predictions_sample': staged_preds[-5:] if len(staged_preds) >= 5 else staged_preds,
                                'individual_predictions_sample': predictions[:10] if len(predictions) >= 10 else predictions
                            }
                        
                    except Exception as staged_error:
                        print(f"⚠️ Staged predictions failed: {staged_error}")
                    
                    # Fallback method using individual predictions variance
                    if pred_mean != 0:
                        confidence_fallback = max(0.0, min(1.0, 1.0 - (pred_std / abs(pred_mean))))
                    else:
                        confidence_fallback = 0.5
                    
                    return {
                        'confidence_score': float(confidence_fallback),
                        'method': 'individual_predictions_variance',
                        'prediction_std': float(pred_std),
                        'prediction_mean': float(pred_mean),
                        'individual_predictions': [float(p) for p in predictions[:20]],
                        'total_predictions': len(predictions),
                        'note': 'Using fallback method due to staged prediction failure'
                    }
                
                else:
                    return {
                        'confidence_score': 0.5,
                        'error': f'Insufficient predictions gathered: {len(predictions)}',
                        'note': 'Not enough individual estimator predictions available'
                    }
            
            # Method 5: Alternative for non-ensemble models
            elif hasattr(self.model, 'predict'):
                print("🔍 Using alternative confidence calculation for non-ensemble model...")
                
                # Use prediction bounds or feature analysis
                features_reshaped = features.reshape(1, -1)
                base_prediction = self.model.predict(features_reshaped)[0]
                
                # Perturb features slightly and see prediction stability
                confidence_tests = []
                noise_levels = [0.01, 0.02, 0.05]  # Small noise levels
                
                for noise_level in noise_levels:
                    for _ in range(5):  # Multiple tests per noise level
                        noise = np.random.normal(0, noise_level, features.shape)
                        perturbed_features = features + noise
                        perturbed_prediction = self.model.predict(perturbed_features.reshape(1, -1))[0]
                        confidence_tests.append(abs(perturbed_prediction - base_prediction))
                
                if confidence_tests:
                    avg_deviation = float(np.mean(confidence_tests))
                    # Lower deviation under perturbation = higher confidence
                    confidence = max(0.0, min(1.0, 1.0 - (avg_deviation / abs(base_prediction)) if base_prediction != 0 else 0.5))
                    
                    return {
                        'confidence_score': float(confidence),
                        'method': 'perturbation_analysis',
                        'average_deviation': float(avg_deviation),
                        'base_prediction': float(base_prediction),
                        'perturbation_tests': len(confidence_tests),
                        'deviation_details': [float(d) for d in confidence_tests[:10]]
                    }
                
            return {
                'confidence_score': 0.5,
                'error': 'No suitable confidence calculation method available',
                'model_type': type(self.model).__name__,
                'available_attributes': [attr for attr in dir(self.model) if not attr.startswith('_')]
            }
            
        except Exception as e:
            print(f"❌ Confidence calculation error: {e}")
            return {
                'confidence_score': 0.0,
                'error': f'Confidence calculation failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def predict_temperature_detailed(self, image):
        """Detailed prediction with comprehensive analysis"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Extract features with detailed breakdown
            features, patches_info, feature_breakdown = self.extract_enhanced_features_detailed(image)
            features_reshaped = features.reshape(1, -1)
            
            # Apply scaling if needed
            features_scaled = features_reshaped.copy()
            scaling_info = {'applied': False}
            
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_reshaped)
                scaling_info = {
                    'applied': True,
                    'scaler_type': type(self.scaler).__name__,
                    'feature_mean': self.scaler.mean_.tolist(),
                    'feature_scale': self.scaler.scale_.tolist(),
                    'original_feature_range': {
                        'min': float(np.min(features)),
                        'max': float(np.max(features)),
                        'mean': float(np.mean(features)),
                        'std': float(np.std(features))
                    },
                    'scaled_feature_range': {
                        'min': float(np.min(features_scaled)),
                        'max': float(np.max(features_scaled)),
                        'mean': float(np.mean(features_scaled)),
                        'std': float(np.std(features_scaled))
                    }
                }
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Analyze feature importance
            importance_analysis = self.analyze_feature_importance(features)
            
            # Get model confidence
            confidence_info = self.get_model_confidence(features_scaled)
            
            # Create visualization
            vis_image = self.create_visualization_image(image, patches_info, prediction)
            
            # Convert visualization to base64 for response
            _, buffer = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            vis_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Image statistics
            image_stats = {
                'dimensions': image.shape,
                'total_pixels': image.shape[0] * image.shape[1],
                'channels': image.shape[2] if len(image.shape) > 2 else 1,
                'overall_brightness': float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                'overall_contrast': float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                'color_distribution': {
                    'red_mean': float(np.mean(image[:, :, 2])) if len(image.shape) > 2 else None,
                    'green_mean': float(np.mean(image[:, :, 1])) if len(image.shape) > 2 else None,
                    'blue_mean': float(np.mean(image[:, :, 0])) if len(image.shape) > 2 else None
                }
            }
            
            # Comprehensive response
            result = {
                # Main prediction
                'prediction': {
                    'temperature': float(prediction),
                    'unit': '°C',
                    'model_type': 'gradient_boosting',
                    'timestamp': datetime.now().isoformat()
                },
                
                # Feature analysis
                'feature_analysis': {
                    'total_features': len(features),
                    'feature_names': self.feature_names,
                    'raw_features': features.tolist(),
                    'scaled_features': features_scaled.flatten().tolist() if self.scaler else None,
                    'feature_breakdown_by_name': feature_breakdown,
                    'patches_analyzed': len(patches_info)
                },
                
                # Patch details
                'patch_details': patches_info,
                
                # Feature importance
                'feature_importance': importance_analysis,
                
                # Model confidence
                'confidence_analysis': confidence_info,
                
                # Preprocessing info
                'preprocessing': {
                    'scaling': scaling_info,
                    'patch_size': 64,
                    'patches_extracted': [p['name'] for p in patches_info]
                },
                
                # Image analysis
                'image_analysis': image_stats,
                
                # Visualization
                'visualization': {
                    'annotated_image_base64': vis_image_b64,
                    'format': 'JPEG',
                    'description': 'Original image with analyzed patches marked and prediction overlay'
                },
                
                # Model info
                'model_metadata': {
                    'model_loaded': True,
                    'scaler_loaded': self.scaler is not None,
                    'feature_count_expected': len(self.feature_names),
                    'model_info': self.model_info if self.model_info else None
                },
                
                # Processing stats
                'processing_stats': {
                    'processing_timestamp': datetime.now().isoformat(),
                    'feature_extraction_successful': True,
                    'all_patches_processed': len(patches_info) == 5
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }

# Global API instance
prediction_api = DetailedWaterTempPredictionAPI()

@app.route('/predict', methods=['POST'])
def predict_temperature():
    """Enhanced prediction endpoint with detailed analysis"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Read image
        image_file = request.files['image']
        
        # Convert to OpenCV format
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Get detailed prediction
        result = prediction_api.predict_temperature_detailed(image)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict_simple', methods=['POST'])
def predict_temperature_simple():
    """Simple prediction endpoint (original format)"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Simple prediction (original format)
        features = prediction_api.extract_enhanced_features_detailed(image)[0]
        features_reshaped = features.reshape(1, -1)
        
        if prediction_api.scaler:
            features_reshaped = prediction_api.scaler.transform(features_reshaped)
        
        prediction = prediction_api.model.predict(features_reshaped)[0]
        
        return jsonify({
            "temperature": float(prediction),
            "timestamp": datetime.now().isoformat(),
            "model_type": "gradient_boosting",
            "features_count": len(features)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... (other endpoints remain the same)

if __name__ == '__main__':
    print("🌊 Enhanced Water Temperature Prediction API")
    print("=" * 60)
    print("🚀 Starting server with detailed analysis...")
    print("📡 Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /predict - Detailed prediction with analysis")
    print("   POST /predict_simple - Simple prediction (original format)")
    print("   POST /predict_base64 - Predict from base64 image")
    print("   GET  /model_info - Model information")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)