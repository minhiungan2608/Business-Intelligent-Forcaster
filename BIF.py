import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import os

class BusinessIntelligenceForecaster:
    """
    Core class for the Business Intelligence Forecaster (BIF) tool.
    Handles data preprocessing, model training, and predictions.
    """
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.numeric_features = []
        self.categorical_features = []
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            # Update Lasso parameters for better convergence
            'Lasso Regression': Lasso(
                alpha=1.0,
                max_iter=10000,  # Increase max iterations
                tol=0.0001      # Adjust tolerance
            ),
            'Gradient Boosting': GradientBoostingRegressor()
        }
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_importance = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            return True, f"Data loaded successfully. Shape: {self.data.shape}"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def analyze_data(self):
        """Perform initial data analysis"""
        if self.data is None:
            return "No data loaded"
        
        analysis = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "data_types": self.data.dtypes.astype(str).to_dict(),
            "numeric_summary": self.data.describe().to_dict() if not self.data.empty else {}
        }
        return analysis
    
    def prepare_data(self, target_column, test_size=0.2, random_state=42):
        """Prepare data for modeling by identifying feature types and splitting dataset"""
        if self.data is None:
            return False, "No data loaded"
        
        if target_column not in self.data.columns:
            return False, f"Target column '{target_column}' not found in data"
        
        self.target_column = target_column
        
        # Identify numeric and categorical features
        self.numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features
        if target_column in self.numeric_features:
            self.numeric_features.remove(target_column)
        if target_column in self.categorical_features:
            self.categorical_features.remove(target_column)
            
        # Check if we have any features left
        if not self.numeric_features and not self.categorical_features:
            return False, "No features available for training after removing target column"
        
        # Create X and y
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Split data
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return True, "Data prepared successfully"
        except Exception as e:
            return False, f"Error preparing data: {str(e)}"
    
    def create_preprocessor(self):
        """Create preprocessing pipeline for numeric and categorical features"""
        # Numeric preprocessing: imputation and scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing: imputation and one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessors
        preprocessor_steps = []
        if self.numeric_features:
            preprocessor_steps.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features:
            preprocessor_steps.append(('cat', categorical_transformer, self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers=preprocessor_steps)
        return self.preprocessor
    
    def train_models(self):
        """Train and evaluate multiple regression models"""
        if self.X_train is None or self.y_train is None:
            return False, "Data not prepared. Call prepare_data first."
        
        # Create preprocessor
        preprocessor = self.create_preprocessor()
        
        results = {}
        best_score = -float('inf')
        
        for name, model in self.models.items():
            try:
                # Create pipeline with preprocessing
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train and evaluate with cross-validation
                cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                          cv=5, scoring='neg_mean_squared_error')
                rmse_scores = np.sqrt(-cv_scores)
                
                # Train final model on all training data
                pipeline.fit(self.X_train, self.y_train)
                
                # Make predictions on test set
                y_pred = pipeline.predict(self.X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                results[name] = {
                    'cv_rmse': rmse_scores.mean(),
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'pipeline': pipeline
                }
                
                # Track best model
                if r2 > best_score:
                    best_score = r2
                    self.best_model = pipeline
                    self.best_model_name = name
                    
                    # Get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        # For tree-based models like Gradient Boosting
                        self.feature_importance = self.extract_feature_importance(pipeline)
                    elif hasattr(model, 'coef_'):
                        # For linear models
                        self.feature_importance = self.extract_feature_importance(pipeline)
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return True, {'results': results, 'best_model': self.best_model_name}
    
    def extract_feature_importance(self, pipeline):
        """Extract feature importance or coefficients from the model"""
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        feature_names = []
        
        # Get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    for feature in features:
                        feature_names.append(f"{feature}_encoded")
        
        # Get importance values
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            if model.coef_.ndim > 1:
                importance_values = np.abs(model.coef_[0])
            else:
                importance_values = np.abs(model.coef_)
        else:
            return None
        
        # Ensure we have the same number of names and values
        if len(feature_names) != len(importance_values):
            # Fallback to indices
            feature_names = [f"Feature {i}" for i in range(len(importance_values))]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance_values))
        return feature_importance
    
    def predict(self, input_data):
        """Make predictions using the best trained model"""
        if self.best_model is None:
            return False, "No trained model available. Train models first."
        
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.best_model.predict(input_data)
            return True, prediction
        except Exception as e:
            return False, f"Error making prediction: {str(e)}"
    
    def get_feature_importance_plot(self):
        """Create a feature importance plot"""
        if self.feature_importance is None:
            return None
        
        # Sort features by importance
        sorted_features = dict(sorted(self.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        # Take top 15 features if there are more
        if len(sorted_features) > 15:
            sorted_features = dict(list(sorted_features.items())[:15])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(sorted_features.keys()), list(sorted_features.values()))
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        
        return fig
    
    def save_model(self, file_path):
        """Save the best model to a file"""
        if self.best_model is None:
            return False, "No trained model available to save"
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'model_name': self.best_model_name,
                    'target_column': self.target_column,
                    'numeric_features': self.numeric_features,
                    'categorical_features': self.categorical_features,
                    'feature_importance': self.feature_importance
                }, f)
            return True, "Model saved successfully"
        except Exception as e:
            return False, f"Error saving model: {str(e)}"
    
    def load_model(self, file_path):
        """Load a saved model from a file"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.target_column = model_data['target_column']
            self.numeric_features = model_data['numeric_features']
            self.categorical_features = model_data['categorical_features']
            self.feature_importance = model_data['feature_importance']
            
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"


class BIForecastGUI:
    """
    GUI for the Business Intelligence Forecaster (BIF) tool.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Business Intelligence Forecaster")
        self.root.geometry("1000x700")
        
        self.forecaster = BusinessIntelligenceForecaster()
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        self.tab_data = ttk.Frame(self.tab_control)
        self.tab_model = ttk.Frame(self.tab_control)
        self.tab_predict = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_data, text='Data')
        self.tab_control.add(self.tab_model, text='Model')
        self.tab_control.add(self.tab_predict, text='Predict')
        self.tab_control.pack(expand=1, fill="both")
        
        # Set up each tab
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_predict_tab()
    
    def setup_data_tab(self):
        """Set up the Data tab"""
        # File loading section
        frame_load = ttk.LabelFrame(self.tab_data, text="Load Data")
        frame_load.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Button(frame_load, text="Load CSV", command=self.load_data).grid(row=0, column=0, padx=5, pady=5)
        self.lbl_data_status = ttk.Label(frame_load, text="No data loaded")
        self.lbl_data_status.grid(row=0, column=1, padx=5, pady=5)
        
        # Data analysis section
        frame_analysis = ttk.LabelFrame(self.tab_data, text="Data Analysis")
        frame_analysis.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        self.txt_data_analysis = tk.Text(frame_analysis, height=15, width=80)
        self.txt_data_analysis.grid(row=0, column=0, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(frame_analysis, orient='vertical', command=self.txt_data_analysis.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.txt_data_analysis['yscrollcommand'] = scrollbar.set
        
        ttk.Button(frame_analysis, text="Analyze Data", command=self.analyze_data).grid(row=1, column=0, padx=5, pady=5)
        
        # Data preparation section
        frame_prep = ttk.LabelFrame(self.tab_data, text="Data Preparation")
        frame_prep.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Label(frame_prep, text="Target Column:").grid(row=0, column=0, padx=5, pady=5)
        self.target_column_var = tk.StringVar()
        self.cmb_target_column = ttk.Combobox(frame_prep, textvariable=self.target_column_var)
        self.cmb_target_column.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_prep, text="Test Size (%):").grid(row=1, column=0, padx=5, pady=5)
        self.test_size_var = tk.StringVar(value="20")
        ttk.Entry(frame_prep, textvariable=self.test_size_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(frame_prep, text="Prepare Data", command=self.prepare_data).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        self.lbl_prep_status = ttk.Label(frame_prep, text="")
        self.lbl_prep_status.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Make the text area expandable
        self.tab_data.columnconfigure(0, weight=1)
        self.tab_data.rowconfigure(1, weight=1)
        frame_analysis.columnconfigure(0, weight=1)
        frame_analysis.rowconfigure(0, weight=1)
    
    def setup_model_tab(self):
        """Set up the Model tab"""
        # Model training section
        frame_train = ttk.LabelFrame(self.tab_model, text="Train Models")
        frame_train.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Button(frame_train, text="Train Models", command=self.train_models).grid(row=0, column=0, padx=5, pady=5)
        self.lbl_train_status = ttk.Label(frame_train, text="")
        self.lbl_train_status.grid(row=0, column=1, padx=5, pady=5)
        
        # Model results section
        frame_results = ttk.LabelFrame(self.tab_model, text="Model Results")
        frame_results.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        self.txt_model_results = tk.Text(frame_results, height=10, width=80)
        self.txt_model_results.grid(row=0, column=0, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(frame_results, orient='vertical', command=self.txt_model_results.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.txt_model_results['yscrollcommand'] = scrollbar.set
        
        # Feature importance section
        frame_importance = ttk.LabelFrame(self.tab_model, text="Feature Importance")
        frame_importance.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        
        self.frame_for_plot = ttk.Frame(frame_importance)
        self.frame_for_plot.grid(row=0, column=0, padx=5, pady=5)
        
        # Model saving section
        frame_save = ttk.LabelFrame(self.tab_model, text="Save/Load Model")
        frame_save.grid(row=3, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Button(frame_save, text="Save Model", command=self.save_model).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_save, text="Load Model", command=self.load_model).grid(row=0, column=1, padx=5, pady=5)
        self.lbl_save_status = ttk.Label(frame_save, text="")
        self.lbl_save_status.grid(row=0, column=2, padx=5, pady=5)
        
        # Make areas expandable
        self.tab_model.columnconfigure(0, weight=1)
        self.tab_model.rowconfigure(1, weight=1)
        self.tab_model.rowconfigure(2, weight=1)
        frame_results.columnconfigure(0, weight=1)
        frame_results.rowconfigure(0, weight=1)
        frame_importance.columnconfigure(0, weight=1)
        frame_importance.rowconfigure(0, weight=1)
    
    def setup_predict_tab(self):
        """Set up the Predict tab"""
        # Input section
        frame_input = ttk.LabelFrame(self.tab_predict, text="Prediction Input")
        frame_input.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        self.prediction_inputs = {}
        self.prediction_entries = {}
        self.input_row = 0
        
        ttk.Label(frame_input, text="Enter values for prediction:").grid(row=self.input_row, column=0, columnspan=2, padx=5, pady=5)
        self.input_row += 1
        
        # Prediction section
        frame_prediction = ttk.LabelFrame(self.tab_predict, text="Prediction Results")
        frame_prediction.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Button(frame_prediction, text="Make Prediction", command=self.make_prediction).grid(row=0, column=0, padx=5, pady=5)
        
        self.lbl_prediction = ttk.Label(frame_prediction, text="")
        self.lbl_prediction.grid(row=0, column=1, padx=5, pady=5)
        
        # Make areas expandable
        self.tab_predict.columnconfigure(0, weight=1)
        self.tab_predict.rowconfigure(0, weight=1)
        frame_input.columnconfigure(1, weight=1)
    
    def load_data(self):
        """Load data from a CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            success, message = self.forecaster.load_data(file_path)
            self.lbl_data_status.config(text=message)
            
            if success:
                # Update target column dropdown
                self.cmb_target_column['values'] = list(self.forecaster.data.columns)
                
                # Clear previous analysis
                self.txt_data_analysis.delete(1.0, tk.END)
    
    def analyze_data(self):
        """Analyze the loaded data"""
        analysis = self.forecaster.analyze_data()
        
        if isinstance(analysis, str):
            self.txt_data_analysis.delete(1.0, tk.END)
            self.txt_data_analysis.insert(tk.END, analysis)
        else:
            # Format and display analysis
            text = f"Data Shape: {analysis['shape']}\n\n"
            text += f"Columns: {', '.join(analysis['columns'])}\n\n"
            
            text += "Missing Values:\n"
            for col, count in analysis['missing_values'].items():
                text += f"  {col}: {count}\n"
            text += "\n"
            
            text += "Data Types:\n"
            for col, dtype in analysis['data_types'].items():
                text += f"  {col}: {dtype}\n"
            text += "\n"
            
            if analysis['numeric_summary']:
                text += "Numeric Summary (first few columns):\n"
                for col, stats in list(analysis['numeric_summary'].items())[:3]:
                    text += f"  {col}:\n"
                    for stat, value in stats.items():
                        text += f"    {stat}: {value:.2f}\n"
            
            self.txt_data_analysis.delete(1.0, tk.END)
            self.txt_data_analysis.insert(tk.END, text)
    
    def prepare_data(self):
        """Prepare data for modeling"""
        target_column = self.target_column_var.get()
        
        if not target_column:
            self.lbl_prep_status.config(text="Please select a target column")
            return
        
        try:
            test_size = float(self.test_size_var.get()) / 100
            if not (0 < test_size < 1):
                self.lbl_prep_status.config(text="Test size must be between 1 and 99")
                return
        except ValueError:
            self.lbl_prep_status.config(text="Invalid test size value")
            return
        
        success, message = self.forecaster.prepare_data(target_column, test_size=test_size)
        self.lbl_prep_status.config(text=message)
        
        if success:
            # Update prediction tab with input fields
            self.update_prediction_inputs()
    
    def update_prediction_inputs(self):
        """Update the prediction inputs based on the model features"""
        # Clear previous inputs
        for widget in self.prediction_entries.values():
            widget.destroy()
        self.prediction_inputs.clear()
        self.prediction_entries.clear()
        
        # Create input fields for each feature
        self.input_row = 1  # Reset row counter
        
        for feature in self.forecaster.numeric_features + self.forecaster.categorical_features:
            ttk.Label(self.tab_predict.winfo_children()[0], text=feature).grid(row=self.input_row, column=0, padx=5, pady=2, sticky='w')
            
            self.prediction_inputs[feature] = tk.StringVar()
            entry = ttk.Entry(self.tab_predict.winfo_children()[0], textvariable=self.prediction_inputs[feature])
            entry.grid(row=self.input_row, column=1, padx=5, pady=2, sticky='ew')
            self.prediction_entries[feature] = entry
            
            self.input_row += 1
    
    def train_models(self):
        """Train the models"""
        if self.forecaster.X_train is None:
            self.lbl_train_status.config(text="Please prepare data first")
            return
        
        self.lbl_train_status.config(text="Training models... Please wait.")
        self.root.update()
        
        success, results = self.forecaster.train_models()
        
        if success:
            # Display results
            text = f"Best Model: {results['best_model']}\n\n"
            text += "Model Performance:\n"
            
            for model_name, metrics in results['results'].items():
                if 'error' in metrics:
                    text += f"{model_name}: Error - {metrics['error']}\n"
                else:
                    text += f"{model_name}:\n"
                    text += f"  Cross-val RMSE: {metrics['cv_rmse']:.4f}\n"
                    text += f"  Test RMSE: {metrics['test_rmse']:.4f}\n"
                    text += f"  Test MAE: {metrics['test_mae']:.4f}\n"
                    text += f"  Test R²: {metrics['test_r2']:.4f}\n\n"
            
            self.txt_model_results.delete(1.0, tk.END)
            self.txt_model_results.insert(tk.END, text)
            
            self.lbl_train_status.config(text=f"Training complete. Best model: {results['best_model']}")
            
            # Display feature importance
            self.display_feature_importance()
        else:
            self.lbl_train_status.config(text=f"Error: {results}")
    
    def display_feature_importance(self):
        """Display feature importance plot"""
        # Clear previous plot
        for widget in self.frame_for_plot.winfo_children():
            widget.destroy()
        
        # Get new plot
        fig = self.forecaster.get_feature_importance_plot()
        
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.frame_for_plot)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def make_prediction(self):
        """Make a prediction using the input values"""
        if self.forecaster.best_model is None:
            self.lbl_prediction.config(text="No trained model available")
            return
        
        # Collect input values
        input_data = {}
        for feature, var in self.prediction_inputs.items():
            value = var.get()
            
            # Convert numeric values
            if feature in self.forecaster.numeric_features:
                try:
                    value = float(value)
                except ValueError:
                    self.lbl_prediction.config(text=f"Invalid value for {feature}")
                    return
            
            input_data[feature] = value
        
        # Make prediction
        success, prediction = self.forecaster.predict(input_data)
        
        if success:
            self.lbl_prediction.config(text=f"Prediction: {prediction[0]:.4f}")
        else:
            self.lbl_prediction.config(text=f"Error: {prediction}")
    
    def save_model(self):
        """Save the trained model"""
        if self.forecaster.best_model is None:
            self.lbl_save_status.config(text="No trained model to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = self.forecaster.save_model(file_path)
            self.lbl_save_status.config(text=message)
    
    def load_model(self):
        """Load a saved model"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            success, message = self.forecaster.load_model(file_path)
            self.lbl_save_status.config(text=message)
            
            if success:
                # Update prediction inputs
                self.update_prediction_inputs()
                
                # Display feature importance
                self.display_feature_importance()
                
                # Update model results
                self.txt_model_results.delete(1.0, tk.END)
                self.txt_model_results.insert(tk.END, f"Loaded model: {self.forecaster.best_model_name}")


def main():
    root = tk.Tk()
    app = BIForecastGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()