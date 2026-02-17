"""
SALARY PREDICTION USING ARTIFICIAL NEURAL NETWORKS
Training Script for Bank Customer Data
Compatible with TensorFlow 2.20.0 and Python 3.12.3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow imports
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l1_l2 

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"Python Version: 3.12.3")
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print("="*60)

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the bank customer data
        """
        print("="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("Creating sample data for testing...")
            return self.create_sample_data()
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info
        print("\n📊 First 5 rows:")
        print(df.head())
        
        # Drop unnecessary columns
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            print(f"\n✅ Dropped unnecessary columns: {existing_cols_to_drop}")
        
        # Handle missing values
        df = df.dropna()
        
        # Create target variable (Salary) if not present
        if 'Exited' in df.columns and 'Salary' not in df.columns:
            print("\n✅ Creating synthetic salary data from 'Exited' column...")
            df['Salary'] = df['Exited'].apply(
                lambda x: np.random.randint(50000, 150000) if x == 1 
                else np.random.randint(30000, 80000)
            )
        
        print(f"\n✅ Final dataset shape: {df.shape}")
        print(f"\n✅ Columns: {df.columns.tolist()}")
        
        return df
    
    def create_sample_data(self, n_samples=5000):
        """Create sample data if no file exists"""
        print("\n📊 Creating sample dataset...")
        
        np.random.seed(42)
        
        data = {
            'CreditScore': np.random.randint(300, 900, n_samples),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 100, n_samples),
            'Tenure': np.random.randint(0, 11, n_samples),
            'Balance': np.random.uniform(0, 250000, n_samples).round(2),
            'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(20000, 200000, n_samples).round(2),
            'Exited': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create synthetic salary
        df['Salary'] = df['Exited'].apply(
            lambda x: np.random.randint(50000, 150000) if x == 1 
            else np.random.randint(30000, 80000)
        )
        
        print(f"✅ Sample dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def exploratory_data_analysis(self, df):
        """
        Perform EDA and create visualizations
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Create figure for EDA
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Salary Distribution
        axes[0, 0].hist(df['Salary'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Salary Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Salary ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['Salary'].mean(), color='red', linestyle='--', 
                           label=f'Mean: ${df["Salary"].mean():,.0f}')
        axes[0, 0].legend()
        
        # 2. Salary by Geography
        if 'Geography' in df.columns:
            df.groupby('Geography')['Salary'].mean().plot(kind='bar', ax=axes[0, 1], 
                                                          color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Average Salary by Geography', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Geography')
            axes[0, 1].set_ylabel('Average Salary ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Salary by Gender
        if 'Gender' in df.columns:
            df.groupby('Gender')['Salary'].mean().plot(kind='bar', ax=axes[0, 2], 
                                                       color='lightcoral', edgecolor='black')
            axes[0, 2].set_title('Average Salary by Gender', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Gender')
            axes[0, 2].set_ylabel('Average Salary ($)')
        
        # 4. Salary vs Age
        axes[1, 0].scatter(df['Age'], df['Salary'], alpha=0.5, c='blue')
        axes[1, 0].set_title('Salary vs Age', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Salary ($)')
        
        # 5. Correlation Heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 6. Active vs Inactive Members
        if 'IsActiveMember' in df.columns:
            active_salary = df.groupby('IsActiveMember')['Salary'].mean()
            axes[1, 2].bar(['Inactive', 'Active'], active_salary.values, 
                           color=['red', 'green'], edgecolor='black')
            axes[1, 2].set_title('Average Salary: Active vs Inactive', 
                                 fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Membership Status')
            axes[1, 2].set_ylabel('Average Salary ($)')
        
        plt.suptitle('SALARY PREDICTION - EXPLORATORY DATA ANALYSIS', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def feature_engineering(self, df):
        """
        Create additional features for better prediction
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df_fe = df.copy()
        
        # 1. Age groups
        df_fe['Age_Group'] = pd.cut(df_fe['Age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['Young', 'Early Career', 'Mid Career', 'Senior', 'Expert'])
        
        # 2. Credit Score category
        if 'CreditScore' in df_fe.columns:
            df_fe['Credit_Category'] = pd.cut(df_fe['CreditScore'], 
                                              bins=[0, 580, 670, 740, 800, 850], 
                                              labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # 3. Balance to Salary ratio
        if 'Balance' in df_fe.columns and 'EstimatedSalary' in df_fe.columns:
            df_fe['Balance_to_Salary'] = df_fe['Balance'] / (df_fe['EstimatedSalary'] + 1)
        
        # 4. Age-Tenure interaction
        if 'Tenure' in df_fe.columns:
            df_fe['Age_Tenure'] = df_fe['Age'] * df_fe['Tenure']
        
        # 5. Products per tenure
        if 'NumOfProducts' in df_fe.columns and 'Tenure' in df_fe.columns:
            df_fe['Products_per_Tenure'] = df_fe['NumOfProducts'] / (df_fe['Tenure'] + 1)
        
        # 6. Estimated Salary squared
        if 'EstimatedSalary' in df_fe.columns:
            df_fe['EstimatedSalary_Squared'] = df_fe['EstimatedSalary'] ** 2
        
        # 7. Age squared
        df_fe['Age_Squared'] = df_fe['Age'] ** 2
        
        print(f"✅ Created {len(df_fe.columns) - len(df.columns)} new features")
        print(f"✅ Total features now: {len(df_fe.columns)}")
        
        return df_fe
    
    def prepare_data(self, df, target_col='Salary'):
        """
        Prepare data for training
        """
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n📊 Categorical features: {categorical_cols}")
        print(f"📊 Numerical features: {numerical_cols}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Training set: {X_train.shape}")
        print(f"📊 Validation set: {X_val.shape}")
        print(f"📊 Test set: {X_test.shape}")
        
        # Create preprocessor
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        numerical_transformer = StandardScaler()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ])
        
        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Scale target
        self.scaler = StandardScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Save feature names
        self.feature_names = numerical_cols + categorical_cols
        
        print(f"\n✅ Processed training shape: {X_train_processed.shape}")
        
        return (X_train_processed, X_val_processed, X_test_processed,
                y_train_scaled, y_val_scaled, y_test_scaled,
                y_train, y_val, y_test, X_train, X_val, X_test)
    
    def build_model(self, input_dim):
        """
        Build ANN model architecture
        """
        print("\n" + "="*60)
        print("BUILDING NEURAL NETWORK MODEL")
        print("="*60)
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1)
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("\n✅ Model architecture created")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """
        Train the model
        """
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        input_dim = X_train.shape[1]
        self.model = self.build_model(input_dim)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        model_checkpoint = ModelCheckpoint(
            'models/model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\n✅ Training complete!")
        if self.history and hasattr(self.history, 'history'):
            print(f"✅ Best validation loss: {min(self.history.history['val_loss']):.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test_scaled, y_test_original):
        """
        Evaluate the model
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        
        # Convert to numpy arrays and ensure 1-dimensional
        if hasattr(y_test_original, 'values'):
            y_test_original = y_test_original.values  # Convert pandas Series to numpy array
        
        y_test_original = np.array(y_test_original).flatten()  # Ensure 1D
        y_pred = np.array(y_pred).flatten()  # Ensure 1D
        
        print(f"✅ Shapes after flattening - y_test: {y_test_original.shape}, y_pred: {y_pred.shape}")
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred)
        
        # Add epsilon to avoid division by zero in MAPE
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + epsilon))) * 100
        
        print("\n📊 EVALUATION METRICS:")
        print(f"   Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test_original, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test_original.min(), y_test_original.max()], 
                        [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Salary ($)')
        axes[0, 0].set_ylabel('Predicted Salary ($)')
        axes[0, 0].set_title('Actual vs Predicted Salary')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test_original - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Salary ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of Residuals
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training History
        if self.history is not None and hasattr(self.history, 'history'):
            axes[1, 1].plot(self.history.history['loss'], label='Training Loss')
            axes[1, 1].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training History')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Training history not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training History')
        
        plt.suptitle('MODEL EVALUATION RESULTS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_test_original, y_pred, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def save_models(self):
        """
        Save trained models and preprocessors
        """
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model in .h5 format (to match app.py)
        model_path = 'models/model.h5'
        self.model.save(model_path)
        print(f"✅ Model saved to '{model_path}'")
        
        # Save preprocessor and scaler
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print("✅ Preprocessor saved to 'models/preprocessor.pkl'")
        print("✅ Scaler saved to 'models/scaler.pkl'")
        print("✅ Feature names saved to 'models/feature_names.pkl'")
        
        # Verify files were saved correctly
        print("\n📊 Verifying saved files:")
        files_to_check = [
            'models/model.h5',
            'models/preprocessor.pkl', 
            'models/scaler.pkl',
            'models/feature_names.pkl'
        ]
        
        all_good = True
        for file in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # Size in KB
                if size > 0:
                    print(f"   ✅ {file}: {size:.2f} KB")
                else:
                    print(f"   ❌ {file}: 0 KB (EMPTY!)")
                    all_good = False
            else:
                print(f"   ❌ {file}: NOT FOUND")
                all_good = False
        
        if all_good:
            print("\n✅ All model files saved successfully!")
        else:
            print("\n⚠️ Some files may not have saved correctly.")
    
    def load_models(self):
        """
        Load saved models
        """
        print("\n" + "="*60)
        print("LOADING SAVED MODELS")
        print("="*60)
        
        try:
            self.model = models.load_model('models/model.h5')
            self.preprocessor = joblib.load('models/preprocessor.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def main():
    """
    Main training function
    """
    print("\n" + "="*60)
    print("SALARY PREDICTION MODEL TRAINING")
    print("="*60)
    print(f"Python: 3.12.3")
    print(f"TensorFlow: {tf.__version__}")
    print("="*60)
    
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Check for data file
    data_path = 'data/bank_data.csv'
    print(f"\n📂 Looking for data file: {data_path}")
    
    # Load data
    df = predictor.load_and_preprocess_data(data_path)
    
    # Perform EDA
    predictor.exploratory_data_analysis(df)
    
    # Feature engineering
    df = predictor.feature_engineering(df)
    
    # Prepare data
    (X_train, X_val, X_test, y_train_scaled, y_val_scaled, y_test_scaled,
     y_train, y_val, y_test, X_train_raw, X_val_raw, X_test_raw) = predictor.prepare_data(df)
    
    # Train model
    history = predictor.train(X_train, y_train_scaled, X_val, y_val_scaled, 
                             epochs=50, batch_size=32)
    
    # Evaluate model
    y_test_original, y_pred, metrics = predictor.evaluate(X_test, y_test_scaled, y_test)
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📊 Final Model Performance:")
    print(f"   Mean Absolute Error (MAE): ${metrics['mae']:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${metrics['rmse']:,.2f}")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
    
    print("\n📁 Files created:")
    print("   - plots/eda_plots.png")
    print("   - plots/evaluation_results.png")
    print("   - models/model.h5")
    print("   - models/preprocessor.pkl")
    print("   - models/scaler.pkl")
    print("   - models/feature_names.pkl")
    
    print("\n🚀 You can now run the Streamlit app:")
    print("   streamlit run app.py")
    
    return predictor

if __name__ == "__main__":
    main()