import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle
import logging
from pathlib import Path
import re
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoLogAI:
    """
    AI/ML enhancement module for Automotive Log Analysis Platform
    Provides anomaly detection, pattern recognition, and issue clustering
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the AI module with models and vectorizers
        
        Args:
            model_dir (Path, optional): Directory to load/save models. If None, models will be initialized but not loaded.
        """
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.anomaly_model = None
        self.vectorizer = None
        self.svd_model = None
        self.is_trained = False
        
        # Try to load existing models
        self._load_models()
        
        # If no models exist, initialize new ones
        if not self.is_trained:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize new models"""
        logger.info("Initializing new AI models")
        # Initialize anomaly detection model (Isolation Forest)
        self.anomaly_model = IsolationForest(
            n_estimators=100, 
            contamination=0.05,  # Assumes 5% of logs are anomalies
            random_state=42
        )
        
        # Initialize text vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize dimensionality reduction for text
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        
        self.is_trained = False
    
    def _load_models(self):
        """Attempt to load saved models"""
        try:
            anomaly_path = self.model_dir / "anomaly_model.joblib"
            vectorizer_path = self.model_dir / "vectorizer.joblib"
            svd_path = self.model_dir / "svd_model.joblib"
            
            if anomaly_path.exists() and vectorizer_path.exists() and svd_path.exists():
                logger.info("Loading existing AI models")
                self.anomaly_model = joblib.load(anomaly_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.svd_model = joblib.load(svd_path)
                self.is_trained = True
                logger.info("AI models loaded successfully")
            else:
                logger.info("No existing models found")
                self.is_trained = False
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_trained = False
    
    def save_models(self):
        """Save all models to disk"""
        try:
            logger.info("Saving AI models")
            joblib.dump(self.anomaly_model, self.model_dir / "anomaly_model.joblib")
            joblib.dump(self.vectorizer, self.model_dir / "vectorizer.joblib")
            joblib.dump(self.svd_model, self.model_dir / "svd_model.joblib")
            logger.info("AI models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def preprocess_logs(self, issue_entries):
        """
        Prepare log data for AI analysis
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            
        Returns:
            tuple: (df, features) DataFrame and numerical features for ML analysis
        """
        if not issue_entries:
            logger.warning("No issues to preprocess")
            return None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(issue_entries)
        
        # Extract numerical features
        df['timestamp_obj'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp_obj'].dt.hour
        df['minute'] = df['timestamp_obj'].dt.minute
        df['second'] = df['timestamp_obj'].dt.second
        df['pid_num'] = pd.to_numeric(df['pid'], errors='coerce')
        df['tid_num'] = pd.to_numeric(df['tid'], errors='coerce')
        
        # Convert severity levels to numerical values
        severity_mapping = {'F': 1, 'E': 2, 'W': 3, 'I': 4, 'D': 5, 'V': 6}
        df['level_num'] = df['level'].map(severity_mapping)
        
        # Create sequence IDs (messages that occur close in time)
        df = df.sort_values(by='timestamp_obj')
        df['seq_id'] = (df['timestamp_obj'].diff() > pd.Timedelta(seconds=5)).cumsum()
        
        # Extract basic numerical features for anomaly detection
        numerical_features = df[['level_num', 'hour', 'minute', 'second']].fillna(0)
        
        # Add one-hot encoding for components (top 20 most common)
        top_components = df['component'].value_counts().nlargest(20).index
        for component in top_components:
            numerical_features[f'comp_{component}'] = (df['component'] == component).astype(int)
        
        return df, numerical_features.fillna(0)
    
    def detect_anomalies(self, issue_entries):
        """
        Detect anomalous log entries using ML
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            
        Returns:
            dict: Original data with anomaly scores
        """
        if not issue_entries:
            return {"error": "No issues to analyze"}
        
        df, features = self.preprocess_logs(issue_entries)
        
        if features is None or features.empty:
            return {"error": "Failed to extract features from logs"}
        
        # If model is not trained or has very limited data, use statistical methods
        if not self.is_trained or len(issue_entries) < 100:
            logger.info("Using statistical anomaly detection (model not yet trained)")
            # Simple statistical method: mark entries with rare components or levels as anomalies
            component_counts = df['component'].value_counts()
            rare_components = component_counts[component_counts < 2].index
            
            # Mark entries with rare components or Fatal/Error levels as anomalies
            df['is_anomaly'] = df['component'].isin(rare_components) | df['level'].isin(['F', 'E'])
            df['anomaly_score'] = df['is_anomaly'].apply(lambda x: -10 if x else 0)
            
        else:
            logger.info("Using ML-based anomaly detection")
            # Use the trained model to predict anomalies
            df['anomaly_score'] = self.anomaly_model.decision_function(features)
            df['is_anomaly'] = self.anomaly_model.predict(features) == -1
            # Initialize rare_components for the trained model case
            component_counts = df['component'].value_counts()
            rare_components = component_counts[component_counts < 2].index
        
        # Add information about why it's considered an anomaly
        # Define the anomaly_reason function inside detect_anomalies so it has access to rare_components
        def anomaly_reason(row):
            if not row['is_anomaly']:
                return None
            
            reasons = []
            if row['level'] in ['F', 'E']:
                reasons.append(f"Severe log level: {row['level']}")
            
            component = row['component']
            if component in rare_components:
                reasons.append(f"Rare component: {component} (seen {component_counts.get(component, 0)} times)")
            
            if not reasons:
                reasons.append("Unusual pattern detected")
            
            return " | ".join(reasons)
        
        df['anomaly_reason'] = df.apply(anomaly_reason, axis=1)
        
        # Add to the original entries
        result = []
        for i, entry in enumerate(issue_entries):
            entry_copy = entry.copy()
            if i < len(df):
                entry_copy['is_anomaly'] = bool(df['is_anomaly'].iloc[i])
                entry_copy['anomaly_score'] = float(df['anomaly_score'].iloc[i])
                entry_copy['anomaly_reason'] = df['anomaly_reason'].iloc[i]
            result.append(entry_copy)
        
        return result
    
    def cluster_similar_issues(self, issue_entries, similarity_threshold=0.7):
        """
        Group similar log entries into clusters
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            similarity_threshold (float): Threshold for considering entries similar (0-1)
            
        Returns:
            dict: Entries with cluster IDs
        """
        if not issue_entries:
            return {"error": "No issues to analyze"}
        
        try:
            # Extract messages for clustering
            messages = [entry['message'] for entry in issue_entries]
            
            # Use TF-IDF to vectorize the messages
            if not self.is_trained or len(messages) > len(self.vectorizer.vocabulary_) * 2:
                # If not trained or significantly more data, fit the vectorizer
                message_vectors = self.vectorizer.fit_transform(messages)
                # Reduce dimensions for better clustering
                reduced_vectors = self.svd_model.fit_transform(message_vectors)
            else:
                # Use pre-trained vectorizer
                message_vectors = self.vectorizer.transform(messages)
                reduced_vectors = self.svd_model.transform(message_vectors)
            
            # Perform DBSCAN clustering (density-based)
            clustering = DBSCAN(
                eps=1.0 - similarity_threshold,  # Convert similarity to distance
                min_samples=2,
                metric='euclidean'
            ).fit(reduced_vectors)
            
            # Get cluster labels (-1 means no cluster/outlier)
            labels = clustering.labels_
            
            # Add cluster info to the original entries
            result = []
            for i, entry in enumerate(issue_entries):
                entry_copy = entry.copy()
                entry_copy['cluster_id'] = int(labels[i]) if i < len(labels) and labels[i] >= 0 else None
                result.append(entry_copy)
            
            return result
        
        except Exception as e:
            logger.error(f"Error clustering issues: {str(e)}")
            # Return original entries without clustering
            return issue_entries
    
    def identify_root_causes(self, issue_entries):
        """
        Identify potential root causes by analyzing temporal patterns
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            
        Returns:
            dict: Root cause analysis results
        """
        if not issue_entries or len(issue_entries) < 5:
            return {"error": "Insufficient data for root cause analysis"}
        
        df, _ = self.preprocess_logs(issue_entries)
        
        if df is None or df.empty:
            return {"error": "Failed to process log data"}
        
        # Sort by timestamp
        df = df.sort_values(by='timestamp_obj')
        
        # Look for patterns of errors following specific informational logs
        root_causes = []
        
        # Group by sequence ID
        for seq_id, group in df.groupby('seq_id'):
            if len(group) < 3:
                continue
                
            # Check if sequence contains errors/warnings after info messages
            errors = group[group['level'].isin(['F', 'E', 'W'])]
            if len(errors) < 1:
                continue
                
            # Get messages before the first error
            first_error_idx = group.index[group['level'].isin(['F', 'E', 'W'])][0]
            before_error = group.loc[:first_error_idx-1]
            
            if len(before_error) > 0:
                # This sequence has potential causes before errors
                cause = {
                    'sequence_id': int(seq_id),
                    'error_count': len(errors),
                    'first_error': errors.iloc[0]['message'],
                    'potential_cause_component': before_error.iloc[-1]['component'] if len(before_error) > 0 else None,
                    'potential_cause_message': before_error.iloc[-1]['message'] if len(before_error) > 0 else None,
                    'timestamp_start': group['timestamp'].iloc[0],
                    'timestamp_end': group['timestamp'].iloc[-1],
                }
                root_causes.append(cause)
        
        return {
            'root_cause_count': len(root_causes),
            'root_causes': root_causes[:5],  # Return top 5 most likely causes
            'analysis_method': 'Temporal sequence analysis' if self.is_trained else 'Basic temporal pattern detection'
        }
    
    def analyze_all(self, issue_entries):
        """
        Perform comprehensive AI analysis on log entries
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            
        Returns:
            dict: Analysis results
        """
        if not issue_entries:
            return {
                "error": "No issues to analyze",
                "entries": [],
                "summary": {"anomaly_count": 0, "cluster_count": 0}
            }
        
        # Step 1: Detect anomalies
        enriched_entries = self.detect_anomalies(issue_entries)
        
        # Handle errors
        if isinstance(enriched_entries, dict) and "error" in enriched_entries:
            return {
                "error": enriched_entries["error"],
                "entries": issue_entries,
                "summary": {"anomaly_count": 0, "cluster_count": 0}
            }
        
        # Step 2: Cluster similar issues
        clustered_entries = self.cluster_similar_issues(enriched_entries)
        
        # Handle errors
        if isinstance(clustered_entries, dict) and "error" in clustered_entries:
            return {
                "error": clustered_entries["error"],
                "entries": enriched_entries,
                "summary": {"anomaly_count": sum(1 for e in enriched_entries if e.get('is_anomaly', False)), 
                           "cluster_count": 0}
            }
        
        # Step 3: Root cause analysis
        root_cause_analysis = self.identify_root_causes(clustered_entries)
        
        # Count anomalies and clusters
        anomaly_count = sum(1 for entry in clustered_entries if entry.get('is_anomaly', False))
        cluster_ids = set(entry.get('cluster_id') for entry in clustered_entries if entry.get('cluster_id') is not None)
        
        # Prepare summary
        summary = {
            "anomaly_count": anomaly_count,
            "cluster_count": len(cluster_ids),
            "root_cause_analysis": root_cause_analysis if "error" not in root_cause_analysis else None
        }
        
        # If we have enough data, train or update models
        if len(issue_entries) >= 100 and not self.is_trained:
            logger.info(f"Training models with {len(issue_entries)} entries")
            self._train_models(issue_entries)
        
        return {
            "entries": clustered_entries,
            "summary": summary
        }
    
    def _train_models(self, issue_entries):
        """
        Train ML models with the provided data
        
        Args:
            issue_entries (list): List of log entries to train on
        """
        try:
            df, features = self.preprocess_logs(issue_entries)
            
            if features is None or features.empty:
                logger.error("Failed to extract features for training")
                return
                
            # Train anomaly detection model
            logger.info("Training anomaly detection model")
            self.anomaly_model.fit(features)
            
            # Train text vectorization
            messages = [entry['message'] for entry in issue_entries]
            logger.info("Training text vectorizer")
            message_vectors = self.vectorizer.fit_transform(messages)
            
            # Train dimensionality reduction
            logger.info("Training SVD model")
            self.svd_model.fit(message_vectors)
            
            # Save the trained models
            self.save_models()
            
            self.is_trained = True
            logger.info("Model training complete")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    def get_training_status(self):
        """Get information about the AI model training status"""
        return {
            "is_trained": self.is_trained,
            "anomaly_model": str(self.anomaly_model.__class__.__name__) if self.anomaly_model else None,
            "vectorizer": f"TF-IDF with {len(self.vectorizer.vocabulary_) if self.is_trained else 0} features" if self.vectorizer else None,
            "svd_model": f"SVD with {self.svd_model.n_components} components" if self.svd_model else None,
            "last_updated": datetime.fromtimestamp(self.model_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') if self.is_trained else None
        }