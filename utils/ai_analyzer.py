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
    
    def __init__(self, model_dir=None, config_manager=None):
        """
        Initialize the AI module with models and vectorizers
        
        Args:
            model_dir (Path, optional): Directory to load/save models
            config_manager (ConfigManager, optional): Configuration manager for ML parameters
        """
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL FIX #1: Make contamination_rate configurable
        self.config = config_manager
        if self.config:
            self.contamination_rate = self.config.get("ml_model.contamination_rate", 0.05)
            self.sequence_time_threshold = self.config.get("ml_model.sequence_time_threshold_seconds", 5)
        else:
            self.contamination_rate = 0.05
            self.sequence_time_threshold = 5
        
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
        self.anomaly_model = IsolationForest(
            n_estimators=100, 
            contamination=self.contamination_rate,
            random_state=42
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
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
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(issue_entries)
            
            # CRITICAL FIX #2: Validate required columns exist
            required_cols = ['timestamp', 'level', 'component', 'message', 'pid', 'tid']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None, None
            
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
            
            # CRITICAL FIX #3: Use configurable sequence time threshold
            df = df.sort_values(by='timestamp_obj')
            df['seq_id'] = (df['timestamp_obj'].diff() > pd.Timedelta(seconds=self.sequence_time_threshold)).cumsum()
            
            # Extract basic numerical features
            numerical_features = df[['level_num', 'hour', 'minute', 'second']].fillna(0)
            
            # CRITICAL FIX #4: Improve feature consistency validation
            component_counts = df['component'].value_counts()
            df['component_frequency'] = df['component'].map(component_counts)
            numerical_features['component_frequency'] = df['component_frequency'].fillna(0)
            
            # Add severity-related features
            numerical_features['is_error_or_fatal'] = (df['level'].isin(['E', 'F'])).astype(int)
            numerical_features['is_warning'] = (df['level'] == 'W').astype(int)
            
            # Add message length feature
            df['message_length'] = df['message'].astype(str).str.len()
            numerical_features['message_length'] = df['message_length'].fillna(0)
            
            # Validate that features have no NaN or inf values
            numerical_features = numerical_features.fillna(0)
            numerical_features = numerical_features.replace([np.inf, -np.inf], 0)
            
            return df, numerical_features
        
        except Exception as e:
            logger.error(f"Error preprocessing logs: {str(e)}")
            return None, None
    
    def detect_anomalies(self, issue_entries):
        """
        Detect anomalous log entries using ML
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            
        Returns:
            list: Original data with anomaly scores
        """
        if not issue_entries:
            return issue_entries
        
        try:
            df, features = self.preprocess_logs(issue_entries)
            
            if features is None or features.empty:
                return issue_entries
            
            # CRITICAL FIX #5: Add robust model fallback mechanism
            if not self.is_trained or len(issue_entries) < 100:
                logger.info("Using statistical anomaly detection (model not yet trained)")
                component_counts = df['component'].value_counts()
                rare_components = component_counts[component_counts < 2].index
                
                df['is_anomaly'] = df['component'].isin(rare_components) | df['level'].isin(['F', 'E'])
                df['anomaly_score'] = df['is_anomaly'].apply(lambda x: -10 if x else 0)
                
            else:
                logger.info("Using ML-based anomaly detection")
                try:
                    df['anomaly_score'] = self.anomaly_model.decision_function(features)
                    df['is_anomaly'] = self.anomaly_model.predict(features) == -1
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error using trained model: {str(e)}. Falling back to statistical detection.")
                    component_counts = df['component'].value_counts()
                    rare_components = component_counts[component_counts < 2].index
                    
                    df['is_anomaly'] = df['component'].isin(rare_components) | df['level'].isin(['F', 'E'])
                    df['anomaly_score'] = df['is_anomaly'].apply(lambda x: -10 if x else 0)
                    self.is_trained = False
            
            # Get component counts for anomaly reasons
            component_counts = df['component'].value_counts()
            rare_components = component_counts[component_counts < 2].index
            
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
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return issue_entries
    
    def cluster_similar_issues(self, issue_entries, similarity_threshold=0.7):
        """
        Group similar log entries into clusters
        
        Args:
            issue_entries (list): List of dictionaries containing log entries
            similarity_threshold (float): Threshold for considering entries similar (0-1)
            
        Returns:
            list: Entries with cluster IDs
        """
        if not issue_entries:
            return issue_entries
        
        try:
            messages = [entry.get('message', '') for entry in issue_entries]
            
            # CRITICAL FIX #6: Support multiple vectorization methods
            if not self.is_trained or len(messages) > len(self.vectorizer.vocabulary_) * 2:
                message_vectors = self.vectorizer.fit_transform(messages)
                reduced_vectors = self.svd_model.fit_transform(message_vectors)
            else:
                message_vectors = self.vectorizer.transform(messages)
                reduced_vectors = self.svd_model.transform(message_vectors)
            
            min_samples = 2
            clustering = DBSCAN(
                eps=1.0 - similarity_threshold,
                min_samples=min_samples,
                metric='euclidean'
            ).fit(reduced_vectors)
            
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
        
        try:
            df, _ = self.preprocess_logs(issue_entries)
            
            if df is None or df.empty:
                return {"error": "Failed to process log data"}
            
            # Sort by timestamp
            df = df.sort_values(by='timestamp_obj')
            
            root_causes = []
            
            # Group by sequence ID
            for seq_id, group in df.groupby('seq_id'):
                if len(group) < 3:
                    continue
                    
                errors = group[group['level'].isin(['F', 'E', 'W'])]
                if len(errors) < 1:
                    continue
                    
                try:
                    first_error_idx = group.index[group['level'].isin(['F', 'E', 'W'])][0]
                    before_error = group.loc[:first_error_idx-1]
                    
                    if len(before_error) > 0:
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
                except (IndexError, KeyError) as e:
                    logger.debug(f"Error processing sequence {seq_id}: {str(e)}")
                    continue
            
            return {
                'root_cause_count': len(root_causes),
                'root_causes': root_causes[:5],
                'analysis_method': 'Temporal sequence analysis' if self.is_trained else 'Basic temporal pattern detection'
            }
        
        except Exception as e:
            logger.error(f"Error analyzing root causes: {str(e)}")
            return {"error": f"Root cause analysis failed: {str(e)}"}
    
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
        
        try:
            # Step 1: Detect anomalies
            enriched_entries = self.detect_anomalies(issue_entries)
            
            if not enriched_entries:
                return {
                    "error": "Failed to detect anomalies",
                    "entries": issue_entries,
                    "summary": {"anomaly_count": 0, "cluster_count": 0}
                }
            
            # Step 2: Cluster similar issues
            clustered_entries = self.cluster_similar_issues(enriched_entries)
            
            if not clustered_entries:
                return {
                    "error": "Failed to cluster issues",
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
            if len(issue_entries) >= 100:
                if not self.is_trained:
                    logger.info(f"Training initial model with {len(issue_entries)} entries")
                    self._train_models(issue_entries)
                elif len(issue_entries) >= 500:
                    try:
                        model_age = datetime.now() - datetime.fromtimestamp(self.model_dir.stat().st_mtime)
                        if model_age.days >= 7:
                            logger.info(f"Updating model with {len(issue_entries)} new entries (model is {model_age.days} days old)")
                            self._train_models(issue_entries)
                    except (OSError, AttributeError):
                        logger.info(f"Cannot determine model age, retraining with {len(issue_entries)} entries")
                        self._train_models(issue_entries)
            
            return {
                "entries": clustered_entries,
                "summary": summary
            }
        
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "entries": issue_entries,
                "summary": {"anomaly_count": 0, "cluster_count": 0}
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
                
            # CRITICAL FIX #7: Use configurable contamination_rate for training
            logger.info("Training anomaly detection model")
            self.anomaly_model = IsolationForest(
                n_estimators=100, 
                contamination=self.contamination_rate,
                random_state=42
            )
            self.anomaly_model.fit(features)
            
            messages = [entry.get('message', '') for entry in issue_entries]
            logger.info("Training text vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            message_vectors = self.vectorizer.fit_transform(messages)
            
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
        try:
            vocab_size = len(self.vectorizer.vocabulary_) if self.vectorizer and self.is_trained else 0
            last_updated = None
            try:
                last_updated = datetime.fromtimestamp(self.model_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') if self.is_trained else None
            except (OSError, AttributeError):
                pass
            
            return {
                "is_trained": self.is_trained,
                "anomaly_model": str(self.anomaly_model.__class__.__name__) if self.anomaly_model else None,
                "vectorizer": f"TF-IDF with {vocab_size} features" if self.vectorizer else None,
                "svd_model": f"SVD with {self.svd_model.n_components} components" if self.svd_model else None,
                "last_updated": last_updated,
                "contamination_rate": self.contamination_rate
            }
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            return {
                "is_trained": self.is_trained,
                "error": str(e)
            }