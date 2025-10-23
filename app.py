import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
from datetime import datetime

# Import our utility modules
from utils.log_analyzer import analyze_logs_refactored, filter_issues_by_relevance, enhance_log_analysis, extract_key_utterance_components
from utils.report_generator import generate_individual_csv_reports, generate_summary_reports_targetted
from utils.ai_analyzer import AutoLogAI

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
def setup_directories():
    """Create logs, reports, and models directories if they don't exist"""
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Automotive Log Analysis Platform (ALAP)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

if 'reports_generated' not in st.session_state:
    st.session_state.reports_generated = False

if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = True  # Enable AI by default

if 'selected_session_id' not in st.session_state:
    st.session_state.selected_session_id = None

if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

def reset_analysis():
    """Reset analysis results"""
    st.session_state.analysis_results = []
    st.session_state.reports_generated = False

def process_utterances(log_result, enable_utterance_analysis=True):
    """
    Process utterance flows in the log data
    
    Args:
        log_result: The result from analyze_logs_refactored
        enable_utterance_analysis: Whether to process utterances
    
    Returns:
        Enhanced log_result with utterance data
    """
    if not enable_utterance_analysis or "error" in log_result:
        return log_result
    
    try:
        # Convert issue entries to DataFrame for utterance analysis
        if log_result.get('issue_entries'):
            df = pd.DataFrame(log_result['issue_entries'])
            
            # Ensure required columns exist with proper data types
            if 'timestamp' not in df.columns:
                # Create timestamp from date and time if available
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m-%d %H:%M:%S.%f', errors='coerce')
                else:
                    logger.warning(f"Cannot create timestamps for utterance analysis in {log_result.get('log_file_name', 'unknown file')}")
                    return log_result  # Skip utterance processing if we can't create timestamps
            
            # Ensure tag column exists (mapped from component)
            if 'tag' not in df.columns and 'component' in df.columns:
                df['tag'] = df['component']
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'tag', 'message', 'pid', 'tid', 'level']
            if all(col in df.columns for col in required_columns):
                # Enhanced parsing with utterance tracing
                enhanced_df, utterance_sessions = enhance_log_analysis(df)
                
                # Add to the result
                log_result['parsed_data'] = enhanced_df
                log_result['utterance_sessions'] = utterance_sessions
                
                # Extract key components for the voice pipeline
                log_result['voice_components'] = extract_key_utterance_components(enhanced_df)
                
                # Count statistics
                log_result['utterance_count'] = len(utterance_sessions)
                log_result['successful_utterances'] = len([s for s in utterance_sessions if not s['has_errors']])
                
                logger.info(f"Processed {len(utterance_sessions)} utterance sessions for {log_result.get('log_file_name', 'unknown file')}")
            else:
                missing_cols = [col for col in required_columns if col not in df.columns]
                logger.warning(f"Missing required columns for utterance analysis: {missing_cols}")
                
    except Exception as e:
        logger.error(f"Error processing utterances for {log_result.get('log_file_name', 'unknown file')}: {str(e)}")
    
    return log_result

def perform_analysis(uploaded_files, use_ai=True, enable_utterance_analysis=True):
    """
    Process and analyze uploaded log files
    
    Args:
        uploaded_files (list): List of uploaded file objects
        use_ai (bool): Whether to use AI enhancement
    """
    setup_directories()
    
    # Reset previous analysis
    reset_analysis()
    
    # Initialize AI module if enabled
    ai_analyzer = None
    if use_ai:
        with st.spinner("Initializing AI module..."):
            ai_analyzer = AutoLogAI(model_dir="models")
            st.session_state.ai_enabled = True
    else:
        st.session_state.ai_enabled = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing file {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        # Save the uploaded file to the logs directory
        file_path = Path("logs") / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        
        # Analyze the log file
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            result = analyze_logs_refactored(file_path)
            
            # Apply AI analysis if enabled
            if use_ai and ai_analyzer and "error" not in result:
                with st.spinner(f"Applying AI analysis to {uploaded_file.name}..."):
                    ai_result = ai_analyzer.analyze_all(result['issue_entries'])
                    
                    if "error" not in ai_result:
                        # Update issue entries with AI-enhanced data
                        result['issue_entries'] = ai_result['entries']
                        # Add AI summary to result
                        result['ai_analysis'] = {
                            'summary': ai_result['summary'],
                            'training_status': ai_analyzer.get_training_status()
                        }
            
            # Process utterances if enabled
            if enable_utterance_analysis and "error" not in result:
                with st.spinner(f"Processing utterance flows in {uploaded_file.name}..."):
                    result = process_utterances(result, enable_utterance_analysis)
            
            st.session_state.analysis_results.append(result)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("Analysis complete!")
    time.sleep(1)  # Give user time to see the completion message
    status_text.empty()
    progress_bar.empty()

def visualize_analysis():
    """Create and display visualizations based on analysis results"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please upload and analyze logs first.")
        return
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Log Level Distribution")
        # Aggregate level counts across all files
        all_levels = {'F': 0, 'E': 0, 'W': 0, 'I': 0, 'D': 0, 'V': 0}
        
        for result in st.session_state.analysis_results:
            if "error" in result:
                continue
            for level, count in result.get('level_counts', {}).items():
                if level in all_levels:
                    all_levels[level] += count
        
        # Create severity level names for better readability
        level_names = {
            'F': 'Fatal',
            'E': 'Error',
            'W': 'Warning',
            'I': 'Info',
            'D': 'Debug',
            'V': 'Verbose'
        }
        
        # Create bar chart for log levels
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Only plot levels that have data
        plot_levels = {level_names.get(k, k): v for k, v in all_levels.items() if v > 0}
        
        colors = ['darkred', 'red', 'orange', 'blue', 'green', 'gray']
        ax = sns.barplot(x=list(plot_levels.keys()), y=list(plot_levels.values()), palette=colors[:len(plot_levels)])
        plt.title("Log Level Distribution")
        plt.xlabel("Log Level")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top Components with Issues")
        # Aggregate component counts across all files
        all_components = {}
        
        for result in st.session_state.analysis_results:
            if "error" in result:
                continue
            for comp, count in result.get('component_counts', {}).items():
                if comp in all_components:
                    all_components[comp] += count
                else:
                    all_components[comp] = count
        
        # Get top 10 components
        top_components = dict(sorted(all_components.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Create bar chart for top components
        fig, ax = plt.subplots(figsize=(8, 5))
        ax = sns.barplot(x=list(top_components.keys()), y=list(top_components.values()))
        plt.title("Top 10 Components with Issues")
        plt.xlabel("Component")
        plt.ylabel("Issue Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Add AI-specific visualizations if AI was used
    if any("ai_analysis" in result for result in st.session_state.analysis_results if "error" not in result):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI-Detected Anomalies")
            
            # Count anomalies per file
            anomaly_counts = []
            file_names = []
            
            for result in st.session_state.analysis_results:
                if "error" in result or "ai_analysis" not in result:
                    continue
                    
                if "summary" in result["ai_analysis"] and "anomaly_count" in result["ai_analysis"]["summary"]:
                    anomaly_counts.append(result["ai_analysis"]["summary"]["anomaly_count"])
                    file_names.append(result["log_file_name"])
            
            if anomaly_counts:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax = sns.barplot(x=file_names, y=anomaly_counts)
                plt.title("AI-Detected Anomalies by File")
                plt.xlabel("Log File")
                plt.ylabel("Anomaly Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No anomalies detected by AI analysis.")
        
        with col2:
            st.subheader("Pattern Groups")
            
            # Count clusters per file
            cluster_counts = []
            file_names = []
            
            for result in st.session_state.analysis_results:
                if "error" in result or "ai_analysis" not in result:
                    continue
                    
                if "summary" in result["ai_analysis"] and "cluster_count" in result["ai_analysis"]["summary"]:
                    cluster_counts.append(result["ai_analysis"]["summary"]["cluster_count"])
                    file_names.append(result["log_file_name"])
            
            if cluster_counts:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax = sns.barplot(x=file_names, y=cluster_counts)
                plt.title("Pattern Groups Identified by AI")
                plt.xlabel("Log File")
                plt.ylabel("Pattern Group Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No pattern groups identified by AI analysis.")

def display_detailed_issues():
    """Display detailed issues in expandable sections"""
    if not st.session_state.analysis_results:
        return
    
    st.header("Detailed Issue Analysis")
    
    # Display tabs for each log file
    if len(st.session_state.analysis_results) > 1:
        tabs = st.tabs([result.get('log_file_name', f"Result {i+1}") 
                        for i, result in enumerate(st.session_state.analysis_results)])
        
        for i, tab in enumerate(tabs):
            result = st.session_state.analysis_results[i]
            
            # Skip if there was an error with this file
            if "error" in result:
                with tab:
                    st.error(f"Error analyzing file: {result['error']}")
                continue
            
            with tab:
                display_single_result(result, i)
    else:
        # For a single result, don't use tabs
        result = st.session_state.analysis_results[0]
        if "error" in result:
            st.error(f"Error analyzing file: {result['error']}")
        else:
            display_single_result(result, 0)

def display_single_result(result, i):
    """Display analysis results for a single log file"""
    st.subheader(f"Analysis of {result['log_file_name']}")
    st.write(f"Total Lines Processed: {result['total_lines']:,}")
    st.write(f"Total Issues Found: {result['issues_found']:,}")
    
    # Show AI status if available
    if "ai_analysis" in result and "training_status" in result["ai_analysis"]:
        ai_status = "Trained" if result["ai_analysis"]["training_status"]["is_trained"] else "Learning"
        st.write(f"AI Analysis: Enabled ({ai_status})")
        
        if "summary" in result["ai_analysis"]:
            st.write(f"AI-Detected Anomalies: {result['ai_analysis']['summary'].get('anomaly_count', 0)}")
            st.write(f"Pattern Groups: {result['ai_analysis']['summary'].get('cluster_count', 0)}")
    
    # Create filter for QA relevance
    col1, col2 = st.columns(2)
    with col1:
        qa_only = st.checkbox("Show QA Relevant Issues Only", key=f"qa_only_{i}")
    with col2:
        if "ai_analysis" in result:
            anomalies_only = st.checkbox("Show AI-Detected Anomalies Only", key=f"anomalies_{i}")
        else:
            anomalies_only = False
    
    # Get filtered issues
    filtered_issues = filter_issues_by_relevance(result['issue_entries'], qa_only)
    
    # Further filter by anomalies if requested
    if anomalies_only and "ai_analysis" in result:
        filtered_issues = [entry for entry in filtered_issues if entry.get('is_anomaly', False)]
    
    if not filtered_issues:
        st.info("No issues found with current filter settings.")
        return
    
    # Allow additional filtering
    with st.expander("Filter Options"):
        col1, col2 = st.columns(2)
        with col1:
            filter_severity = st.multiselect(
                "Filter by Severity",
                options=['F', 'E', 'W', 'I', 'D', 'V'],
                default=['F', 'E', 'W'],
                key=f"severity_{i}"
            )
        with col2:
            # Get unique components from the issues
            all_components = sorted(set(entry['component'] for entry in filtered_issues))
            filter_components = st.multiselect(
                "Filter by Component",
                options=all_components,
                key=f"components_{i}"
            )
            
        # Add cluster filter if AI analysis was performed
        if "ai_analysis" in result and any("cluster_id" in entry for entry in filtered_issues):
            cluster_ids = sorted(set(entry.get('cluster_id') for entry in filtered_issues if entry.get('cluster_id') is not None))
            
            if cluster_ids:
                filter_clusters = st.multiselect(
                    "Filter by Pattern Group ID",
                    options=cluster_ids,
                    key=f"clusters_{i}"
                )
            else:
                filter_clusters = []
        else:
            filter_clusters = []
    
    # Apply filters
    if filter_severity:
        filtered_issues = [entry for entry in filtered_issues if entry['level'] in filter_severity]
    
    if filter_components:
        filtered_issues = [entry for entry in filtered_issues if entry['component'] in filter_components]
        
    if filter_clusters:
        filtered_issues = [entry for entry in filtered_issues if entry.get('cluster_id') in filter_clusters]
    
    # Display filtered issues in a DataFrame
    if filtered_issues:
        df = pd.DataFrame(filtered_issues)
        
        # Determine columns to display
        base_columns = ['timestamp', 'level', 'component', 'message', 'pid', 'tid']
        ai_columns = []
        
        # Add AI-specific columns if they exist
        if "ai_analysis" in result:
            if any('is_anomaly' in entry for entry in filtered_issues):
                ai_columns.append('is_anomaly')
            if any('anomaly_reason' in entry for entry in filtered_issues):
                ai_columns.append('anomaly_reason')
            if any('cluster_id' in entry for entry in filtered_issues):
                ai_columns.append('cluster_id')
        
        qa_columns = ['is_qa_relevant', 'matched_qa_keywords']
        
        # Combine and filter columns that exist in the DataFrame
        all_columns = base_columns + ai_columns + qa_columns
        columns = [col for col in all_columns if col in df.columns]
        
        # Display the DataFrame
        st.dataframe(df[columns], use_container_width=True)
        st.write(f"Displaying {len(filtered_issues)} issues")
        
        # Display anomaly details if AI analysis was used
        if "ai_analysis" in result and anomalies_only:
            st.subheader("Anomaly Details")
            st.write("These logs were flagged as anomalies by the AI because they represent unusual patterns or behaviors.")
            
            for entry in filtered_issues[:10]:  # Limit to first 10 for readability
                with st.expander(f"{entry['timestamp']} - {entry['component']} - {entry['level']}"):
                    st.write(f"**Message:** {entry['message']}")
                    if 'anomaly_reason' in entry:
                        st.write(f"**Anomaly Reason:** {entry['anomaly_reason']}")
                    st.write(f"**Process ID:** {entry['pid']}")
                    st.write(f"**Thread ID:** {entry['tid']}")
        
        # Display pattern group details if AI analysis was used
        if "ai_analysis" in result and filter_clusters:
            st.subheader("Pattern Group Details")
            st.write("Messages in the same pattern group represent similar types of issues.")
            
            for cluster_id in filter_clusters:
                cluster_entries = [entry for entry in filtered_issues if entry.get('cluster_id') == cluster_id]
                if cluster_entries:
                    with st.expander(f"Pattern Group {cluster_id} - {len(cluster_entries)} entries"):
                        st.write("**Sample messages in this group:**")
                        for entry in cluster_entries[:5]:  # Show first 5 examples
                            st.write(f"- {entry['message']}")
    else:
        st.info("No issues match the selected filters.")

def display_ai_insights():
    """Display AI-specific insights and findings"""
    if not st.session_state.analysis_results or not st.session_state.ai_enabled:
        return
    
    # Check if any results have AI analysis
    if not any("ai_analysis" in result for result in st.session_state.analysis_results if "error" not in result):
        return
        
    st.header("🧠 AI Insights")
    
    # Collect all root causes across files
    all_root_causes = []
    
    for result in st.session_state.analysis_results:
        if "error" in result or "ai_analysis" not in result:
            continue
            
        if ("summary" in result["ai_analysis"] and 
            "root_cause_analysis" in result["ai_analysis"]["summary"] and 
            result["ai_analysis"]["summary"]["root_cause_analysis"] is not None and
            "root_causes" in result["ai_analysis"]["summary"]["root_cause_analysis"]):
            
            file_causes = result["ai_analysis"]["summary"]["root_cause_analysis"]["root_causes"]
            for cause in file_causes:
                cause['log_file'] = result['log_file_name']
                all_root_causes.append(cause)
    
    # Display root causes if available
    if all_root_causes:
        st.subheader("Potential Root Causes")
        st.write("""
        The AI has identified the following sequences where an informational message 
        was shortly followed by errors or warnings, suggesting potential cause-and-effect relationships.
        """)
        
        for i, cause in enumerate(all_root_causes[:10]):  # Limit to 10 for readability
            with st.expander(f"Potential Root Cause #{i+1} - {cause['potential_cause_component']}"):
                st.write(f"**Log File:** {cause.get('log_file', 'Unknown')}")
                st.write(f"**Component:** {cause.get('potential_cause_component', 'Unknown')}")
                st.write(f"**Preceding Message (potential cause):**")
                st.code(cause.get('potential_cause_message', 'Unknown'))
                st.write(f"**Followed By Error:**")
                st.code(cause.get('first_error', 'Unknown'))
                st.write(f"**Timeframe:** {cause.get('timestamp_start', '')} to {cause.get('timestamp_end', '')}")
                st.write(f"**Error Count in Sequence:** {cause.get('error_count', 0)}")
    
    # Display model training status
    for result in st.session_state.analysis_results:
        if "error" not in result and "ai_analysis" in result and "training_status" in result["ai_analysis"]:
            training_status = result["ai_analysis"]["training_status"]
            
            st.subheader("AI Model Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Trained", "Yes" if training_status.get('is_trained', False) else "No")
                if training_status.get('last_updated'):
                    st.write(f"Last Updated: {training_status.get('last_updated', 'Never')}")
            
            with col2:
                # Create a simple "training progress" indicator
                if training_status.get('is_trained', False):
                    st.progress(100, "Model fully trained")
                    st.info("The AI model has been trained on your log data and is now providing more accurate insights.")
                else:
                    st.progress(25, "Model learning in progress")
                    st.info("""
                    The AI is currently using statistical methods while learning from your log data.
                    Results will improve with more log files. You can retrain the model at any time.
                    """)
            
            break  # Only show status once

def display_utterance_analysis():
    """
    Displays the utterance analysis section in the Streamlit UI.
    """
    st.header("🔊 Utterance Flow Analysis")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please upload and analyze logs first.")
        return
    
    # Process results with utterance enhancement
    enhanced_results = []
    all_sessions = []
    
    for result in st.session_state.analysis_results:
        if "error" in result:
            continue
            
        try:
            # Convert issue entries to DataFrame for utterance analysis
            if result.get('issue_entries'):
                df = pd.DataFrame(result['issue_entries'])
                
                # Ensure required columns exist with proper data types
                if 'timestamp' not in df.columns:
                    # Create timestamp from date and time if available
                    if 'date' in df.columns and 'time' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m-%d %H:%M:%S.%f', errors='coerce')
                    else:
                        continue  # Skip if we can't create timestamps
                
                # Ensure tag column exists (mapped from component)
                if 'tag' not in df.columns and 'component' in df.columns:
                    df['tag'] = df['component']
                
                # Ensure required columns exist
                required_columns = ['timestamp', 'tag', 'message', 'pid', 'tid', 'level']
                if all(col in df.columns for col in required_columns):
                    # Apply utterance enhancement
                    enhanced_df, session_details = enhance_log_analysis(df)
                    
                    # Store enhanced results
                    enhanced_result = result.copy()
                    enhanced_result['utterance_sessions'] = session_details
                    enhanced_result['enhanced_df'] = enhanced_df
                    enhanced_results.append(enhanced_result)
                    
                    # Collect sessions for display
                    for session in session_details:
                        session['log_file'] = result['log_file_name']
                        all_sessions.append(session)
        except Exception as e:
            st.warning(f"Could not process utterance analysis for {result.get('log_file_name', 'unknown file')}: {str(e)}")
            continue
    
    if not all_sessions:
        st.info("No utterance sessions were detected in the logs. This may be because:")
        st.write("- The logs don't contain voice command processing")
        st.write("- The utterance start patterns weren't found")
        st.write("- The log format doesn't include required fields")
        return
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Utterances", len(all_sessions))
    with col2:
        successful = len([s for s in all_sessions if not s['has_errors']])
        st.metric("Successful Utterances", successful)
    with col3:
        if all_sessions:
            success_rate = (successful / len(all_sessions)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        if all_sessions:
            avg_duration = sum(s['duration_ms'] for s in all_sessions) / len(all_sessions)
            st.metric("Avg Duration", f"{avg_duration:.0f}ms")
    
    # Create visualization of utterance durations
    st.subheader("Utterance Processing Times")
    
    if len(all_sessions) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        durations = [s['duration_ms'] for s in all_sessions]
        has_errors_list = [s['has_errors'] for s in all_sessions]
        utterances = [s['utterance'][:40] + '...' if len(s['utterance']) > 40 else s['utterance'] for s in all_sessions]
        
        # Create color-coded bar plot
        colors = ['salmon' if error else 'lightgreen' for error in has_errors_list]
        bars = ax.barh(range(len(utterances)), durations, color=colors)
        
        ax.set_yticks(range(len(utterances)))
        ax.set_yticklabels(utterances)
        ax.set_xlabel("Duration (ms)")
        ax.set_title("Utterance Processing Duration")
        
        # Add legend
        from matplotlib.patches import Rectangle
        error_patch = Rectangle((0,0),1,1, color='salmon', label='Had Errors')
        success_patch = Rectangle((0,0),1,1, color='lightgreen', label='Successful')
        ax.legend(handles=[success_patch, error_patch])
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Component pipeline analysis
    st.subheader("Voice Processing Pipeline Analysis")
    
    if enhanced_results:
        # Analyze components across all files
        all_components_data = []
        for result in enhanced_results:
            if 'enhanced_df' in result:
                components = extract_key_utterance_components(result['enhanced_df'])
                all_components_data.append(components)
        
        if all_components_data:
            # Combine all component data
            combined_components = {
                'voice_activation': [],
                'nlu_processing': [],
                'action_execution': [],
                'response_generation': []
            }
            
            for comp_data in all_components_data:
                for category, components in comp_data.items():
                    combined_components[category].extend(components)
            
            # Remove duplicates
            for category in combined_components:
                combined_components[category] = list(set(combined_components[category]))
            
            # Display pipeline stages
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Voice Activation Components:**")
                for comp in combined_components['voice_activation'][:10]:  # Limit display
                    st.write(f"• {comp}")
                
                st.write("**NLU Processing Components:**")
                for comp in combined_components['nlu_processing'][:10]:
                    st.write(f"• {comp}")
            
            with col2:
                st.write("**Action Execution Components:**")
                for comp in combined_components['action_execution'][:10]:
                    st.write(f"• {comp}")
                
                st.write("**Response Generation Components:**")
                for comp in combined_components['response_generation'][:10]:
                    st.write(f"• {comp}")
    
    # Show individual utterance sessions
    st.subheader("Individual Utterance Sessions")
    
    if len(all_sessions) > 10:
        st.write(f"Showing first 10 of {len(all_sessions)} sessions")
        sessions_to_show = all_sessions[:10]
    else:
        sessions_to_show = all_sessions
    
    for i, session in enumerate(sessions_to_show):
        status = "❌ Failed" if session['has_errors'] else "✅ Successful"
        duration_color = "🔴" if session['duration_ms'] > 5000 else "🟡" if session['duration_ms'] > 2000 else "🟢"
        
        with st.expander(f"{status} {duration_color}: \"{session['utterance']}\" ({session['log_file']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Session ID:** {session['session_id']}")
                st.write(f"**Duration:** {session['duration_ms']:.2f} ms")
                st.write(f"**Entry Count:** {session['entries']}")
            
            with col2:
                st.write(f"**Start Time:** {session['timestamp_start']}")
                st.write(f"**End Time:** {session['timestamp_end']}")
                st.write(f"**Has Errors:** {'Yes' if session['has_errors'] else 'No'}")
            
            st.write(f"**Components Involved:** {', '.join(session['components'][:10])}")
            if len(session['components']) > 10:
                st.write(f"... and {len(session['components']) - 10} more")
            
            # Show session timing analysis
            if session['duration_ms'] > 3000:
                st.warning(f"⚠️ Long processing time detected ({session['duration_ms']:.0f}ms)")
            
            if session['has_errors']:
                st.error("🚨 This utterance session encountered errors")
            
            # Add button to view full trace
            if st.button(f"View Full Trace", key=f"trace_{i}_{session['session_id']}"):
                st.session_state.selected_session_id = session['session_id']
                st.session_state.selected_file = session['log_file']
                st.rerun()

def display_full_trace():
    """
    Displays the full trace of an utterance session when selected.
    """
    if not hasattr(st.session_state, 'selected_session_id') or not st.session_state.selected_session_id:
        return
    
    st.header(f"Full Trace for Session {st.session_state.selected_session_id}")
    
    # Find the selected session data
    for result in st.session_state.analysis_results:
        if "error" in result or result.get('log_file_name') != st.session_state.selected_file:
            continue
        
        # Find all entries for this session
        if "parsed_data" in result:
            session_data = result["parsed_data"][
                result["parsed_data"]["session_id"] == st.session_state.selected_session_id
            ]
            
            if not session_data.empty:
                # Allow user to close this view
                if st.button("Close Full Trace"):
                    st.session_state.selected_session_id = None
                    st.session_state.selected_file = None
                    st.rerun()
                    return
                
                # Display timeline view
                st.subheader("Timeline View")
                
                # Create a table with all entries
                session_table = session_data[['timestamp', 'level', 'tag', 'message']].copy()
                
                # Highlight errors and warnings
                def highlight_level(row):
                    if row['level'] == 'E':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['level'] == 'W':
                        return ['background-color: #ffffcc'] * len(row)
                    elif row['level'] == 'F':
                        return ['background-color: #ff9999'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    session_table.style.apply(highlight_level, axis=1),
                    height=400,
                    use_container_width=True
                )
                
                # Show key events in the timeline
                st.subheader("Key Events")
                
                # Filter for important entries (errors, warnings, state changes)
                key_events = session_data[
                    (session_data['level'].isin(['E', 'W', 'F'])) | 
                    (session_data['message'].str.contains('state|response|result|complete', case=False, na=False))
                ]
                
                if not key_events.empty:
                    for _, event in key_events.iterrows():
                        level_icon = "🔴" if event['level'] == 'E' or event['level'] == 'F' else "🟡" if event['level'] == 'W' else "🔵"
                        st.write(f"{level_icon} **{event['timestamp']}** - **{event['tag']}**: {event['message']}")
                else:
                    st.info("No key events found in this session.")
                
                return  # Found and displayed the session
    
    # If we reach here, session wasn't found
    st.error(f"Session {st.session_state.selected_session_id} not found in {st.session_state.selected_file}")
    if st.button("Clear Selection"):
        st.session_state.selected_session_id = None
        st.session_state.selected_file = None
        st.rerun()

def main():
    """Main application function"""
    st.title("🚗 Automotive Log Analysis Platform (ALAP)")
    
    st.markdown("""
    Upload, analyze, and generate reports for automotive log files. This platform handles large log files,
    filters issues by severity and relevance, and generates targeted reports for QA and Development teams.
    It also includes AI capabilities for anomaly detection and root cause analysis.
    """)
    
    # AI toggle in sidebar
    st.sidebar.title("Settings")
    use_ai = st.sidebar.checkbox("Enable AI Analysis", value=True)
    enable_utterance_analysis = st.sidebar.checkbox("Enable Utterance Flow Analysis", value=True)
    
    if use_ai:
        st.sidebar.info("""
        **AI Features Enabled:**
        - Anomaly detection
        - Similar issue clustering
        - Root cause analysis
        - Pattern recognition
        
        The AI starts with statistical methods and learns from your data over time.
        """)
        
    if enable_utterance_analysis:
        st.sidebar.info("""
        **Utterance Analysis Enabled:**
        - Voice command flow tracking
        - Session correlation
        - Component pipeline analysis
        - Processing time analysis
        """)
    
    # File uploader section
    st.header("Log File Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more log files", 
        type=["log", "txt"], 
        accept_multiple_files=True
    )
    
    # Check if files are uploaded and analyze button is clicked
    if uploaded_files:
        if st.button("Analyze Logs"):
            perform_analysis(uploaded_files, use_ai=use_ai, enable_utterance_analysis=enable_utterance_analysis)
    
    # Display analysis results if available
    if st.session_state.analysis_results:
        st.header("Analysis Results")
        
        # Check for errors in all results
        if all("error" in result for result in st.session_state.analysis_results):
            st.error("Analysis failed for all uploaded files. Please check the logs.")
        else:
            # Display visualizations
            visualize_analysis()
            
            # Display AI-specific insights
            if use_ai:
                display_ai_insights()
            
            # Display utterance flow analysis
            if enable_utterance_analysis:
                display_utterance_analysis()
            
            # Show full trace if a session is selected
            display_full_trace()
            
            # Display detailed issues
            display_detailed_issues()
            
            # Report generation section
        st.header("Report Generation")

        if st.button("Generate Reports"):
            if not st.session_state.analysis_results:
                st.error("No analysis results available. Please analyze logs first.")
            else:
                with st.spinner("Generating reports..."):
                    # Generate reports
                    csv_paths = generate_individual_csv_reports(st.session_state.analysis_results)
                    qa_report_path, dev_report_path = generate_summary_reports_targetted(st.session_state.analysis_results)
                    
                    st.session_state.reports_generated = True
                    
                    # Display success message and download buttons only if reports were generated
                    if csv_paths or qa_report_path or dev_report_path:
                        st.success("Reports generated successfully!")
                        
                        # Add download buttons for CSV reports
                        if csv_paths:
                            st.write("#### CSV Reports")
                            for path in csv_paths:
                                with open(path, "rb") as file:
                                    st.download_button(
                                        label=f"Download {path.name}",
                                        data=file,
                                        file_name=path.name,
                                        mime="text/csv",
                                        key=f"csv_{path.name}"  # Add a unique key for each button
                                    )
                                st.write(f"- {path}")  # Keep the original path display for reference
                        
                        # Add download buttons for summary reports
                        if qa_report_path or dev_report_path:
                            st.write("#### Summary Reports")
                            if qa_report_path:
                                with open(qa_report_path, "rb") as file:
                                    st.download_button(
                                        label=f"Download QA Summary",
                                        data=file,
                                        file_name=qa_report_path.name,
                                        mime="text/markdown",
                                        key="qa_summary"
                                    )
                                st.write(f"- QA Summary: {qa_report_path}")  # Keep the original path display
                            
                            if dev_report_path:
                                with open(dev_report_path, "rb") as file:
                                    st.download_button(
                                        label=f"Download Developer Summary",
                                        data=file,
                                        file_name=dev_report_path.name,
                                        mime="text/markdown",
                                        key="dev_summary"
                                    )
                                st.write(f"- Developer Summary: {dev_report_path}")  # Keep the original path display
                    else:
                        st.error("Failed to generate reports. Please check the logs for errors.")

        # Show previously generated reports if they exist
        elif st.session_state.reports_generated and st.session_state.analysis_results:
            st.info("Reports have been generated for the current analysis. Click 'Generate Reports' to regenerate.")

if __name__ == "__main__":
    main()