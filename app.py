import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
import re
from datetime import datetime

# Import our utility modules
from utils.log_analyzer import analyze_logs_refactored, filter_issues_by_relevance, enhance_log_analysis, extract_key_utterance_components
from utils.report_generator import generate_individual_csv_reports, generate_summary_reports_targetted
from utils.ai_analyzer import AutoLogAI
from utils.security_utils import sanitize_path, validate_file
from utils.input_validator import validate_string

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

# Initialize session state for storing analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

if 'reports_generated' not in st.session_state:
    st.session_state.reports_generated = False

if 'ai_enabled' not in st.session_state:
    st.session_state.ai_enabled = True

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
    CRITICAL FIX #8: Add security validation before file operations
    Process utterance flows in the log data
    """
    if not enable_utterance_analysis or "error" in log_result:
        return log_result
    
    try:
        # Validate the log file path
        if 'log_file_path' in log_result:
            file_path = log_result['log_file_path']
            # CRITICAL FIX #1: Add file path sanitization
            try:
                sanitized_path = sanitize_path(file_path)
                validate_file(sanitized_path)
            except Exception as e:
                logger.warning(f"File validation failed: {str(e)}")
                return log_result
        
        # Convert issue entries to DataFrame
        if log_result.get('issue_entries'):
            df = pd.DataFrame(log_result['issue_entries'])
            
            # Ensure required columns exist
            if 'timestamp' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], 
                                                    format='%m-%d %H:%M:%S.%f', errors='coerce')
                else:
                    return log_result
            
            # Ensure tag column exists (mapped from component)
            if 'tag' not in df.columns and 'component' in df.columns:
                df['tag'] = df['component']
            
            # Ensure all required columns exist
            required_columns = ['timestamp', 'tag', 'message', 'pid', 'tid', 'level']
            if all(col in df.columns for col in required_columns):
                # Initialize utterance tracer
                from utils.utterance_tracer import UtteranceTracer
                tracer = UtteranceTracer()
                
                # Get utterance sessions
                utterance_sessions = tracer.identify_utterance_sessions(df)
                session_details = tracer.extract_session_details(utterance_sessions)
                
                # Get enhanced analysis insights
                insights = enhance_log_analysis(df)
                
                # Add to the result
                log_result['parsed_data'] = df
                log_result['utterance_sessions'] = utterance_sessions
                log_result['session_details'] = session_details
                log_result['insights'] = insights
                
                # Extract key components for the voice pipeline from all messages
                all_components = []
                for message in df['message'].dropna():
                    components = extract_key_utterance_components(message)
                    if any(components.values()):  # Only add if components were found
                        all_components.append(components)
                
                log_result['voice_components'] = all_components
                
                # Count statistics - handle both dict and list formats
                if isinstance(utterance_sessions, dict):
                    log_result['utterance_count'] = len(utterance_sessions)
                    log_result['successful_utterances'] = len([s for s in utterance_sessions.values() if isinstance(s, dict) and not s.get('has_errors', False)])
                elif isinstance(utterance_sessions, list):
                    log_result['utterance_count'] = len(utterance_sessions)
                    log_result['successful_utterances'] = len([s for s in utterance_sessions if isinstance(s, dict) and not s.get('has_errors', False)])
                else:
                    log_result['utterance_count'] = 0
                    log_result['successful_utterances'] = 0
                
                logger.info(f"Processed {log_result['utterance_count']} utterance sessions")
            else:
                missing_cols = [col for col in required_columns if col not in df.columns]
                logger.warning(f"Missing required columns for utterance analysis: {missing_cols}")
    
    except Exception as e:
        logger.error(f"Error processing utterances: {str(e)}")
    
    return log_result

def perform_analysis(uploaded_files, use_ai=True, enable_utterance_analysis=True):
    """
    CRITICAL FIX #2: Implement file size validation and security checks
    Process and analyze uploaded log files
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
        
        try:
            # CRITICAL FIX #2: Add file size validation
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            if file_size_mb > 500:
                logger.error(f"File {uploaded_file.name} exceeds 500MB limit ({file_size_mb:.2f}MB)")
                st.session_state.analysis_results.append({
                    "error": f"File size ({file_size_mb:.2f}MB) exceeds 500MB limit"
                })
                continue
            
            # CRITICAL FIX #1: Add input validation for filenames
            try:
                validate_string(uploaded_file.name)
            except Exception as e:
                logger.error(f"Invalid filename: {str(e)}")
                continue
            
            # Save the uploaded file
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
                            result['issue_entries'] = ai_result['entries']
                            result['ai_analysis'] = {
                                'summary': ai_result['summary'],
                                'training_status': ai_analyzer.get_training_status()
                            }
                
                # Process utterances if enabled
                if enable_utterance_analysis and "error" not in result:
                    with st.spinner(f"Processing utterance flows in {uploaded_file.name}..."):
                        result = process_utterances(result, enable_utterance_analysis)
                
                st.session_state.analysis_results.append(result)
        
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            st.session_state.analysis_results.append({
                "error": f"Error analyzing file: {str(e)}"
            })
        
        # Update progress
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("Analysis complete!")
    time.sleep(1)
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
        
        all_levels = {'F': 0, 'E': 0, 'W': 0, 'I': 0, 'D': 0, 'V': 0}
        
        for result in st.session_state.analysis_results:
            if "error" in result:
                continue
            for level, count in result.get('level_counts', {}).items():
                if level in all_levels:
                    all_levels[level] += count
        
        # CRITICAL FIX #3: Safe division for chart
        plot_levels = {name: count for name, count in all_levels.items() if count > 0}
        
        if plot_levels:
            fig, ax = plt.subplots()
            colors = ['darkred', 'red', 'orange', 'blue', 'green', 'gray']
            ax = sns.barplot(x=list(plot_levels.keys()), y=list(plot_levels.values()), palette=colors[:len(plot_levels)])
            plt.title("Log Level Distribution")
            plt.xlabel("Log Level")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No log level data available")
    
    with col2:
        st.subheader("Top Components with Issues")
        
        all_components = {}
        
        for result in st.session_state.analysis_results:
            if "error" in result:
                continue
            for comp, count in result.get('component_counts', {}).items():
                if comp in all_components:
                    all_components[comp] += count
                else:
                    all_components[comp] = count
        
        top_components = dict(sorted(all_components.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_components:
            fig, ax = plt.subplots()
            ax = sns.barplot(x=list(top_components.values()), y=list(top_components.keys()))
            plt.title("Top 10 Components with Issues")
            plt.xlabel("Issue Count")
            plt.ylabel("Component")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No component data available")
    
    # AI-specific visualizations
    if any("ai_analysis" in result for result in st.session_state.analysis_results if "error" not in result):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI-Detected Anomalies")
            
            anomaly_counts = []
            file_names = []
            
            for result in st.session_state.analysis_results:
                if "error" in result or "ai_analysis" not in result:
                    continue
                if "summary" in result["ai_analysis"] and "anomaly_count" in result["ai_analysis"]["summary"]:
                    anomaly_counts.append(result["ai_analysis"]["summary"]["anomaly_count"])
                    file_names.append(result["log_file_name"])
            
            if anomaly_counts:
                fig, ax = plt.subplots()
                ax = sns.barplot(x=file_names, y=anomaly_counts)
                plt.title("AI-Detected Anomalies by File")
                plt.xlabel("Log File")
                plt.ylabel("Anomaly Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No anomaly data available")
        
        with col2:
            st.subheader("Pattern Groups Identified")
            
            cluster_counts = []
            file_names = []
            
            for result in st.session_state.analysis_results:
                if "error" in result or "ai_analysis" not in result:
                    continue
                if "summary" in result["ai_analysis"] and "cluster_count" in result["ai_analysis"]["summary"]:
                    cluster_counts.append(result["ai_analysis"]["summary"]["cluster_count"])
                    file_names.append(result["log_file_name"])
            
            if cluster_counts:
                fig, ax = plt.subplots()
                ax = sns.barplot(x=file_names, y=cluster_counts)
                plt.title("Pattern Groups Identified by File")
                plt.xlabel("Log File")
                plt.ylabel("Pattern Group Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No pattern group data available")

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
            
            if "error" in result:
                with tab:
                    st.error(f"Error analyzing file: {result['error']}")
                continue
            
            with tab:
                display_single_result(result, i)
    else:
        result = st.session_state.analysis_results[0]
        
        if "error" in result:
            st.error(f"Error analyzing file: {result['error']}")
        else:
            display_single_result(result, 0)

def display_single_result(result, idx):
    """Display analysis results for a single log file"""
    st.subheader(f"Analysis of {result['log_file_name']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Lines Processed", result['total_lines'])
    with col2:
        st.metric("Total Issues Found", result['issues_found'])
    with col3:
        st.metric("QA Relevant Issues", result['qa_relevant_issues'])
    
    # Show AI status if available
    if "ai_analysis" in result and "training_status" in result["ai_analysis"]:
        ai_status = result["ai_analysis"]["training_status"]
        st.subheader("🤖 AI Model Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Trained", "✓ Yes" if ai_status.get('is_trained') else "✗ No")
        with col2:
            if ai_status.get('anomaly_model'):
                st.metric("Anomaly Detection", ai_status['anomaly_model'])
        with col3:
            if ai_status.get('last_updated'):
                st.metric("Last Updated", ai_status['last_updated'])
    
    # Create filter options
    st.subheader("Filter Options")
    col1, col2 = st.columns(2)
    
    with col1:
        qa_only = st.checkbox("Show QA Relevant Issues Only", key=f"qa_only_{idx}")
    
    with col2:
        if "ai_analysis" in result:
            anomalies_only = st.checkbox("Show AI-Detected Anomalies Only", key=f"anomalies_{idx}")
        else:
            anomalies_only = False
    
    # Filter filter_severity
    filter_severity = st.multiselect(
        "Filter by Severity",
        options=['F', 'E', 'W', 'I', 'D', 'V'],
        default=['F', 'E', 'W'],
        key=f"severity_{idx}"
    )
    
    # Get filtered issues
    filtered_issues = filter_issues_by_relevance(result['issue_entries'], qa_only)
    
    if anomalies_only and "ai_analysis" in result:
        filtered_issues = [entry for entry in filtered_issues if entry.get('is_anomaly', False)]
    
    if filter_severity:
        filtered_issues = [entry for entry in filtered_issues if entry['level'] in filter_severity]
    
    if not filtered_issues:
        st.info("No issues match the current filter settings.")
        return
    
    # Display issues in a table
    st.write(f"Displaying {len(filtered_issues)} issues")
    
    # Create DataFrame for display
    display_df = pd.DataFrame(filtered_issues)
    
    # Select columns to display
    base_columns = ['timestamp', 'level', 'component', 'message', 'pid', 'tid']
    ai_columns = []
    
    if "ai_analysis" in result:
        if 'is_anomaly' in display_df.columns:
            ai_columns.append('is_anomaly')
        if 'anomaly_reason' in display_df.columns:
            ai_columns.append('anomaly_reason')
        if 'cluster_id' in display_df.columns:
            ai_columns.append('cluster_id')
    
    qa_columns = []
    if 'is_qa_relevant' in display_df.columns:
        qa_columns.append('is_qa_relevant')
    if 'matched_qa_keywords' in display_df.columns:
        qa_columns.append('matched_qa_keywords')
    
    # Combine all available columns
    columns_to_display = base_columns + ai_columns + qa_columns
    columns_to_display = [col for col in columns_to_display if col in display_df.columns]
    
    st.dataframe(display_df[columns_to_display], use_container_width=True)
    
    # Show detailed view of selected issues
    if len(filtered_issues) > 0:
        st.subheader("Detailed Issue View")
        
        for i, issue in enumerate(filtered_issues[:10]):  # Show first 10
            with st.expander(f"{issue['timestamp']} - {issue['component']} - {issue['level']}"):
                st.write(f"**Timestamp:** {issue['timestamp']}")
                st.write(f"**Component:** {issue['component']}")
                st.write(f"**Level:** {issue['level']}")
                st.write(f"**Message:** {issue['message']}")
                st.write(f"**PID:** {issue['pid']}, **TID:** {issue['tid']}")
                
                if "ai_analysis" in result and issue.get('is_anomaly'):
                    st.warning(f"⚠️ **AI Anomaly Detected:** {issue.get('anomaly_reason', 'Unknown')}")
                
                if issue.get('is_qa_relevant'):
                    st.info(f"✓ **QA Relevant:** {issue.get('matched_qa_keywords', '')}")

def display_ai_insights():
    """Display AI-specific insights and findings"""
    if not any("ai_analysis" in result for result in st.session_state.analysis_results if "error" not in result):
        return
    
    st.header("🤖 AI Insights")
    
    # Collect all root causes
    all_root_causes = []
    
    for result in st.session_state.analysis_results:
        if "error" in result or "ai_analysis" not in result:
            continue
        if ("summary" in result["ai_analysis"] and 
            "root_cause_analysis" in result["ai_analysis"]["summary"] and 
            result["ai_analysis"]["summary"]["root_cause_analysis"] is not None and
            "root_causes" in result["ai_analysis"]["summary"]["root_cause_analysis"]):
            
            all_root_causes.extend(result["ai_analysis"]["summary"]["root_cause_analysis"]["root_causes"])
    
    if all_root_causes:
        st.subheader("Potential Root Causes")
        st.write("The AI has identified the following potential root causes:")
        
        for i, cause in enumerate(all_root_causes[:5]):
            with st.expander(f"Root Cause #{i+1}: {cause.get('potential_cause_component', 'Unknown')}"):
                st.write(f"**Component:** {cause.get('potential_cause_component', 'Unknown')}")
                st.write(f"**Preceding Message:** {cause.get('potential_cause_message', 'Unknown')}")
                st.write(f"**Followed By Error:** {cause.get('first_error', 'Unknown')}")
                st.write(f"**Timeframe:** {cause.get('timestamp_start', '')} to {cause.get('timestamp_end', '')}")
                st.write(f"**Error Count:** {cause.get('error_count', 0)}")

def display_utterance_analysis():
    """Display utterance flow analysis"""
    st.header("🎤 Utterance Flow Analysis")
    
    all_sessions = []
    
    for result in st.session_state.analysis_results:
        if "error" in result or "utterance_sessions" not in result:
            continue
        
        for session in result["utterance_sessions"]:
            session['log_file'] = result['log_file_name']
            all_sessions.append(session)
    
    if not all_sessions:
        st.info("No utterance sessions detected in the logs.")
        return
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Utterances", len(all_sessions))
    
    with col2:
        successful = len([s for s in all_sessions if not s['has_errors']])
        st.metric("Successful", successful)
    
    with col3:
        if len(all_sessions) > 0:
            # CRITICAL FIX #6: Safe division for percentage
            success_rate = (successful / len(all_sessions)) * 100 if len(all_sessions) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        if all_sessions:
            avg_duration = sum(s['duration_ms'] for s in all_sessions) / len(all_sessions) if len(all_sessions) > 0 else 0
            st.metric("Avg Duration (ms)", f"{avg_duration:.0f}")
    
    # Display utterances
    st.subheader("Utterance Sessions")
    
    for i, session in enumerate(all_sessions[:10]):  # Show first 10
        status = "✅ Success" if not session['has_errors'] else "❌ Failed"
        
        with st.expander(f"{status} - {session['utterance'][:50]} ({session['duration_ms']:.0f}ms) - {session['log_file']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Session ID:** {session['session_id']}")
                st.write(f"**Duration:** {session['duration_ms']:.2f} ms")
                st.write(f"**Entry Count:** {session['entries']}")
                st.write(f"**Status:** {status}")
            
            with col2:
                st.write(f"**Start Time:** {session['timestamp_start']}")
                st.write(f"**End Time:** {session['timestamp_end']}")
                st.write(f"**Final State:** {session['final_state']}")
                st.write(f"**Components:** {len(session['components'])}")
            
            st.write(f"**Components Involved:** {', '.join(session['components'][:5])}")
            if len(session['components']) > 5:
                st.write(f"... and {len(session['components']) - 5} more")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Automotive Log Analysis Platform (ALAP)",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Automotive Log Analysis Platform (ALAP)")
    
    st.markdown("""
    Upload, analyze, and generate reports for automotive log files. This platform handles large log files,
    filters issues by severity and relevance, and generates targeted reports for QA and Development teams.
    It also includes AI capabilities for anomaly detection and root cause analysis.
    """)
    
    # Settings in sidebar
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
    
    # File upload section
    st.header("Log File Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more log files",
        type=["log", "txt"],
        accept_multiple_files=True
    )
    
    # Analyze button
    if uploaded_files:
        if st.button("Analyze Logs"):
            perform_analysis(uploaded_files, use_ai=use_ai, enable_utterance_analysis=enable_utterance_analysis)
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.header("Analysis Results")
        
        # Visualizations
        visualize_analysis()
        
        # AI Insights
        if use_ai:
            display_ai_insights()
        
        # Utterance Analysis
        if enable_utterance_analysis:
            display_utterance_analysis()
        
        # Detailed Issues
        display_detailed_issues()
        
        # Report Generation
        st.header("Report Generation")
        
        if st.button("Generate Reports"):
            with st.spinner("Generating reports..."):
                csv_paths = generate_individual_csv_reports(st.session_state.analysis_results)
                qa_path, dev_path = generate_summary_reports_targetted(st.session_state.analysis_results)
                
                st.session_state.reports_generated = True
                
                # Display download buttons
                st.subheader("📥 Download Reports")
                
                if csv_paths:
                    st.write("### CSV Reports")
                    for path in csv_paths:
                        with open(path, "rb") as file:
                            st.download_button(
                                label=f"Download {path.name}",
                                data=file,
                                file_name=path.name,
                                mime="text/csv",
                                key=f"csv_{path.name}"
                            )
                
                if qa_path:
                    st.write("### QA Summary")
                    with open(qa_path, "rb") as file:
                        st.download_button(
                            label="Download QA Summary Report",
                            data=file,
                            file_name=qa_path.name,
                            mime="text/markdown",
                            key="qa_summary"
                        )
                
                if dev_path:
                    st.write("### Developer Summary")
                    with open(dev_path, "rb") as file:
                        st.download_button(
                            label="Download Developer Summary Report",
                            data=file,
                            file_name=dev_path.name,
                            mime="text/markdown",
                            key="dev_summary"
                        )
                
                st.success("Reports generated successfully!")

if __name__ == "__main__":
    main()