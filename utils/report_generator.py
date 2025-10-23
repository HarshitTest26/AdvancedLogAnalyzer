import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_report_directories():
    """Ensure that the reports directory exists."""
    Path("reports").mkdir(parents=True, exist_ok=True)
    logger.info("Ensured reports directory exists")

def generate_individual_csv_reports(analysis_results_list):
    """
    Generate individual CSV reports for each analyzed log file.
    
    Args:
        analysis_results_list (list): List of analysis results dictionaries
    
    Returns:
        list: List of paths to generated CSV reports
    """
    ensure_report_directories()
    csv_paths = []
    
    for result in analysis_results_list:
        if "error" in result:
            logger.error(f"Skipping CSV generation for error result: {result['error']}")
            continue
            
        try:
            log_name = Path(result['log_file_name']).stem
            csv_path = Path("reports") / f"{log_name}_issues_report.csv"
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(result['issue_entries'])
            df.to_csv(csv_path, index=False)
            
            csv_paths.append(csv_path)
            logger.info(f"Generated CSV report: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {str(e)}")
    
    return csv_paths

def generate_summary_reports_targetted(analysis_results_list):
    """
    Generate targeted summary reports for QA and Dev teams.
    
    Args:
        analysis_results_list (list): List of analysis results dictionaries
    
    Returns:
        tuple: Paths to the QA and Dev summary reports
    """
    ensure_report_directories()
    
    # Skip if no valid results
    if not analysis_results_list or all("error" in result for result in analysis_results_list):
        logger.error("No valid analysis results to generate summary reports")
        return None, None
    
    # Generate the reports
    qa_report_path = generate_qa_summary(analysis_results_list)
    dev_report_path = generate_dev_summary(analysis_results_list)
    
    return qa_report_path, dev_report_path

def generate_qa_summary(analysis_results_list):
    """
    Generate QA-focused summary report with AI insights and utterance analysis.
    
    Args:
        analysis_results_list (list): List of analysis results dictionaries
    
    Returns:
        Path: Path to the generated QA summary report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"qa_summary_report_{timestamp}.md"
    
    # Aggregate metrics for QA
    total_issues = sum(result.get('issues_found', 0) for result in analysis_results_list 
                       if "error" not in result)
    qa_relevant_issues = sum(result.get('qa_relevant_issues', 0) for result in analysis_results_list 
                            if "error" not in result)
    
    # Count anomalies if AI analysis was performed
    total_anomalies = 0
    for result in analysis_results_list:
        if "error" in result or "ai_analysis" not in result:
            continue
        if "summary" in result["ai_analysis"] and "anomaly_count" in result["ai_analysis"]["summary"]:
            total_anomalies += result["ai_analysis"]["summary"]["anomaly_count"]
    
    # Aggregate utterance metrics
    total_utterance_flows = 0
    failed_utterances = 0
    completed_utterances = 0
    for result in analysis_results_list:
        if "error" in result or "utterance_analysis" not in result:
            continue
        if "patterns" in result["utterance_analysis"]:
            patterns = result["utterance_analysis"]["patterns"]
            total_utterance_flows += patterns.get("total_flows", 0)
            failed_utterances += patterns.get("failed", 0)
            completed_utterances += patterns.get("completed", 0)
    
    # Aggregate component counts
    all_components = {}
    for result in analysis_results_list:
        if "error" in result:
            continue
        for comp, count in result.get('component_counts', {}).items():
            if comp in all_components:
                all_components[comp] += count
            else:
                all_components[comp] = count
    
    # Get top failing components
    top_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Build the QA report content
    qa_content = [
        "# QA Summary Report with AI Insights and Utterance Analysis",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Log Files Analyzed:** {len([r for r in analysis_results_list if 'error' not in r])}",
        "\n## Overview",
        f"**Total Issues Found:** {total_issues}",
        f"**QA Relevant Issues:** {qa_relevant_issues} ({round(qa_relevant_issues/total_issues*100 if total_issues else 0, 1)}%)",
    ]
    
    # Add utterance metrics if available
    if total_utterance_flows > 0:
        utterance_success_rate = (completed_utterances / total_utterance_flows * 100) if total_utterance_flows > 0 else 0
        qa_content.extend([
            f"**Voice Assistant Utterances:** {total_utterance_flows}",
            f"**Completed Utterances:** {completed_utterances}",
            f"**Failed Utterances:** {failed_utterances}",
            f"**Utterance Success Rate:** {round(utterance_success_rate, 1)}%",
        ])
    
    # Add AI insights if available
    if total_anomalies > 0:
        qa_content.extend([
            f"**Anomalies Detected by AI:** {total_anomalies}",
            "\n## AI Insights",
            "The AI analysis has identified anomalies that may require special attention. These anomalies represent unusual patterns or behaviors that deviate from normal operation."
        ])
        
        # Collect root causes across all files
        all_root_causes = []
        for result in analysis_results_list:
            if "error" in result or "ai_analysis" not in result:
                continue
            if ("summary" in result["ai_analysis"] and 
                "root_cause_analysis" in result["ai_analysis"]["summary"] and 
                result["ai_analysis"]["summary"]["root_cause_analysis"] is not None and
                "root_causes" in result["ai_analysis"]["summary"]["root_cause_analysis"]):
                
                all_root_causes.extend(result["ai_analysis"]["summary"]["root_cause_analysis"]["root_causes"])
        
        # Show top root causes
        if all_root_causes:
            qa_content.append("\n### Potential Root Causes")
            qa_content.append("AI has identified the following potential root causes of issues:")
            
            for i, cause in enumerate(all_root_causes[:5]):
                qa_content.append(f"\n**Root Cause #{i+1}:**")
                qa_content.append(f"- **Component:** {cause.get('potential_cause_component', 'Unknown')}")
                qa_content.append(f"- **Message:** {cause.get('potential_cause_message', 'Unknown')}")
                qa_content.append(f"- **Resulting Error:** {cause.get('first_error', 'Unknown')}")
                qa_content.append(f"- **Timestamp:** {cause.get('timestamp_start', 'Unknown')}")
    
    # Add utterance flow section if available
    if total_utterance_flows > 0:
        qa_content.append("\n## Voice Assistant Utterance Analysis")
        qa_content.append("This section provides insights into voice assistant interactions traced through the logs.")
        
        # Collect failed utterances for detailed reporting
        failed_utterance_details = []
        for result in analysis_results_list:
            if "error" in result or "utterance_analysis" not in result:
                continue
            if "flows" in result["utterance_analysis"]:
                for flow in result["utterance_analysis"]["flows"]:
                    if flow.get("status") == "failed":
                        failed_utterance_details.append({
                            'session_id': flow.get('session_id'),
                            'duration': flow.get('duration_seconds'),
                            'components': ', '.join(flow.get('components', [])),
                            'file': result['log_file_name']
                        })
        
        if failed_utterance_details:
            qa_content.append("\n### Failed Utterances")
            qa_content.append("The following voice commands failed to complete successfully:")
            qa_content.extend([
                "\n| Session ID | Duration (s) | Components Involved | Log File |",
                "| ---------- | -----------: | ------------------- | -------- |",
            ])
            
            for detail in failed_utterance_details[:10]:  # Limit to 10 for readability
                duration = f"{detail['duration']:.2f}" if detail['duration'] is not None else "N/A"
                qa_content.append(
                    f"| {detail['session_id']} | {duration} | {detail['components']} | {detail['file']} |"
                )
    
    # Standard top components section
    qa_content.append("\n## Top Failing Components")
    qa_content.extend([
        "| Component | Issue Count |",
        "| --------- | ----------: |",
    ])
    
    for comp, count in top_components:
        qa_content.append(f"| {comp} | {count} |")
    
    # Add file-specific summaries
    qa_content.append("\n## File Summaries")
    
    for result in analysis_results_list:
        if "error" in result:
            qa_content.append(f"### ❌ {result.get('log_file_name', 'Unknown file')}")
            qa_content.append(f"Error: {result['error']}")
            continue
            
        qa_content.append(f"### {result['log_file_name']}")
        qa_content.append(f"- **Total Issues:** {result['issues_found']}")
        qa_content.append(f"- **QA Relevant Issues:** {result['qa_relevant_issues']}")
        
        # Add utterance insights for this file if available
        if "utterance_analysis" in result and "patterns" in result["utterance_analysis"]:
            patterns = result["utterance_analysis"]["patterns"]
            if patterns.get("total_flows", 0) > 0:
                qa_content.append(f"- **Utterance Flows:** {patterns['total_flows']}")
                qa_content.append(f"  - Completed: {patterns.get('completed', 0)}")
                qa_content.append(f"  - Failed: {patterns.get('failed', 0)}")
                if patterns.get('avg_duration') is not None:
                    qa_content.append(f"  - Average Duration: {patterns['avg_duration']}s")
        
        # Add AI insights for this file if available
        if ("ai_analysis" in result and 
            "summary" in result["ai_analysis"] and 
            "anomaly_count" in result["ai_analysis"]["summary"]):
            
            anomaly_count = result["ai_analysis"]["summary"]["anomaly_count"]
            cluster_count = result["ai_analysis"]["summary"].get("cluster_count", 0)
            
            qa_content.append(f"- **AI Detected Anomalies:** {anomaly_count}")
            qa_content.append(f"- **Issue Pattern Groups:** {cluster_count}")
        
        # Add severity breakdown
        qa_content.append("- **Severity Breakdown:**")
        for level, count in result['level_counts'].items():
            if count > 0 and level in ['F', 'E', 'W']:
                qa_content.append(f"  - {level}: {count}")
    
    # Write the report
    report_path.write_text("\n".join(qa_content))
    logger.info(f"Generated QA summary report: {report_path}")
    
    return report_path

def generate_dev_summary(analysis_results_list):
    """
    Generate Developer-focused summary report with AI insights and utterance analysis.
    
    Args:
        analysis_results_list (list): List of analysis results dictionaries
    
    Returns:
        Path: Path to the generated Dev summary report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"dev_summary_report_{timestamp}.md"
    
    # Build the Dev report content
    dev_content = [
        "# Developer Technical Summary Report with AI Insights and Utterance Analysis",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Log Files Analyzed:** {len([r for r in analysis_results_list if 'error' not in r])}",
        "\n## Technical Overview"
    ]
    
    # Add AI Training Status if available
    for result in analysis_results_list:
        if "error" not in result and "ai_analysis" in result and "training_status" in result["ai_analysis"]:
            training_status = result["ai_analysis"]["training_status"]
            dev_content.append("\n### AI Model Status")
            dev_content.append(f"- **Model Trained:** {'Yes' if training_status.get('is_trained', False) else 'No'}")
            if training_status.get('last_updated'):
                dev_content.append(f"- **Last Updated:** {training_status.get('last_updated', 'Never')}")
            dev_content.append(f"- **Anomaly Detection:** {training_status.get('anomaly_model', 'Not initialized')}")
            dev_content.append(f"- **Text Processing:** {training_status.get('vectorizer', 'Not initialized')}")
            break
    
    # Add detailed per-file analysis
    for result in analysis_results_list:
        if "error" in result:
            dev_content.append(f"\n## ❌ {result.get('log_file_name', 'Unknown file')}")
            dev_content.append(f"Error: {result['error']}")
            continue
            
        dev_content.append(f"\n## {result['log_file_name']}")
        
        # Add utterance analysis section if available
        if "utterance_analysis" in result and "flows" in result["utterance_analysis"]:
            flows = result["utterance_analysis"]["flows"]
            patterns = result["utterance_analysis"].get("patterns", {})
            
            if patterns.get("total_flows", 0) > 0:
                dev_content.append("\n### Utterance Flow Analysis")
                dev_content.append(f"- **Total Utterance Flows:** {patterns['total_flows']}")
                dev_content.append(f"- **Completed:** {patterns.get('completed', 0)}")
                dev_content.append(f"- **Failed:** {patterns.get('failed', 0)}")
                dev_content.append(f"- **Incomplete:** {patterns.get('incomplete', 0)}")
                if patterns.get('avg_duration') is not None:
                    dev_content.append(f"- **Average Duration:** {patterns['avg_duration']}s")
                dev_content.append(f"- **Error Rate:** {patterns.get('error_rate', 0)}%")
                
                # Add common failure components
                if patterns.get('common_failure_components'):
                    dev_content.append("\n**Components Most Frequently Involved in Failed Utterances:**")
                    for comp_info in patterns['common_failure_components']:
                        dev_content.append(f"- {comp_info['component']}: {comp_info['count']} failures")
                
                # Add sample failed utterances
                failed_flows = [f for f in flows if f.get('status') == 'failed']
                if failed_flows:
                    dev_content.append("\n**Sample Failed Utterance Flows:**")
                    for i, flow in enumerate(failed_flows[:3], 1):  # Show up to 3 examples
                        dev_content.append(f"\n**Flow {i} - Session: {flow.get('session_id')}**")
                        dev_content.append(f"- Duration: {flow.get('duration_seconds', 'N/A')}s")
                        dev_content.append(f"- Components: {', '.join(flow.get('components', []))}")
                        dev_content.append(f"- Timeline Events: {flow.get('entry_count', 0)}")
        
        # Add AI insights section if available
        if "ai_analysis" in result and "summary" in result["ai_analysis"]:
            ai_summary = result["ai_analysis"]["summary"]
            
            dev_content.append("\n### AI Analysis Insights")
            dev_content.append(f"- **Anomalies Detected:** {ai_summary.get('anomaly_count', 0)}")
            dev_content.append(f"- **Pattern Groups Identified:** {ai_summary.get('cluster_count', 0)}")
            
            # Add root cause analysis if available
            if (ai_summary.get("root_cause_analysis") and 
                ai_summary["root_cause_analysis"] is not None and
                "root_causes" in ai_summary["root_cause_analysis"] and
                ai_summary["root_cause_analysis"]["root_causes"]):
                
                dev_content.append("\n#### Potential Root Causes")
                dev_content.append(f"Analysis Method: {ai_summary['root_cause_analysis'].get('analysis_method', 'Unknown')}")
                
                for i, cause in enumerate(ai_summary["root_cause_analysis"]["root_causes"]):
                    dev_content.append(f"\n**Sequence {i+1}:**")
                    dev_content.append(f"- **Component:** {cause.get('potential_cause_component', 'Unknown')}")
                    dev_content.append(f"- **Preceding Message:** `{cause.get('potential_cause_message', 'Unknown')}`")
                    dev_content.append(f"- **Followed By Error:** `{cause.get('first_error', 'Unknown')}`")
                    dev_content.append(f"- **Timeframe:** {cause.get('timestamp_start', '')} to {cause.get('timestamp_end', '')}")
                    dev_content.append(f"- **Error Count in Sequence:** {cause.get('error_count', 0)}")
        
        # Component stability section
        dev_content.append("\n### Component Stability Analysis")
        
        # Get the components with errors/warnings
        component_issues = {}
        for entry in result['issue_entries']:
            if entry['level'] in ['F', 'E', 'W']:
                comp = entry['component']
                if comp not in component_issues:
                    component_issues[comp] = {'F': 0, 'E': 0, 'W': 0}
                component_issues[comp][entry['level']] += 1
        
        # Create component stability table
        dev_content.extend([
            "| Component | Fatal | Error | Warning | Total |",
            "| --------- | ----: | ----: | ------: | ----: |"
        ])
        
        for comp, counts in sorted(component_issues.items(), 
                                  key=lambda x: (x[1]['F'], x[1]['E'], x[1]['W']), 
                                  reverse=True):
            total = counts['F'] + counts['E'] + counts['W']
            dev_content.append(
                f"| {comp} | {counts['F']} | {counts['E']} | {counts['W']} | {total} |"
            )
        
        # Process correlation section
        dev_content.append("\n### Process/Thread Correlation")
        
        # Find busiest processes (PIDs with most issues)
        pid_counts = {}
        for entry in result['issue_entries']:
            pid = entry['pid']
            if pid not in pid_counts:
                pid_counts[pid] = 0
            pid_counts[pid] += 1
        
        top_pids = sorted(pid_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        dev_content.append("**Top Active Processes:**")
        for pid, count in top_pids:
            dev_content.append(f"- PID {pid}: {count} issues")
        
        # Timeline reconstruction hints
        dev_content.append("\n### Timeline Reconstruction")
        dev_content.append("Key timestamps with clustered errors/warnings:")
        
        # Group issues by timestamp (first 16 chars of timestamp to group by minute)
        timestamp_clusters = {}
        for entry in result['issue_entries']:
            if entry['level'] in ['F', 'E', 'W']:
                ts = entry['timestamp'][:16]  # Group by minute
                if ts not in timestamp_clusters:
                    timestamp_clusters[ts] = []
                timestamp_clusters[ts].append(entry)
        
        # Find dense error clusters (times with multiple errors)
        dense_clusters = {ts: entries for ts, entries in timestamp_clusters.items() if len(entries) >= 3}
        
        # Show the densest error clusters
        for ts, entries in sorted(dense_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            error_count = sum(1 for e in entries if e['level'] == 'E')
            warning_count = sum(1 for e in entries if e['level'] == 'W')
            fatal_count = sum(1 for e in entries if e['level'] == 'F')
            
            dev_content.append(f"- **{ts}**: {len(entries)} issues ({fatal_count}F, {error_count}E, {warning_count}W)")
            # Add the top components involved
            comps = {}
            for e in entries:
                if e['component'] not in comps:
                    comps[e['component']] = 0
                comps[e['component']] += 1
            
            top_comps = sorted(comps.items(), key=lambda x: x[1], reverse=True)[:3]
            dev_content.append(f"  - Main components: {', '.join(f'{c} ({n})' for c, n in top_comps)}")
            
            # Add anomalies in this cluster if available
            anomalies = [e for e in entries if e.get('is_anomaly', False)]
            if anomalies:
                dev_content.append(f"  - Contains {len(anomalies)} anomalies detected by AI")
    
    # Write the report
    report_path.write_text("\n".join(dev_content))
    logger.info(f"Generated Dev summary report: {report_path}")
    
    return report_path