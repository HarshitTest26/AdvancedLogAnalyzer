import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """
    CRITICAL FIX #1: Add filename sanitization to prevent path traversal
    """
    import re
    # Remove any path separators and invalid characters
    filename = re.sub(r'[/\\:*?"<>|]', '_', filename)
    return filename

def ensure_report_directories():
    """Ensure that the reports directory exists."""
    try:
        Path("reports").mkdir(parents=True, exist_ok=True)
        logger.info("Ensured reports directory exists")
    except Exception as e:
        logger.error(f"Error creating reports directory: {str(e)}")

def generate_individual_csv_reports(analysis_results_list):
    """
    Generate individual CSV reports for each analyzed log file.
    """
    ensure_report_directories()
    csv_paths = []
    
    for result in analysis_results_list:
        if "error" in result:
            logger.error(f"Skipping CSV generation for error result: {result['error']}")
            continue
        
        try:
            # CRITICAL FIX #2: Implement secure path validation
            log_name = Path(result['log_file_name']).stem
            log_name = sanitize_filename(log_name)
            csv_path = Path("reports") / f"{log_name}_issues_report.csv"
            
            # Validate path is within reports directory
            csv_path = csv_path.resolve()
            if not str(csv_path).startswith(str(Path("reports").resolve())):
                logger.error(f"Invalid path detected: {csv_path}")
                continue
            
            # CRITICAL FIX #3: Add DataFrame validation
            if not result.get('issue_entries'):
                logger.warning(f"No issue entries for {log_name}")
                continue
            
            df = pd.DataFrame(result['issue_entries'])
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {log_name}")
                continue
            
            df.to_csv(csv_path, index=False)
            
            csv_paths.append(csv_path)
            logger.info(f"Generated CSV report: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {str(e)}")
    
    return csv_paths

def enhance_qa_summary(qa_content, analysis_results_list):
    """
    Enhances the QA summary report with utterance flow information.
    """
    all_sessions = []
    for result in analysis_results_list:
        if "error" in result or "utterance_sessions" not in result:
            continue
        
        for session in result["utterance_sessions"]:
            session['log_file'] = result['log_file_name']
            all_sessions.append(session)
    
    if not all_sessions:
        return qa_content
    
    qa_content.append("\n## Utterance Analysis")
    
    successful = len([s for s in all_sessions if not s['has_errors']])
    total = len(all_sessions)
    
    # CRITICAL FIX #4: Fix division by zero
    if total > 0:
        success_rate = (successful / total) * 100
    else:
        success_rate = 0
    
    qa_content.append(f"- **Total Utterances Detected:** {total}")
    qa_content.append(f"- **Successful Utterances:** {successful}")
    qa_content.append(f"- **Success Rate:** {success_rate:.1f}%")
    
    qa_content.append("\n### Utterance Status")
    qa_content.extend([
        "| Utterance | Status | Duration (ms) | Log File |",
        "| --------- | ------ | ------------: | -------- |",
    ])
    
    for session in all_sessions:
        status = "❌ Failed" if session['has_errors'] else "✅ Success"
        qa_content.append(f"| {session['utterance'][:50]} | {status} | {session['duration_ms']:.1f} | {session['log_file']} |")
    
    return qa_content

def enhance_dev_summary(dev_content, analysis_results_list):
    """
    Enhances the Developer summary report with detailed utterance flow information.
    """
    all_sessions = []
    for result in analysis_results_list:
        if "error" in result or "utterance_sessions" not in result:
            continue
        
        for session in result["utterance_sessions"]:
            session['log_file'] = result['log_file_name']
            all_sessions.append(session)
    
    if not all_sessions:
        return dev_content
    
    dev_content.append("\n## Utterance Flow Analysis")
    
    failed_sessions = [s for s in all_sessions if s['has_errors']]
    
    if failed_sessions:
        dev_content.append("\n### Failed Utterance Flows")
        
        for i, session in enumerate(failed_sessions):
            dev_content.append(f"\n#### {i+1}. \"{session['utterance']}\"")
            dev_content.append(f"- **Session ID:** {session['session_id']}")
            dev_content.append(f"- **Log File:** {session['log_file']}")
            dev_content.append(f"- **Duration:** {session['duration_ms']:.2f} ms")
            dev_content.append(f"- **Start Time:** {session['timestamp_start']}")
            dev_content.append(f"- **Components Involved:** {', '.join(session['components'])}")
    
    # Add component interaction analysis
    dev_content.append("\n### Component Interaction Analysis")
    
    component_error_count = {}
    
    for session in all_sessions:
        if session['has_errors']:
            for component in session['components']:
                if component not in component_error_count:
                    component_error_count[component] = 0
                component_error_count[component] += 1
    
    if component_error_count:
        sorted_components = sorted(component_error_count.items(), key=lambda x: x[1], reverse=True)
        
        dev_content.append("\n**Top Components in Failed Utterances:**")
        dev_content.extend([
            "| Component | Occurrence in Failed Utterances |",
            "| --------- | ------------------------------: |",
        ])
        
        for component, count in sorted_components[:10]:
            dev_content.append(f"| {component} | {count} |")
    
    return dev_content

def generate_summary_reports_targetted(analysis_results_list):
    """
    Generate targeted summary reports for QA and Dev teams.
    """
    ensure_report_directories()
    
    if not analysis_results_list or all("error" in result for result in analysis_results_list):
        logger.error("No valid analysis results to generate summary reports")
        return None, None
    
    qa_report_path = generate_qa_summary(analysis_results_list)
    dev_report_path = generate_dev_summary(analysis_results_list)
    
    return qa_report_path, dev_report_path

def generate_qa_summary(analysis_results_list):
    """
    Generate QA-focused summary report with AI insights.
    """
    report_path = Path("reports") / f"qa_summary_report.md"
    
    try:
        # Aggregate metrics for QA
        total_issues = sum(result.get('issues_found', 0) for result in analysis_results_list 
                          if "error" not in result)
        qa_relevant_issues = sum(result.get('qa_relevant_issues', 0) for result in analysis_results_list 
                                if "error" not in result)
        
        # CRITICAL FIX #4: Safe division
        if total_issues > 0:
            qa_percentage = round(qa_relevant_issues / total_issues * 100, 1)
        else:
            qa_percentage = 0
        
        # Count anomalies if AI analysis was performed
        total_anomalies = 0
        for result in analysis_results_list:
            if "error" in result or "ai_analysis" not in result:
                continue
            if "summary" in result["ai_analysis"] and "anomaly_count" in result["ai_analysis"]["summary"]:
                total_anomalies += result["ai_analysis"]["summary"]["anomaly_count"]
        
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
            "# QA Summary Report with AI Insights",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Log Files Analyzed:** {len([r for r in analysis_results_list if 'error' not in r])}",
            "\n## Overview",
            f"**Total Issues Found:** {total_issues}",
            f"**QA Relevant Issues:** {qa_relevant_issues} ({qa_percentage}%)",
        ]
        
        if total_anomalies > 0:
            qa_content.extend([
                f"**Anomalies Detected by AI:** {total_anomalies}",
                "\n## AI Insights",
                "The AI analysis has identified anomalies that may require special attention."
            ])
        
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
            
            if ("ai_analysis" in result and 
                "summary" in result["ai_analysis"] and 
                "anomaly_count" in result["ai_analysis"]["summary"]):
                
                anomaly_count = result["ai_analysis"]["summary"]["anomaly_count"]
                cluster_count = result["ai_analysis"]["summary"].get("cluster_count", 0)
                
                qa_content.append(f"- **AI Detected Anomalies:** {anomaly_count}")
                qa_content.append(f"- **Issue Pattern Groups:** {cluster_count}")
            
            qa_content.append("- **Severity Breakdown:**")
            for level, count in result['level_counts'].items():
                if count > 0 and level in ['F', 'E', 'W']:
                    qa_content.append(f"  - {level}: {count}")
        
        # Write the report
        report_path.write_text("\n".join(qa_content))
        logger.info(f"Generated QA summary report: {report_path}")
        
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating QA summary: {str(e)}")
        return None

def generate_dev_summary(analysis_results_list):
    """
    Generate Developer-focused summary report with AI insights.
    """
    report_path = Path("reports") / f"dev_summary_report.md"
    
    try:
        # Build the Dev report content
        dev_content = [
            "# Developer Technical Summary Report with AI Insights",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Log Files Analyzed:** {len([r for r in analysis_results_list if 'error' not in r])}",
            "\n## Technical Overview"
        ]
        
        # Add per-file analysis
        for result in analysis_results_list:
            if "error" in result:
                dev_content.append(f"\n## ❌ {result.get('log_file_name', 'Unknown file')}")
                dev_content.append(f"Error: {result['error']}")
                continue
            
            dev_content.append(f"\n## {result['log_file_name']}")
            
            # Add AI insights section if available
            if "ai_analysis" in result and "summary" in result["ai_analysis"]:
                ai_summary = result["ai_analysis"]["summary"]
                
                dev_content.append("\n### AI Analysis Insights")
                dev_content.append(f"- **Anomalies Detected:** {ai_summary.get('anomaly_count', 0)}")
                dev_content.append(f"- **Pattern Groups Identified:** {ai_summary.get('cluster_count', 0)}")
            
            # Component stability section
            dev_content.append("\n### Component Stability Analysis")
            
            component_issues = {}
            for entry in result['issue_entries']:
                if entry['level'] in ['F', 'E', 'W']:
                    comp = entry['component']
                    if comp not in component_issues:
                        component_issues[comp] = {'F': 0, 'E': 0, 'W': 0}
                    component_issues[comp][entry['level']] += 1
            
            if component_issues:
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
        
        # Write the report
        report_path.write_text("\n".join(dev_content))
        logger.info(f"Generated Dev summary report: {report_path}")
        
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating Dev summary: {str(e)}")
        return None