# Utterance Tracing Feature

## Overview

The Utterance Tracing feature helps testers and developers better trace and analyze voice assistant interactions in automotive log files. It automatically identifies, groups, and analyzes voice command flows from initiation through completion or failure.

## Features

### 1. Automatic Utterance Detection

The system automatically identifies voice assistant-related log entries based on:
- **Pattern matching**: Recognizes common voice assistant keywords (utterance, speech recognition, voice command, etc.)
- **Component filtering**: Identifies logs from voice-related components (SpeechRecognizer, AHE-NLU, TTS, etc.)
- **Event classification**: Categorizes logs into stages (start, processing, response, error)

### 2. Session/Request ID Extraction

Extracts unique identifiers from log messages to group related events:
- `session_id: abc123`
- `reqId: xyz789`
- `for voice_001`
- `[uuid-like-patterns]`

### 3. Flow Tracing

Groups log entries into complete utterance flows:
- **Completed flows**: Voice commands that successfully generated responses
- **Failed flows**: Commands that encountered errors
- **Incomplete flows**: Commands that started but have no clear completion

### 4. Visual Timeline

Creates easy-to-read timelines showing:
- Each event in the utterance lifecycle
- Component responsible for each step
- Duration of each stage
- Status indicators (🎤 start, ⚙️ processing, ✅ response, ❌ error)

### 5. Pattern Analysis

Provides aggregate statistics across all utterances:
- Total utterance count
- Success/failure rates
- Average processing duration
- Common failure components

## How It Works

### In Log Analysis

When logs are analyzed, the system:

1. **Enriches** each log entry with utterance metadata
2. **Groups** related entries by session ID
3. **Analyzes** the flow status (completed/failed/incomplete)
4. **Calculates** metrics like duration and error rates

### In the UI

The Streamlit interface shows:

- **Summary metrics**: Total flows, success rate, error rate
- **Visual charts**: Status distribution, duration analysis
- **Flow details**: Expandable sections for each utterance with timeline
- **Filtering options**: Filter by status (completed/failed/incomplete)

### In Reports

Both QA and Developer reports include:

#### QA Reports
- Utterance success rate overview
- List of failed utterances with session IDs
- Component involvement in failures
- Non-technical explanations

#### Developer Reports
- Detailed flow analysis with timings
- Component-level failure patterns
- Sample failed utterance timelines
- Technical diagnostics

## Usage

### Enabling Utterance Tracing

Utterance tracing is **enabled by default** when analyzing logs. It runs automatically alongside regular log analysis and AI features.

### Viewing Utterance Analysis

1. Upload log files through the Streamlit UI
2. Click "Analyze Logs"
3. Scroll to the "🎤 Utterance Flow Analysis" section
4. Explore individual flows by expanding them

### Filtering Logs

In the "Detailed Issue Analysis" section, you can:
- Check "Show Utterance-Related Only" to see only voice assistant logs
- Combine with other filters (QA relevant, AI anomalies)

### Generating Reports

Click "Generate Reports" to create:
- CSV files with all analyzed issues (including utterance metadata)
- QA Summary with utterance success rates
- Developer Summary with technical flow details

## Integration with AI Features

Utterance tracing works seamlessly with AI analysis:
- AI can detect anomalies in utterance-related logs
- Utterance events may be included in root cause analysis
- Both analyses run in parallel without conflicts

## Example Flow

```
Utterance Flow: voice_001
Status: COMPLETED
Duration: 1.889s
Components: SpeechRecognizer, TC, AHE-NLU

Timeline:
--------------------------------------------------------------------------------
🎤 [01-15 10:30:45.123] SpeechRecognizer
   utterance started, session_id: voice_001
   |
⚙️ [01-15 10:30:45.456] AHE-NLU
   Processing utterance for session_id: voice_001
   |
⚙️ [01-15 10:30:45.567] TC
   Intent recognition started for voice_001
   |
✅ [01-15 10:30:47.012] TTS
   Response generated for session_id: voice_001
--------------------------------------------------------------------------------
```

## Technical Details

### File Structure

- `utils/utterance_tracer.py`: Core tracing logic
- `utils/log_analyzer.py`: Integration with log analysis
- `utils/report_generator.py`: Report generation enhancements
- `app.py`: UI components

### Key Classes

- **UtteranceTracer**: Main class for utterance detection and tracing
  - `extract_session_id()`: Extract IDs from messages
  - `classify_utterance_event()`: Categorize event types
  - `enrich_with_utterance_data()`: Add metadata to log entries
  - `trace_utterance_flows()`: Group entries into flows
  - `analyze_utterance_patterns()`: Generate statistics

### Patterns Detected

The tracer recognizes these event types:

**Start Events**:
- "utterance started"
- "voice command received"
- "speech recognition started"
- "listening started"

**Processing Events**:
- "processing utterance"
- "NLU processing"
- "intent recognition"
- "entity extraction"

**Response Events**:
- "response generated"
- "TTS playing"
- "utterance completed"
- "command executed"

**Error Events**:
- "utterance failed"
- "speech recognition failed"
- "command failed"
- "timeout"

## Performance

Utterance tracing adds minimal overhead:
- Processes logs in a single pass
- Uses efficient regex pattern matching
- Scales with log file size
- No impact on non-voice-related logs

## Troubleshooting

### No utterances detected

If no utterances are found:
- Check if log format matches expected pattern
- Verify voice assistant components are logging
- Ensure session IDs are included in messages

### Incomplete flows

If many flows show as incomplete:
- Check if completion/response messages are present
- Verify session IDs are consistent across flow
- Consider timeout thresholds for grouping

### Inferred flows

Flows marked as "inferred" lack explicit session IDs but were grouped by:
- Temporal proximity (within 30 seconds)
- Component similarity
- Event sequence patterns

## Future Enhancements

Potential improvements:
- Custom pattern configuration
- Multi-turn conversation tracking
- Voice command intent analysis
- Performance benchmarking
- Historical trend analysis
