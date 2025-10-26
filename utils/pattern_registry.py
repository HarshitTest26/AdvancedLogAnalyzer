import json
import os

class PatternRegistry:
    def __init__(self):
        self.patterns = {
            "alexa": [],
            "google_assistant": [],
            "generic": []
        }
        self.dialog_state = {}
        
        # Load default patterns
        self.load_default_patterns()

    def load_default_patterns(self):
        # Default patterns for Alexa
        self.patterns["alexa"].extend([
            "Alexa, open the app",
            "Alexa, play music",
            "Alexa, tell me a joke"
        ])
        
        # Default patterns for Google Assistant
        self.patterns["google_assistant"].extend([
            "Hey Google, open the app",
            "Hey Google, play music",
            "Hey Google, tell me a joke"
        ])
        
        # Default generic voice commands
        self.patterns["generic"].extend([
            "Open the app",
            "Play music",
            "Tell me a joke"
        ])

    def load_custom_patterns(self, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' does not exist.")
        
        with open(config_file, 'r') as file:
            custom_patterns = json.load(file)
            for platform, patterns in custom_patterns.items():
                if platform in self.patterns:
                    self.patterns[platform].extend(patterns)

    def set_dialog_state(self, state):
        self.dialog_state = state

    def get_dialog_state(self):
        return self.dialog_state
