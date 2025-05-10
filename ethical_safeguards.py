# mental_health_journal/utils/ethical_safeguards.py
import re

class EthicalSafeguards:
    def __init__(self):
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "self-harm", 
            "hurt myself", "don't want to live", "want to die"
        ]
        self.crisis_resources = {
            "US": {
                "National Suicide Prevention Lifeline": "1-800-273-8255",
                "Crisis Text Line": "Text HOME to 741741"
            },
            "International": {
                "International Association for Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/"
            }
        }
    
    def check_for_crisis(self, text):
        """Check if the text contains crisis keywords"""
        text_lower = text.lower()
        for keyword in self.crisis_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return True
        return False
    
    def get_crisis_response(self):
        """Return crisis resources and a supportive message"""
        response = (
            "I've noticed some concerning content in your message that suggests you might be going "
            "through a difficult time. As an AI, I'm not able to provide the help you deserve right now. "
            "Please consider reaching out to a mental health professional or crisis service who can support you properly.\n\n"
            "Here are some resources that might help:\n\n"
        )
        
        for country, resources in self.crisis_resources.items():
            response += f"{country}:\n"
            for name, contact in resources.items():
                response += f"- {name}: {contact}\n"
            response += "\n"
        
        response += (
            "Remember, reaching out for help is a sign of strength, not weakness. "
            "You deserve support during difficult times."
        )
        
        return response