import json
import random
import logging
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapeuticRelationship:
    """Tracks the therapeutic relationship and alliance with the user over time"""
    
    def __init__(self):
        self.session_count = 0
        self.identified_themes = set()
        self.rapport_level = 0  # 0-5 scale
        self.treatment_focus = None
        self.preferred_approach = None
        self.coping_strategies_discussed = set()
    
    def update(self, user_input, emotion):
        """Update the therapeutic relationship based on new interaction"""
        self.session_count += 1
        
        # Increase rapport slightly with each interaction
        if self.rapport_level < 5:
            self.rapport_level += 0.2
        
        # Detect themes from the input
        themes = self._detect_themes(user_input)
        self.identified_themes.update(themes)
        
        # Update treatment focus if we have enough data
        if not self.treatment_focus and self.session_count >= 3:
            self._determine_treatment_focus()
            
        # Update preferred approach based on user responses
        if not self.preferred_approach and self.session_count >= 2:
            self._determine_preferred_approach(user_input, emotion)
        
        return {
            "themes": themes,
            "rapport_level": self.rapport_level,
            "treatment_focus": self.treatment_focus,
            "preferred_approach": self.preferred_approach
        }
    
    def _detect_themes(self, text):
        """Detect common therapeutic themes in text"""
        themes = set()
        theme_patterns = {
            "work_stress": [r"work", r"job", r"boss", r"career", r"office"],
            "relationships": [r"relationship", r"partner", r"friend", r"family", r"marriage"],
            "anxiety": [r"anxiety", r"worry", r"stress", r"nervous", r"overwhelm"],
            "depression": [r"sad", r"depress", r"unmotivated", r"hopeless", r"tired"],
            "self_esteem": [r"confidence", r"worth", r"not good enough", r"failure", r"inadequate"],
            "identity": [r"who I am", r"purpose", r"meaning", r"direction", r"authentic"],
            "growth": [r"improve", r"grow", r"develop", r"change", r"better"],
            "trauma": [r"trauma", r"abuse", r"ptsd", r"flashback", r"trigger"]
        }
        
        for theme, patterns in theme_patterns.items():
            if any(re.search(pattern, text.lower()) for pattern in patterns):
                themes.add(theme)
        
        return themes
    
    def _determine_treatment_focus(self):
        """Determine the primary therapeutic focus based on themes"""
        if not self.identified_themes:
            return
            
        # Count theme frequencies
        theme_counts = {}
        for theme in self.identified_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Set the most common theme as treatment focus
        if theme_counts:
            self.treatment_focus = max(theme_counts, key=theme_counts.get)
    
    def _determine_preferred_approach(self, user_input, emotion):
        """Determine what therapeutic approach seems to resonate with the user"""
        # This is a simplified method - in a real system, this would be more sophisticated
        text = user_input.lower()
        
        if any(word in text for word in ["think", "thought", "perspective", "reframe", "logic"]):
            self.preferred_approach = "cognitive"
        elif any(word in text for word in ["feel", "emotion", "sense", "heart", "gut"]):
            self.preferred_approach = "emotion_focused"
        elif any(word in text for word in ["do", "action", "change", "practice", "try"]):
            self.preferred_approach = "behavioral"
        else:
            # Default to person-centered approach
            self.preferred_approach = "person_centered"

class EmotionalTrajectory:
    """Tracks emotional patterns and trajectories over time"""
    
    def __init__(self):
        self.emotion_history = []
        self.positive_trend = 0
        self.emotional_volatility = 0
        self.dominant_emotion = None
        self.last_positive_emotion = None
        self.last_negative_emotion = None
    
    def update(self, emotion, confidence, timestamp=None):
        """Update the emotional trajectory with a new emotion"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Store the emotion
        self.emotion_history.append({
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": timestamp
        })
        
        # Update dominant emotion
        self._update_dominant_emotion()
        
        # Update positive/negative trend
        self._update_emotional_trend(emotion)
        
        # Update volatility
        if len(self.emotion_history) > 1:
            self._update_emotional_volatility()
        
        # Store latest emotions by category
        self._update_latest_emotions(emotion)
        
        return {
            "dominant_emotion": self.dominant_emotion,
            "positive_trend": self.positive_trend,
            "emotional_volatility": self.emotional_volatility
        }
    
    def _update_dominant_emotion(self):
        """Determine the most frequent emotion"""
        if not self.emotion_history:
            return
            
        # Count emotion frequencies
        emotion_counts = {}
        for entry in self.emotion_history:
            emotion = entry["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Set the most common emotion
        if emotion_counts:
            self.dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    def _update_emotional_trend(self, current_emotion):
        """Update the positive/negative emotional trend"""
        positive_emotions = [
            "joy", "admiration", "amusement", "excitement", "gratitude", 
            "love", "optimism", "pride", "approval", "caring", "desire", "relief"
        ]
        negative_emotions = [
            "sadness", "anger", "annoyance", "disapproval", "disgust", 
            "embarrassment", "fear", "grief", "nervousness", "disappointment", 
            "remorse", "confusion"
        ]
        
        # Adjust the trend based on the current emotion
        if current_emotion in positive_emotions:
            self.positive_trend += 0.2
        elif current_emotion in negative_emotions:
            self.positive_trend -= 0.1
        
        # Cap the trend between -1 and 1
        self.positive_trend = max(-1, min(1, self.positive_trend))
    
    def _update_emotional_volatility(self):
        """Update the emotional volatility measure"""
        # This is a simplified measure - a real system would use more sophisticated methods
        if len(self.emotion_history) < 2:
            return
            
        # Get the last two emotions
        last_emotion = self.emotion_history[-1]["emotion"]
        previous_emotion = self.emotion_history[-2]["emotion"]
        
        # Increase volatility if emotions are changing
        if last_emotion != previous_emotion:
            self.emotional_volatility += 0.2
        else:
            self.emotional_volatility -= 0.1
        
        # Cap volatility between 0 and 1
        self.emotional_volatility = max(0, min(1, self.emotional_volatility))
    
    def _update_latest_emotions(self, emotion):
        """Update the latest positive and negative emotions"""
        positive_emotions = [
            "joy", "admiration", "amusement", "excitement", "gratitude", 
            "love", "optimism", "pride", "approval", "caring", "desire", "relief"
        ]
        negative_emotions = [
            "sadness", "anger", "annoyance", "disapproval", "disgust", 
            "embarrassment", "fear", "grief", "nervousness", "disappointment", 
            "remorse", "confusion"
        ]
        
        if emotion in positive_emotions:
            self.last_positive_emotion = emotion
        elif emotion in negative_emotions:
            self.last_negative_emotion = emotion

class TherapeuticInterventions:
    """Contains therapeutic interventions and techniques"""
    
    def __init__(self):
        self._init_therapeutic_interventions()
    
    def _init_therapeutic_interventions(self):
        """Initialize all therapeutic interventions by approach and issue"""
        self.approaches = {
            "cognitive": {
                "name": "Cognitive Approach",
                "techniques": [
                    "thought_challenging",
                    "cognitive_reframing",
                    "identifying_cognitive_distortions",
                    "evidence_examination",
                    "perspective_taking"
                ]
            },
            "behavioral": {
                "name": "Behavioral Approach",
                "techniques": [
                    "behavioral_activation",
                    "exposure_therapy",
                    "skills_training",
                    "habit_formation",
                    "progressive_muscle_relaxation"
                ]
            },
            "emotion_focused": {
                "name": "Emotion-Focused Approach",
                "techniques": [
                    "emotion_awareness",
                    "emotion_regulation",
                    "self_compassion",
                    "emotional_processing",
                    "emotional_expression"
                ]
            },
            "person_centered": {
                "name": "Person-Centered Approach",
                "techniques": [
                    "unconditional_positive_regard",
                    "empathic_understanding",
                    "active_listening",
                    "authenticity",
                    "client_led_exploration"
                ]
            }
        }
        
        # Interventions by therapeutic approach and technique
        self.interventions = {
            # Cognitive interventions
            "thought_challenging": [
                "I'm curious about the thought '{identified_thought}'. What evidence do you have that supports this thought? And is there any evidence that might not support it?",
                "That thought '{identified_thought}' seems significant. If a good friend shared that same thought with you, what might you say to them?",
                "I notice the thought '{identified_thought}'. On a scale of 0-100%, how much do you believe this thought is completely true? What makes you give it that rating?"
            ],
            "cognitive_reframing": [
                "I'm wondering if there might be another way to look at this situation. What alternative perspectives might be possible here?",
                "Sometimes shifting how we frame a situation can change how we feel about it. Is there a different angle from which you could view what happened?",
                "You've described the situation as '{situation}'. I'm curious if there might be another interpretation that could also be true?"
            ],
            
            # Behavioral interventions
            "behavioral_activation": [
                "What activities have brought you a sense of enjoyment or accomplishment in the past? Even small steps toward those activities might be helpful.",
                "Sometimes when we're feeling down, it helps to schedule small, manageable activities. What's something small you might be able to do this week that could bring even a little satisfaction?",
                "I'm wondering what you could do in the next day or two that might give you a sense of either pleasure or accomplishment, even if it's very small?"
            ],
            "skills_training": [
                "It sounds like this situation was really challenging. What skills or strategies do you think might help you navigate similar situations in the future?",
                "Sometimes developing specific skills can help us manage difficult scenarios. What skill do you think might be most helpful to develop for situations like this?",
                "I'm wondering if practicing a specific approach for these situations might be helpful. What do you think you would need to learn or practice to handle this differently next time?"
            ],
            
            # Emotion-focused interventions
            "emotion_awareness": [
                "If you were to sit with this feeling for a moment, where do you notice it in your body? What physical sensations are associated with it?",
                "You've mentioned feeling {emotion}. I'm curious what other emotions might be present underneath or alongside that feeling?",
                "Sometimes our emotions have layers to them. As you reflect on how you're feeling, what do you notice about the different emotional responses you're having?"
            ],
            "self_compassion": [
                "This sounds really difficult. How might you speak to yourself with kindness about this situation, the way you would speak to someone you care about?",
                "I'm wondering how you might bring some gentleness or compassion toward yourself and what you're experiencing right now?",
                "Sometimes we're much harder on ourselves than we would be with others. What would it be like to offer yourself the same understanding you'd offer a good friend?"
            ],
            
            # Person-centered interventions
            "active_listening": [
                "I really hear how {emotion} you're feeling about {situation}. Tell me more about what this experience has been like for you.",
                "It sounds like this has had a significant impact on you. I'd like to understand more about what this means to you personally.",
                "I'm hearing that this situation has left you feeling {emotion}. What else comes up for you as you reflect on this experience?"
            ],
            "client_led_exploration": [
                "As you think about our conversation today, what feels most important for us to focus on?",
                "I'm wondering what you would find most helpful to explore further about this situation?",
                "What aspect of what we've been discussing feels most alive or pressing for you right now?"
            ]
        }
        
        # Intervention selection by issue
        self.issue_interventions = {
            "anxiety": ["emotion_awareness", "cognitive_reframing", "progressive_muscle_relaxation"],
            "depression": ["behavioral_activation", "cognitive_reframing", "self_compassion"],
            "self_esteem": ["thought_challenging", "self_compassion", "identifying_cognitive_distortions"],
            "relationships": ["perspective_taking", "emotional_expression", "client_led_exploration"],
            "work_stress": ["cognitive_reframing", "skills_training", "emotion_regulation"],
            "identity": ["client_led_exploration", "authenticity", "perspective_taking"],
            "growth": ["evidence_examination", "behavioral_activation", "client_led_exploration"],
            "trauma": ["emotion_awareness", "emotion_regulation", "self_compassion"]
        }
    
    def get_intervention(self, approach, issue=None, identified_thought=None, situation=None, emotion=None):
        """Get an appropriate therapeutic intervention based on approach and context"""
        # Select techniques based on approach and issue
        available_techniques = []
        
        if issue and issue in self.issue_interventions:
            # First try techniques that match both the approach and the issue
            approach_techniques = self.approaches[approach]["techniques"]
            issue_techniques = self.issue_interventions[issue]
            
            # Find the intersection of approach techniques and issue techniques
            matching_techniques = [t for t in approach_techniques if t in issue_techniques]
            
            if matching_techniques:
                available_techniques = matching_techniques
            else:
                # If no intersection, use issue-specific techniques
                available_techniques = issue_techniques
        else:
            # If no specific issue, use approach-specific techniques
            available_techniques = self.approaches[approach]["techniques"]
        
        # Randomly select a technique from available ones
        if available_techniques:
            technique = random.choice(available_techniques)
            
            # Get interventions for the selected technique
            interventions = self.interventions.get(technique, [])
            
            if interventions:
                # Select a random intervention
                intervention = random.choice(interventions)
                
                # Format the intervention with context if available
                context = {
                    "identified_thought": identified_thought or "that thought",
                    "situation": situation or "this situation",
                    "emotion": emotion or "that"
                }
                
                try:
                    return intervention.format(**context)
                except KeyError:
                    # If formatting fails, return the raw intervention
                    return intervention
        
        # Fallback to a generic person-centered response
        return random.choice(self.interventions["active_listening"])

class ResponseGenerator:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-3.5-turbo", 
        templates_path: str = "response_generator/prompt_templates.json"
    ):
        """
        Initialize the response generator with sophisticated therapeutic capabilities.
        """
        self.api_key = api_key

        # Emotion categories
        self.positive_emotions = [
            "joy", "admiration", "amusement", "excitement", "gratitude", 
            "love", "optimism", "pride", "approval", "caring", "desire", "relief"
        ]
        self.negative_emotions = [
            "sadness", "anger", "annoyance", "disapproval", "disgust", 
            "embarrassment", "fear", "grief", "nervousness", "disappointment", 
            "remorse", "confusion"
        ]
        self.neutral_emotions = ["surprise", "realization", "curiosity", "neutral"]
        
        # Initialize therapeutic components
        self.relationship = TherapeuticRelationship()
        self.emotional_trajectory = EmotionalTrajectory()
        self.interventions = TherapeuticInterventions()
        
        # Initialize response templates
        self._init_response_templates()
        
        # Track session data
        self.session_data = {
            "session_count": 0,
            "identified_thoughts": [],
            "identified_situations": [],
            "recurring_themes": [],
            "coping_strategies_suggested": []
        }
    
    def _get_emotion_category(self, emotion):
        """Categorize the emotion as positive, negative, or neutral"""
        emotion = emotion.lower()
        if emotion in self.positive_emotions:
            return "positive"
        elif emotion in self.negative_emotions:
            return "negative"
        else:
            return "neutral"
    
    def _init_response_templates(self):
        """Initialize therapeutic response templates"""
        # Validation statements
        self.validation_templates = [
            "I can hear that {situation} has been really {emotion_adjective} for you. {follow_up}",
            "What you're feeling makes a lot of sense given {situation}. {follow_up}",
            "It's completely understandable to feel {emotion} about {situation}. {follow_up}",
            "I appreciate you sharing how {emotion} you've been feeling. {follow_up}",
            "That sounds like a {emotion_adjective} experience. {follow_up}"
        ]
        
        # Reflection templates
        self.reflection_templates = [
            "It sounds like {reflection_content} {follow_up}",
            "I'm hearing that {reflection_content} {follow_up}",
            "From what you've shared, it seems that {reflection_content} {follow_up}",
            "What I'm understanding is that {reflection_content} {follow_up}",
            "It feels like {reflection_content} {follow_up}"
        ]
        
        # Open-ended question templates
        self.question_templates = [
            "I'm curious, {question}",
            "I wonder, {question}",
            "What are your thoughts about {question_topic}?",
            "How do you feel about {question_topic}?",
            "What would it be like if {hypothetical}?"
        ]
        
        # Therapeutic insight templates
        self.insight_templates = [
            "Sometimes {general_insight}. I wonder if that might be relevant to your experience?",
            "Many people find that {general_insight}. Does that resonate with you at all?",
            "One thing I've noticed is that {specific_insight}. What do you think about that?",
            "It's common for {general_insight}. Has that been true for you?",
            "I'm struck by {specific_insight}. How does that land for you?"
        ]
        
        # Transition templates for multi-component responses
        self.transition_templates = [
            "And",
            "Also",
            "Additionally",
            "At the same time",
            "What's more"
        ]
        
        # Emotion adjectives for validation statements
        self.emotion_adjectives = {
            "disappointment": "disappointing",
            "sadness": "difficult",
            "anger": "frustrating",
            "fear": "scary",
            "joy": "positive",
            "excitement": "exciting",
            "surprise": "unexpected",
            "confusion": "confusing",
            "default": "significant"
        }
    
    def _extract_situation(self, text):
        """Extract a situation description from the user's text"""
        # Simple extraction based on common patterns
        situation_patterns = [
            r"(?:had|having|dealing with|facing|experiencing) (.*)",
            r"(?:went through|going through) (.*)",
            r"(?:struggling with|stressed about) (.*)"
        ]
        
        for pattern in situation_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        # Default to the first part of their text
        words = text.split()
        if len(words) > 5:
            return " ".join(words[:5]) + "..."
        else:
            return text
    # Add this to your generate.py file (in the ResponseGenerator class)

    def _get_coping_strategies(self, emotion, situation=None):
        """
        Provide relevant coping strategies and practical suggestions based on emotion and situation
        
        Args:
            emotion: The detected emotion
            situation: Optional context about the situation
            
        Returns:
            A string containing relevant coping strategies and suggestions
        """
        # Work-related strategies
        work_strategies = {
            "anger": [
                "Taking a short break before responding when you feel criticized",
                "Writing down your thoughts before discussing them with your boss",
                "Practicing assertive communication using 'I' statements",
                "Setting clearer boundaries around work expectations"
            ],
            "annoyance": [
                "Taking a few deep breaths before responding when feeling criticized",
                "Trying to separate the feedback from your self-worth",
                "Asking for a follow-up conversation to clarify expectations",
                "Reflecting on whether there's anything constructive you can take from the feedback"
            ],
            "sadness": [
                "Scheduling short breaks throughout your workday",
                "Connecting with supportive colleagues",
                "Acknowledging your accomplishments, however small",
                "Considering whether a change in your work environment might be beneficial"
            ],
            "fear": [
                "Breaking down workplace challenges into smaller, manageable steps",
                "Preparing for difficult conversations in advance",
                "Seeking clarification about expectations to reduce uncertainty",
                "Finding a mentor who can provide guidance and support"
            ],
            "confusion": [
                "Asking clarifying questions when instructions aren't clear",
                "Documenting expectations and agreements in writing",
                "Scheduling regular check-ins with your supervisor",
                "Creating a list of questions before important meetings"
            ],
            "disappointment": [
                "Reevaluating your expectations to ensure they're realistic",
                "Focusing on aspects of your work that you can control",
                "Celebrating small wins and progress",
                "Speaking with your supervisor about career development opportunities"
            ]
        }
        
        # Relationship strategies
        relationship_strategies = {
            "anger": [
                "Taking a time-out when emotions are running high",
                "Using 'I' statements instead of accusatory language",
                "Focusing on the specific behavior rather than the person",
                "Practicing active listening to understand the other perspective"
            ],
            "sadness": [
                "Expressing your feelings openly with trusted individuals",
                "Setting healthy boundaries in relationships",
                "Engaging in activities that bring you joy and connection",
                "Practicing self-compassion during difficult interpersonal moments"
            ],
            "fear": [
                "Sharing your concerns with trusted individuals",
                "Taking small steps toward addressing relationship challenges",
                "Practicing vulnerability in safe relationships",
                "Distinguishing between past relationship patterns and current relationships"
            ],
            "confusion": [
                "Asking direct, non-confrontational questions for clarification",
                "Reflecting back what you've heard to ensure understanding",
                "Journaling about confusing interactions to identify patterns",
                "Setting aside time for important conversations when you're both calm"
            ]
        }
        
        # General emotional coping strategies
        general_strategies = {
            "anger": [
                "Practicing deep breathing exercises when you feel anger arising",
                "Physical activity to release tension",
                "Journaling about what triggered your anger",
                "Identifying and challenging unhelpful thoughts"
            ],
            "sadness": [
                "Allowing yourself to feel emotions without judgment",
                "Engaging in gentle self-care activities",
                "Connecting with supportive people",
                "Creating a daily routine that includes activities you enjoy"
            ],
            "fear": [
                "Grounding exercises like the 5-4-3-2-1 technique",
                "Progressive muscle relaxation to reduce physical tension",
                "Breaking overwhelming tasks into smaller steps",
                "Challenging catastrophic thinking with evidence-based alternatives"
            ],
            "confusion": [
                "Taking time to clarify your own thoughts through writing",
                "Breaking complex situations into simpler components",
                "Seeking additional information before making decisions",
                "Talking through your thoughts with someone you trust"
            ],
            "annoyance": [
                "Pausing before responding when irritated",
                "Considering the situation from multiple perspectives",
                "Setting appropriate boundaries",
                "Focusing on what you can control in the situation"
            ],
            "disappointment": [
                "Acknowledging your feelings without judgment",
                "Adjusting expectations when necessary",
                "Focusing on what you can learn from the experience",
                "Practicing gratitude for what is going well"
            ]
        }
        
        # Select appropriate strategies based on context and emotion
        selected_strategies = []
        
        # Work-related context
        if situation and any(word in situation.lower() for word in ["work", "boss", "job", "colleague", "office"]):
            if emotion in work_strategies:
                selected_strategies.extend(random.sample(work_strategies[emotion], min(2, len(work_strategies[emotion]))))
        
        # Relationship context
        elif situation and any(word in situation.lower() for word in ["friend", "partner", "relationship", "family", "parent"]):
            if emotion in relationship_strategies:
                selected_strategies.extend(random.sample(relationship_strategies[emotion], min(2, len(relationship_strategies[emotion]))))
        
        # Add general strategies for the emotion
        general_emotion = emotion
        # Map similar emotions to our strategy categories
        emotion_mapping = {
            "disappointment": "disappointment",
            "sadness": "sadness", 
            "grief": "sadness",
            "remorse": "sadness",
            "anger": "anger", 
            "annoyance": "annoyance",
            "disapproval": "annoyance", 
            "disgust": "anger",
            "fear": "fear", 
            "nervousness": "fear", 
            "anxiety": "fear",
            "confusion": "confusion", 
            "surprise": "confusion",
            "curiosity": "confusion"
        }
        
        if emotion in emotion_mapping:
            general_emotion = emotion_mapping[emotion]
        
        if general_emotion in general_strategies:
            additional_general = random.sample(general_strategies[general_emotion], min(2, len(general_strategies[general_emotion])))
            for strategy in additional_general:
                if strategy not in selected_strategies:
                    selected_strategies.append(strategy)
        
        # Format strategies into a cohesive paragraph
        if selected_strategies:
            strategies_text = "Here are some strategies that might help: "
            strategies_list = ", ".join([f"{i+1}) {strategy}" for i, strategy in enumerate(selected_strategies)])
            return strategies_text + strategies_list
        
        return ""



    def _extract_thought(self, text):
        """Extract a potential thought from the user's text"""
        # Look for thought patterns
        thought_patterns = [
            r"I think that (.*)",
            r"I believe (.*)",
            r"(?:I'm|I am) thinking (.*)",
            r"It seems like (.*)",
            r"I feel like (.*)"
        ]
        
        for pattern in thought_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _build_reflection(self, user_input, emotion):
        """Build a reflective statement based on the user's input"""
        # Extract content for reflection
        situation = self._extract_situation(user_input)
        # Make sure we have a valid situation
        if not situation or len(situation) < 3:
            # Extract basic situation from the input if none found
            words = user_input.split()
            if len(words) > 5:
                situation = " ".join(words[:5]) + "..."
            else:
                situation = user_input
        # Create reflection content based on the emotion
        emotion_category = self._get_emotion_category(emotion)
        
        if emotion_category == "negative":
            reflection_content = f"this {situation} has been really challenging and has left you feeling {emotion}"
        elif emotion_category == "positive":
            reflection_content = f"you're feeling {emotion} about {situation}, which is bringing you some positive feelings"
        else:
            reflection_content = f"you're processing your thoughts and feelings about {situation}"
        
        return reflection_content, situation
    
    def _build_validation(self, emotion, situation):
        """Build a validation statement based on emotion and situation"""
        # Get appropriate adjective for the emotion
        adjective = self.emotion_adjectives.get(emotion.lower(), self.emotion_adjectives["default"])
        
        # Create the follow-up part
        if self._get_emotion_category(emotion) == "negative":
            follow_up = "These feelings are a natural part of the human experience."
        else:
            follow_up = "It's important to recognize and appreciate these positive experiences."
        
        # Format the validation template
        template = random.choice(self.validation_templates)
        validation = template.format(
            emotion=emotion,
            situation=situation,
            emotion_adjective=adjective,
            follow_up=follow_up
        )
        
        return validation
    
    def _build_therapeutic_insight(self, emotion, reflection_content):
        """Build a therapeutic insight based on emotion and content"""
        # Generate general insight based on emotion category
        emotion_category = self._get_emotion_category(emotion)
        
        if emotion_category == "negative":
            general_insight = "difficult emotions can provide important information about our needs and values"
            specific_insight = f"how this experience of {emotion} seems connected to what matters to you"
        elif emotion_category == "positive":
            general_insight = "positive experiences can be anchors that help us navigate more challenging times"
            specific_insight = f"the connection between {reflection_content} and your sense of wellbeing"
        else:
            general_insight = "taking time to reflect can help us better understand our experiences"
            specific_insight = f"how you're making meaning of this experience with {reflection_content}"
        
        # Format the insight template
        template = random.choice(self.insight_templates)
        insight = template.format(
            general_insight=general_insight,
            specific_insight=specific_insight
        )
        
        return insight
    
    def _build_therapeutic_question(self, user_input, emotion, therapeutic_data):
        """Build a therapeutic question based on context and approach"""
        # Get the appropriate therapeutic approach
        approach = therapeutic_data.get("preferred_approach", "person_centered")
        issue = therapeutic_data.get("treatment_focus", None)
        
        # Extract thought and situation for interventions
        thought = self._extract_thought(user_input)
        situation = self._extract_situation(user_input)
        
        # Get an intervention appropriate for the approach and issue
        intervention = self.interventions.get_intervention(
            approach=approach, 
            issue=issue,
            identified_thought=thought,
            situation=situation,
            emotion=emotion
        )
        
        # Replace any placeholders with actual values
        intervention = intervention.replace("{emotion}", emotion)
        intervention = intervention.replace("{situation}", situation if situation else "this situation")
        intervention = intervention.replace("{identified_thought}", thought if thought else "that thought")
        
        # Use regex to find and replace any remaining template placeholders
        import re
        intervention = re.sub(r'\{[a-zA-Z_]+\}', '', intervention)
        
        return intervention
    
    def _build_relationship_element(self, therapeutic_data):
        """Build a statement that reflects the therapeutic relationship"""
        # Only add relationship elements after some rapport has been built
        if therapeutic_data.get("session_count", 0) < 2:
            return None
        
        if therapeutic_data.get("session_count", 0) == 2:
            return "I appreciate you continuing to share your experiences with me."
        
        rapport_level = therapeutic_data.get("rapport_level", 0)
        recurring_themes = therapeutic_data.get("recurring_themes", [])
        
        if rapport_level > 3 and recurring_themes:
            theme = random.choice(list(recurring_themes))
            return f"We've talked about {theme} a few times now, and I'm noticing how important this is in your life."
        
        if rapport_level > 2:
            return "I value the openness you've shown in our conversations."
        
        return None
    
    def _build_continuity_element(self, conversation_history):
        """Build a statement that reflects continuity with past conversations"""
        if not conversation_history or len(conversation_history) < 4:
            return None
        
        # Get previous AI messages
        ai_messages = [msg for msg in conversation_history if msg.get("role") == "assistant"]
        
        if not ai_messages:
            return None
        
        # Reference a previous conversation
        previous_msg = ai_messages[-1]
        previous_emotion = previous_msg.get("emotion", "")
        
        if previous_emotion and previous_emotion != "neutral":
            return f"Last time we spoke, you were feeling {previous_emotion}. I'm curious how things have evolved since then."
        
        return "I'm wondering how things have been for you since our last conversation."
    
    def _build_multi_component_response(self, components):
        """Build a cohesive response from multiple components"""
        # Filter out None components
        components = [c for c in components if c]
        
        if not components:
            return "I'm here to listen. How are you feeling today?"
        
        # Process each component to replace placeholders
        processed_components = []
        for component in components:
            # Skip non-string components (shouldn't happen, but just in case)
            if not isinstance(component, str):
                continue
                
            # Replace any lingering placeholders
            if '{emotion}' in component:
                component = component.replace('{emotion}', 'nervousness')
            if '{situation}' in component:
                component = component.replace('{situation}', 'this situation')
                
            # Use regex to find and replace any remaining template placeholders
            import re
            component = re.sub(r'\{[a-zA-Z_]+\}', '', component)
            processed_components.append(component)
        
        # Combine the first two components without a transition
        if len(processed_components) == 1:
            response = processed_components[0]
        else:
            response = f"{processed_components[0]} {processed_components[1]}"
            
            # Add remaining components with transitions
            for i in range(2, len(processed_components)):
                transition = random.choice(self.transition_templates)
                response += f" {transition}, {processed_components[i]}"
        
        # Final check for any missed placeholders
        import re
        response = re.sub(r'\{[a-zA-Z_]+\}', '', response)
        
        return response
    
    def generate_response(self, user_input, emotion, conversation_history=None):
        """
        Generate a sophisticated therapeutic response based on context and emotion.
        
        Args:
            user_input: The user's journal entry
            emotion: The detected emotion
            conversation_history: Optional history of previous exchanges
            
        Returns:
            A therapeutic response that mimics a human therapist
        """
        try:
            # Update therapeutic relationship and emotional trajectory
            self.session_data["session_count"] += 1
            
            relationship_data = self.relationship.update(user_input, emotion)
            emotion_data = self.emotional_trajectory.update(emotion, 0.8)
            
            # Combine all therapeutic data
            therapeutic_data = {
                "session_count": self.session_data["session_count"],
                "recurring_themes": relationship_data.get("themes", []),
                "dominant_emotion": emotion_data.get("dominant_emotion", ""),
                "positive_trend": emotion_data.get("positive_trend", 0),
                "emotional_volatility": emotion_data.get("emotional_volatility", 0),
                "preferred_approach": relationship_data.get("preferred_approach", "person_centered"),
                "treatment_focus": relationship_data.get("treatment_focus", None),
                "rapport_level": relationship_data.get("rapport_level", 0)
            }
            
            # Extract thought and situation
            thought = self._extract_thought(user_input)
            if thought:
                self.session_data["identified_thoughts"].append(thought)
            
            situation = self._extract_situation(user_input)
            self.session_data["identified_situations"].append(situation)
            
            # Build response components
            reflection_content, situation = self._build_reflection(user_input, emotion)
            
            validation = self._build_validation(emotion, situation)
            therapeutic_question = self._build_therapeutic_question(user_input, emotion, therapeutic_data)
            relationship_element = self._build_relationship_element(therapeutic_data)
            continuity_element = self._build_continuity_element(conversation_history)
            
            # Get appropriate coping strategies
            coping_strategies = self._get_coping_strategies(emotion, situation)
            
            # Determine which components to include based on context
            components = []
            
            # Always include continuity element first if available
            if continuity_element:
                components.append(continuity_element)
            
            # Always include validation 
            components.append(validation)
            
            # Include relationship element if available
            if relationship_element:
                components.append(relationship_element)
            
            # Include coping strategies after validation and relationship elements
            if coping_strategies:
                components.append(coping_strategies)
            
            # Always include a therapeutic question
            components.append(therapeutic_question)
            
            # Combine all components into a cohesive response
            response = self._build_multi_component_response(components)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating therapeutic response: {e}")
            return "I'm here to listen and support you. Could you tell me more about what you're experiencing?"

    
    # def generate_response(self, user_input, emotion, conversation_history=None):
    #     """
    #     Generate a sophisticated therapeutic response based on context and emotion.
        
    #     Args:
    #         user_input: The user's journal entry
    #         emotion: The detected emotion
    #         conversation_history: Optional history of previous exchanges
            
    #     Returns:
    #         A therapeutic response that mimics a human therapist
    #     """
    #     try:
    #         # Update therapeutic relationship and emotional trajectory
    #         self.session_data["session_count"] += 1
            
    #         relationship_data = self.relationship.update(user_input, emotion)
    #         emotion_data = self.emotional_trajectory.update(emotion, 0.8)
            
    #         # Combine all therapeutic data
    #         therapeutic_data = {
    #             "session_count": self.session_data["session_count"],
    #             "recurring_themes": relationship_data.get("themes", []),
    #             "dominant_emotion": emotion_data.get("dominant_emotion", ""),
    #             "positive_trend": emotion_data.get("positive_trend", 0),
    #             "emotional_volatility": emotion_data.get("emotional_volatility", 0),
    #             "preferred_approach": relationship_data.get("preferred_approach", "person_centered"),
    #             "treatment_focus": relationship_data.get("treatment_focus", None),
    #             "rapport_level": relationship_data.get("rapport_level", 0)
    #         }
            
    #         # Extract thought and situation
    #         thought = self._extract_thought(user_input)
    #         if thought:
    #             self.session_data["identified_thoughts"].append(thought)
            
    #         situation = self._extract_situation(user_input)
    #         self.session_data["identified_situations"].append(situation)
            
    #         # Build response components
    #         reflection_content, situation = self._build_reflection(user_input, emotion)
            
    #         validation = self._build_validation(emotion, situation)
    #         therapeutic_question = self._build_therapeutic_question(user_input, emotion, therapeutic_data)
    #         relationship_element = self._build_relationship_element(therapeutic_data)
    #         continuity_element = self._build_continuity_element(conversation_history)
            
    #         # Determine which components to include based on context
    #         components = []
            
    #         # Always include continuity element first if available
    #         if continuity_element:
    #             components.append(continuity_element)
            
    #         # Always include validation 
    #         components.append(validation)
            
    #         # Include relationship element if available
    #         if relationship_element:
    #             components.append(relationship_element)
            
    #         # Always include a therapeutic question
    #         components.append(therapeutic_question)
            
    #         # Combine all components into a cohesive response
    #         response = self._build_multi_component_response(components)
            
    #         return response
            
    #     except Exception as e:
    #         logger.error(f"Error generating therapeutic response: {e}")
    #         return "I'm here to listen and support you. Could you tell me more about what you're experiencing?"








# import json
# import random
# import logging
# import os
# import re
# from typing import Optional, List, Dict, Any

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ResponseGenerator:
#     def __init__(
#         self, 
#         api_key: Optional[str] = None, 
#         model: str = "gpt-3.5-turbo", 
#         templates_path: str = "response_generator/prompt_templates.json"
#     ):
#         """
#         Initialize the response generator with improved therapeutic techniques.
#         """
#         self.api_key = api_key

#         # T5 model setup - keep as is
#         try:
#             device = 0 if torch.cuda.is_available() else -1
#             logger.info("Loading local T5-small for on-device generation")
#             self.local_tok = AutoTokenizer.from_pretrained("t5-small")
#             self.local_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(
#                 "cuda" if torch.cuda.is_available() else "cpu"
#             )
#             self.local_pipe = pipeline(
#                 "text2text-generation",
#                 model=self.local_model,
#                 tokenizer=self.local_tok,
#                 device=device,
#                 do_sample=True,
#                 top_p=0.9,
#                 temperature=0.8
#             )
#         except Exception as e:
#             logger.warning(f"Failed to load local T5-small: {e}")
#             self.local_pipe = None

#         # Load templates
#         try:
#             with open(templates_path, 'r') as f:
#                 self.templates = json.load(f)
#             logger.info(f"Loaded templates from {templates_path}")
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             logger.warning(f"Error loading templates: {e}")
#             self.templates = self._default_templates()
#             try:
#                 os.makedirs(os.path.dirname(templates_path), exist_ok=True)
#                 with open(templates_path, 'w') as f:
#                     json.dump(self.templates, f, indent=2)
#                 logger.info(f"Created default templates at {templates_path}")
#             except Exception as e2:
#                 logger.warning(f"Could not save default templates: {e2}")

#         # Emotion categories
#         self.positive_emotions = [
#             "joy", "admiration", "amusement", "excitement", "gratitude", 
#             "love", "optimism", "pride", "approval", "caring", "desire", "relief"
#         ]
#         self.negative_emotions = [
#             "sadness", "anger", "annoyance", "disapproval", "disgust", 
#             "embarrassment", "fear", "grief", "nervousness", "disappointment", 
#             "remorse", "confusion"
#         ]
#         self.neutral_emotions = ["surprise", "realization", "curiosity", "neutral"]
        
#         # Common therapeutic topics and concerns
#         self.topics = {
#             "work": ["job", "work", "boss", "career", "office", "colleague", "coworker", "promotion"],
#             "relationships": ["friend", "partner", "spouse", "girlfriend", "boyfriend", "relationship", "dating"],
#             "family": ["family", "parent", "mother", "father", "child", "sibling", "brother", "sister"],
#             "health": ["health", "sick", "illness", "pain", "doctor", "sleep", "tired", "energy"],
#             "mental_health": ["stress", "anxiety", "depression", "overwhelm", "therapy", "worried", "panic"],
#             "self_esteem": ["failure", "inadequate", "confidence", "self-doubt", "insecure", "ashamed"]
#         }
        
#         # Initialize therapy approaches for different emotions
#         self.therapy_approaches = self._initialize_therapy_approaches()
        
#         # Initialize coping strategies
#         self.coping_strategies = self._initialize_coping_strategies()

#     def _default_templates(self):
#         """Default templates if none are found"""
#         return {
#             "joy": ["It's wonderful to hear you're feeling positive.", "That sounds like a great experience."],
#             "sadness": ["I'm sorry you're feeling down.", "That sounds really difficult."],
#             "anger": ["I can understand why you'd feel frustrated.", "That situation would be upsetting."],
#             "fear": ["It makes sense you'd feel anxious about that.", "That sounds concerning."],
#             "default": ["I'm here to listen.", "Thank you for sharing that with me."]
#         }

#     def _get_emotion_category(self, emotion):
#         """Categorize the emotion as positive, negative, or neutral"""
#         emotion = emotion.lower()
#         if emotion in self.positive_emotions:
#             return "positive"
#         elif emotion in self.negative_emotions:
#             return "negative"
#         else:
#             return "neutral"
    
#     def _initialize_therapy_approaches(self):
#         """Set up different therapeutic approaches based on emotions"""
#         return {
#             # Anxiety-related emotions use CBT approaches
#             "fear": "cbt",
#             "nervousness": "cbt",
#             "confusion": "cbt",
            
#             # Depression-related emotions use compassion-focused approaches
#             "sadness": "compassion",
#             "grief": "compassion",
#             "disappointment": "compassion",
#             "remorse": "compassion",
            
#             # Anger-related emotions use DBT approaches
#             "anger": "dbt",
#             "annoyance": "dbt",
#             "disapproval": "dbt",
#             "disgust": "dbt",
            
#             # Default to person-centered for other emotions
#             "default": "person_centered"
#         }
    
#     def _initialize_coping_strategies(self):
#         """Set up coping strategies for different emotions"""
#         return {
#             # Anxiety-related strategies
#             "fear": [
#                 "Taking a few deep breaths can help activate your parasympathetic nervous system and reduce anxiety.",
#                 "The 5-4-3-2-1 grounding technique might help: identify 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste.",
#                 "Writing down your worries can sometimes help externalize them and reduce their power."
#             ],
#             "nervousness": [
#                 "Progressive muscle relaxation can help release physical tension associated with nervousness.",
#                 "Setting aside a specific 'worry time' each day can help contain anxious thoughts.",
#                 "Focusing on what's in your control rather than what isn't may help reduce nervous feelings."
#             ],
            
#             # Depression-related strategies
#             "sadness": [
#                 "Even small amounts of physical activity can help lift your mood through endorphin release.",
#                 "Setting very small, achievable goals might help provide a sense of accomplishment.",
#                 "Connecting with others, even briefly, can sometimes reduce feelings of isolation when you're sad."
#             ],
#             "disappointment": [
#                 "Acknowledging your disappointment without judgment is an important first step.",
#                 "Looking for lessons or growth opportunities might help reframe the situation.",
#                 "Practicing self-compassion is especially important when dealing with disappointment."
#             ],
            
#             # Anger-related strategies
#             "anger": [
#                 "Taking a brief 'time-out' can help prevent saying or doing things you might regret.",
#                 "Physical activities like walking or exercise can help process the physiological aspects of anger.",
#                 "Identifying the underlying needs behind your anger might help address the root cause."
#             ],
#             "annoyance": [
#                 "Noticing where you feel irritation in your body can help create distance from the feeling.",
#                 "Checking whether your expectations are realistic can sometimes help with feelings of annoyance.",
#                 "Brief mindfulness practices can help create space between triggers and reactions."
#             ],
            
#             # General strategies for multiple emotions
#             "general": [
#                 "Reaching out to someone you trust can provide both emotional support and perspective.",
#                 "Taking care of basic needs like sleep, nutrition, and movement forms the foundation of emotional resilience.",
#                 "Self-compassion practices can be helpful across many different emotional challenges.",
#                 "Mindfulness meditation can help develop awareness of emotions without becoming overwhelmed by them."
#             ]
#         }

#     def _get_therapeutic_approach(self, emotion):
#         """Get the appropriate therapeutic approach for the emotion"""
#         emotion = emotion.lower()
#         return self.therapy_approaches.get(emotion, self.therapy_approaches["default"])
    
#     def _get_coping_strategy(self, emotion):
#         """Get an appropriate coping strategy for the emotion"""
#         emotion = emotion.lower()
#         strategies = self.coping_strategies.get(emotion, self.coping_strategies.get("general", []))
#         return random.choice(strategies) if strategies else None
    
#     def _detect_topics(self, text):
#         """Detect relevant therapeutic topics in the user's input"""
#         text = text.lower()
#         detected_topics = []
        
#         for topic, keywords in self.topics.items():
#             if any(keyword in text for keyword in keywords):
#                 detected_topics.append(topic)
        
#         return detected_topics
    
#     def _generate_validation(self, user_input, emotion):
#         """Generate a validating statement that acknowledges the user's feelings"""
#         validations = [
#             f"It sounds like you're feeling {emotion}, which is completely understandable.",
#             f"I can see how that experience would lead to feelings of {emotion}.",
#             f"Those feelings of {emotion} make a lot of sense given what you've shared.",
#             f"It's natural to feel {emotion} in a situation like this.",
#             f"Your {emotion} is a valid response to what you're experiencing."
#         ]
#         return random.choice(validations)
    
#     def _generate_reflection(self, user_input, emotion):
#         """Generate a reflective statement that mirrors back content from the user's input"""
#         # Extract potential subjects from input
#         hectic_day_patterns = [
#             "I hear that your day has been hectic, which can be draining.",
#             "A hectic day like you've described can certainly be overwhelming.",
#             "Dealing with a hectic day can be really challenging.",
#             "Hectic days like this can take a lot out of you."
#         ]
        
#         # Check for common patterns in the input
#         if re.search(r'(hectic|busy|chaotic|crazy|stressful)\s+day', user_input.lower()):
#             return random.choice(hectic_day_patterns)
        
#         # Generic reflections based on emotion
#         reflections = [
#             f"From what you've shared, it seems like you're navigating some challenging experiences.",
#             f"It sounds like there's a lot happening for you right now.",
#             f"I'm hearing that today has brought some difficulties your way.",
#             f"It seems like you're processing some complex feelings right now."
#         ]
#         return random.choice(reflections)
    
#     def _generate_normalization(self, emotion):
#         """Generate a normalizing statement that helps the user see their reaction as normal"""
#         normalizations = {
#             "disappointment": [
#                 "Disappointment is a natural response when things don't go as we hoped or expected.",
#                 "Many people experience disappointment when faced with setbacks or unmet expectations.",
#                 "Feeling disappointed is a common human experience that shows you care about outcomes."
#             ],
#             "fear": [
#                 "Fear is our mind's way of trying to protect us from perceived threats.",
#                 "Feeling afraid or anxious is something everyone experiences at different points in their lives.",
#                 "Fear is a natural response that's deeply wired into our nervous systems."
#             ],
#             "sadness": [
#                 "Sadness is a natural part of the human emotional spectrum that everyone experiences.",
#                 "Feeling sad is a normal response to loss or difficult situations.",
#                 "Sadness, though uncomfortable, is a healthy emotion that helps us process difficult experiences."
#             ],
#             "anger": [
#                 "Anger often serves as a signal that something important to us has been threatened.",
#                 "Feeling angry is a normal human emotion that everyone experiences.",
#                 "Anger can be a natural response to perceived injustice or boundary violations."
#             ],
#             "default": [
#                 "What you're feeling is part of the normal human emotional experience.",
#                 "These emotions, though challenging, are a natural part of being human.",
#                 "Many people have similar reactions when facing these kinds of situations."
#             ]
#         }
        
#         emotion_norms = normalizations.get(emotion.lower(), normalizations["default"])
#         return random.choice(emotion_norms)
    
#     def _generate_perspective(self, user_input, emotion):
#         """Generate a perspective-offering statement"""
#         perspectives = {
#             "positive": [
#                 "Recognizing positive moments like this can help build emotional resilience.",
#                 "Being aware of what brings you joy can help guide future choices.",
#                 "Taking time to appreciate positive experiences helps reinforce them in our memory."
#             ],
#             "negative": [
#                 "Even difficult emotions like this can provide important information about our needs.",
#                 "Challenging times, while difficult, often contain opportunities for growth and learning.",
#                 "Sometimes our most difficult experiences eventually lead to meaningful insights."
#             ],
#             "neutral": [
#                 "Taking time to reflect on your experiences can help build self-awareness.",
#                 "Being curious about your own reactions can lead to valuable self-insight.",
#                 "Moments of reflection like this are important for processing your experiences."
#             ]
#         }
        
#         emotion_category = self._get_emotion_category(emotion)
#         return random.choice(perspectives[emotion_category])
    
#     def _generate_exploration_question(self, user_input, emotion):
#         """Generate an open-ended question to explore the user's experience"""
#         # Questions based on detected topics
#         topics = self._detect_topics(user_input)
#         topic_questions = {
#             "work": [
#                 "How has this situation at work been affecting other areas of your life?",
#                 "What aspects of your work environment feel most challenging right now?",
#                 "What would help make your work situation feel more manageable?"
#             ],
#             "relationships": [
#                 "How do you think this is affecting your connection with that person?",
#                 "What would you like to see change in this relationship?",
#                 "What patterns have you noticed in how you respond in this relationship?"
#             ],
#             "family": [
#                 "How do these family dynamics affect how you feel about yourself?",
#                 "What boundaries might be helpful in this family situation?",
#                 "How do past family experiences influence how you're feeling now?"
#             ],
#             "health": [
#                 "How has this health concern been impacting your daily life?",
#                 "What would help you feel more supported with these health challenges?",
#                 "What strategies have helped you cope with health issues in the past?"
#             ],
#             "mental_health": [
#                 "What helps you feel more grounded when these feelings arise?",
#                 "How do these feelings typically show up for you?",
#                 "What patterns have you noticed around when these feelings intensify?"
#             ],
#             "self_esteem": [
#                 "How do these thoughts about yourself affect the choices you make?",
#                 "When did you first start having these kinds of thoughts about yourself?",
#                 "What would it be like to view yourself with more compassion in these moments?"
#             ]
#         }
        
#         # If we detected relevant topics, use a topic-specific question
#         if topics:
#             topic = random.choice(topics)
#             if topic in topic_questions:
#                 return random.choice(topic_questions[topic])
        
#         # Otherwise use general questions based on emotion category
#         emotion_category = self._get_emotion_category(emotion)
#         general_questions = {
#             "positive": [
#                 "What helped create this positive experience for you?",
#                 "How might you bring more of these positive feelings into other areas of your life?",
#                 "What does this positive experience tell you about what matters to you?"
#             ],
#             "negative": [
#                 "What do you think might help you navigate this difficult situation?",
#                 "What kind of support would be most helpful for you right now?",
#                 "How have you coped with similar feelings in the past?"
#             ],
#             "neutral": [
#                 "What thoughts have been on your mind about this situation?",
#                 "What would be most helpful for us to explore about this?",
#                 "What aspects of this experience feel most important to you right now?"
#             ]
#         }
        
#         return random.choice(general_questions[emotion_category])
    
#     def _generate_coping_question(self, emotion):
#         """Generate a question about coping strategies"""
#         strategy = self._get_coping_strategy(emotion)
        
#         if strategy:
#             questions = [
#                 f"{strategy} Have you ever tried something like that before?",
#                 f"I wonder if {strategy.lower()} What are your thoughts about trying that?",
#                 f"Some people find that {strategy.lower()} Would something like that be helpful for you?"
#             ]
#             return random.choice(questions)
        
#         # Fallback to generic coping questions
#         generic_questions = [
#             "What kinds of things have helped you feel better in similar situations?",
#             "What resources or support might help you navigate this challenge?",
#             "What self-care practices tend to work best for you during difficult times?"
#         ]
#         return random.choice(generic_questions)

#     def generate_response(self, user_input, emotion, conversation_history=None):
#         """
#         Generate a response using enhanced therapeutic techniques.
        
#         Args:
#             user_input: The user's journal entry
#             emotion: The detected emotion
#             conversation_history: Optional history of previous exchanges
        
#         Returns:
#             A therapeutic response
#         """
#         # First try T5 model with improved prompt
#         if self.local_pipe is not None:
#             try:
#                 # Enhanced therapeutic prompt
#                 prompt = (
#                     "As an empathetic therapist using evidence-based techniques, respond to this client statement: "
#                     f"'{user_input}' (Emotion detected: {emotion})\n\n"
#                     "In your 3-4 sentence response:\n"
#                     "1. Validate their feelings\n"
#                     "2. Offer perspective\n"
#                     "3. Ask one open-ended question\n"
#                     "Respond in a warm, conversational tone."
#                 )
                
#                 out = self.local_pipe(
#                     prompt, max_length=150, num_return_sequences=1
#                 )
#                 text = out[0]["generated_text"].strip()
#                 if text and len(text) > 20:  # Make sure we got a meaningful response
#                     return text
#             except Exception as e:
#                 logger.warning(f"Local T5 generation error: {e}")
        
#         # Fall back to enhanced template-based response
#         try:
#             logger.info(f"Generating enhanced therapeutic response for emotion: {emotion}")
            
#             # Get therapeutic approach
#             approach = self._get_therapeutic_approach(emotion)
#             logger.info(f"Using therapeutic approach: {approach}")
            
#             # Build response components
#             components = []
            
#             # Always start with validation or reflection
#             if self._get_emotion_category(emotion) == "negative":
#                 components.append(self._generate_validation(user_input, emotion))
#                 # Add normalization for negative emotions
#                 components.append(self._generate_normalization(emotion))
#             else:
#                 components.append(self._generate_reflection(user_input, emotion))
            
#             # Add perspective
#             components.append(self._generate_perspective(user_input, emotion))
            
#             # End with an appropriate question
#             if self._get_emotion_category(emotion) == "negative":
#                 components.append(self._generate_coping_question(emotion))
#             else:
#                 components.append(self._generate_exploration_question(user_input, emotion))
            
#             # Check for emotional shifts in conversation history
#             if conversation_history and len(conversation_history) >= 4:
#                 # Get the previous user message and AI response
#                 prev_user_msg = conversation_history[-4]
#                 prev_ai_msg = conversation_history[-3]
                
#                 # Check if there was an emotion shift
#                 if "emotion" in prev_ai_msg and prev_ai_msg["emotion"] != emotion:
#                     shift_note = f"I notice your feelings seem to have shifted from {prev_ai_msg['emotion']} to {emotion}."
#                     components.insert(1, shift_note)  # Insert after validation/reflection
            
#             # Combine components
#             response = " ".join(components)
#             return response
            
#         except Exception as e:
#             logger.error(f"Error generating therapeutic response: {e}")
#             return "I'm here to listen. Could you tell me more about how you're feeling and what's on your mind?"



# # # response_generator/generate.py
# # import json
# # import random
# # import logging
# # import os
# # from typing import Dict, List, Optional

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # class ResponseGenerator:
# #     def __init__(
# #         self, 
# #         api_key: Optional[str] = None, 
# #         model: str = "gpt-3.5-turbo", 
# #         templates_path: str = "response_generator/prompt_templates.json"
# #     ):
# #         """
# #         Initialize the response generator with templates
# #         """
# #         # Store API key (not used)
# #         self.api_key = api_key
        
# #         # Load prompt templates
# #         try:
# #             with open(templates_path, 'r') as f:
# #                 self.templates = json.load(f)
# #             logger.info(f"Loaded templates from {templates_path}")
# #         except (FileNotFoundError, json.JSONDecodeError) as e:
# #             logger.warning(f"Error loading templates: {e}")
# #             # Fallback to default templates
# #             self.templates = self._default_templates()
            
# #             # Try to create the directory and save default templates
# #             try:
# #                 os.makedirs(os.path.dirname(templates_path), exist_ok=True)
# #                 with open(templates_path, 'w') as f:
# #                     json.dump(self.templates, f, indent=2)
# #                 logger.info(f"Created default templates file at {templates_path}")
# #             except Exception as e:
# #                 logger.warning(f"Could not save default templates: {e}")
        
# #         # Define emotion categories
# #         self.positive_emotions = [
# #             "joy", "admiration", "amusement", "excitement", "gratitude", 
# #             "love", "optimism", "pride", "approval", "caring", "desire", "relief"
# #         ]
        
# #         self.negative_emotions = [
# #             "sadness", "anger", "annoyance", "disapproval", "disgust", 
# #             "embarrassment", "fear", "grief", "nervousness", "disappointment", 
# #             "remorse", "confusion"
# #         ]
        
# #         # Define neutral emotions
# #         self.neutral_emotions = [
# #             "surprise", "realization", "curiosity", "neutral"
# #         ]
    
# #     def _default_templates(self):
# #         """
# #         Default templates in case the template file is not found
# #         """
# #         return {
# #             "joy": [
# #                 "It's wonderful to hear you're feeling positive! What specific moments are bringing you joy?",
# #                 "I'm glad to hear you're experiencing joy. These positive emotions are valuable to recognize and savor.",
# #                 "Your happiness comes through clearly. What do you think contributed most to these positive feelings?"
# #             ],
# #             "admiration": [
# #                 "It's wonderful to hear about your accomplishment! What does this achievement mean to you?",
# #                 "I can tell you're feeling proud of this recognition. Moments like these are worth celebrating!",
# #                 "That's fantastic news about your achievement! How does this success make you feel about yourself?"
# #             ],
# #             "sadness": [
# #                 "I notice you're feeling down. It takes courage to acknowledge these feelings, and I'm here to listen.",
# #                 "I can hear that you're experiencing sadness. Would you like to explore what might be behind these feelings?",
# #                 "When you say you're feeling sad, I wonder if there are specific situations or thoughts contributing to this emotion."
# #             ],
# #             "anger": [
# #                 "I can sense your frustration. Anger often signals that something important to us has been affected.",
# #                 "Your feelings of anger are valid. Sometimes exploring what's underneath can help us understand our reactions better.",
# #                 "I notice you're feeling angry. Would it help to discuss what triggered these emotions?"
# #             ],
# #             "fear": [
# #                 "I hear that you're feeling anxious or afraid. These feelings often come up when facing uncertainty.",
# #                 "Your concerns are understandable. Fear is a natural response when we perceive threats to things we value.",
# #                 "It sounds like you're experiencing some worry or anxiety. Would you like to explore ways to work with these feelings?"
# #             ],
# #             "default": [
# #                 "Thank you for sharing your thoughts with me. I'm interested in understanding more about your experience.",
# #                 "I appreciate you opening up. Would you like to explore these feelings further?",
# #                 "I'm here to listen and support you. Is there a particular aspect of this you'd like to focus on?"
# #             ]
# #         }
    
# #     def _get_emotion_category(self, emotion):
# #         """Determine if the emotion is positive, negative, or neutral"""
# #         emotion = emotion.lower()
# #         if emotion in self.positive_emotions:
# #             return "positive"
# #         elif emotion in self.negative_emotions:
# #             return "negative"
# #         else:
# #             return "neutral"
    
# #     def _celebration_response(self, user_input, emotion):
# #         """Generate responses celebrating positive emotions and achievements"""
# #         celebrations = [
# #             "That's fantastic news! Congratulations on your achievement.",
# #             "What a wonderful accomplishment! You must be feeling proud.",
# #             "That's definitely something to celebrate! Well done!",
# #             "This is great news! It's wonderful to hear about your success.",
# #             "Congratulations on your achievement! That's really impressive."
# #         ]
        
# #         # Add specific responses for achievements like promotion
# #         if "promotion" in user_input.lower():
# #             celebrations.extend([
# #                 "Congratulations on your promotion! That's a significant career achievement.",
# #                 "Getting a promotion is a wonderful recognition of your hard work and abilities. Congratulations!",
# #                 "That's fantastic news about your promotion! Your skills and dedication have been recognized."
# #             ])
        
# #         # Add specific responses for praise or recognition
# #         if "praise" in user_input.lower() or "praised" in user_input.lower():
# #             celebrations.extend([
# #                 "Being praised feels great! It's wonderful to have your efforts recognized.",
# #                 "Receiving praise is a wonderful affirmation of your hard work.",
# #                 "It's always meaningful when others recognize our efforts and contributions."
# #             ])
        
# #         return random.choice(celebrations)
    
# #     def _reflection_response(self, user_input, emotion, emotion_category):
# #         """Generate reflective responses based on emotion category"""
# #         if emotion_category == "positive":
# #             reflections = [
# #                 f"I can see that you're feeling {emotion} about this positive experience.",
# #                 f"It sounds like you're experiencing {emotion} and happiness with this success.",
# #                 f"I'm hearing that this achievement is bringing up feelings of {emotion} and satisfaction for you."
# #             ]
# #         elif emotion_category == "negative":
# #             reflections = [
# #                 f"I can see that you're feeling {emotion} about this situation.",
# #                 f"It sounds like you're experiencing {emotion} right now.",
# #                 f"I'm hearing that this is bringing up feelings of {emotion} for you."
# #             ]
# #         else:
# #             reflections = [
# #                 f"I notice that you're feeling {emotion} about this.",
# #                 f"It seems like this situation is bringing up {emotion} for you.",
# #                 f"I'm hearing that you're experiencing {emotion} right now."
# #             ]
        
# #         return random.choice(reflections)
    
# #     def _empathy_response(self, user_input, emotion, emotion_category):
# #         """Generate empathetic responses based on emotion category"""
# #         if emotion_category == "positive":
# #             empathy = [
# #                 "It's wonderful when our hard work and abilities are recognized.",
# #                 "Those moments of achievement and recognition can be really meaningful.",
# #                 "Successes like this can bring such a sense of satisfaction and pride."
# #             ]
# #         elif emotion_category == "negative":
# #             empathy = [
# #                 "That sounds really challenging to go through.",
# #                 "I can imagine that must be difficult to experience.",
# #                 "It makes sense that you would feel that way in this situation."
# #             ]
# #         else:
# #             empathy = [
# #                 "I appreciate you sharing this experience.",
# #                 "Thank you for letting me know how you're feeling about this.",
# #                 "I value your openness about what you're going through."
# #             ]
        
# #         return random.choice(empathy)
    
# #     def _question_response(self, user_input, emotion, emotion_category):
# #         """Generate open-ended questions based on emotion category"""
# #         if emotion_category == "positive":
# #             questions = [
# #                 "What aspects of this achievement feel most meaningful to you?",
# #                 "How will you celebrate this success?",
# #                 "How might this positive experience influence your path forward?"
# #             ]
            
# #             # Add specific questions for promotions
# #             if "promotion" in user_input.lower():
# #                 questions.extend([
# #                     "What are you looking forward to most in your new role?",
# #                     "How does this promotion align with your career goals?",
# #                     "What skills do you think were most recognized in earning this promotion?"
# #                 ])
# #         elif emotion_category == "negative":
# #             questions = [
# #                 "What aspects of this situation feel most important to you right now?",
# #                 "How have you been taking care of yourself through this?",
# #                 "What would feel supportive or helpful to you in this moment?"
# #             ]
# #         else:
# #             questions = [
# #                 "What more would you like to explore about this experience?",
# #                 "How are you making sense of this situation?",
# #                 "What feels most important about this to you right now?"
# #             ]
        
# #         return random.choice(questions)
    
# #     def _personalized_response(self, user_input):
# #         """Create a personalized response referencing the user's words"""
# #         # Extract meaningful phrases to reflect back
# #         words = user_input.split()
# #         if len(words) > 4:
# #             start_idx = random.randint(0, max(0, len(words) - 4))
# #             phrase_length = min(random.randint(3, 4), len(words) - start_idx)
# #             phrase = " ".join(words[start_idx:start_idx + phrase_length])
            
# #             templates = [
# #                 f"When you mention '{phrase}', it sounds like a significant moment for you.",
# #                 f"Your words '{phrase}' suggest this is meaningful to you.",
# #                 f"I'm curious about what you shared regarding '{phrase}'."
# #             ]
            
# #             return random.choice(templates)
        
# #         return ""
    
# #     def generate_response(self, user_input, emotion):
# #         """
# #         Generate a response based on user input and detected emotion
# #         """
# #         try:
# #             logger.info(f"Generating response for emotion: {emotion}")
            
# #             # Determine emotion category
# #             emotion_category = self._get_emotion_category(emotion)
            
# #             # Select response components based on emotion category
# #             components = []
            
# #             if emotion_category == "positive":
# #                 # For positive emotions, prioritize celebration
# #                 components.append(self._celebration_response(user_input, emotion))
# #                 components.append(self._empathy_response(user_input, emotion, emotion_category))
                
# #                 # Sometimes add a reflective or questioning component
# #                 if random.random() > 0.5:
# #                     components.append(self._question_response(user_input, emotion, emotion_category))
# #             else:
# #                 # For negative or neutral emotions, use a different pattern
# #                 components.append(self._reflection_response(user_input, emotion, emotion_category))
# #                 components.append(self._empathy_response(user_input, emotion, emotion_category))
# #                 components.append(self._question_response(user_input, emotion, emotion_category))
            
# #             # Add personalized component 50% of the time
# #             if random.random() > 0.5:
# #                 personal = self._personalized_response(user_input)
# #                 if personal:
# #                     components.append(personal)
            
# #             # Combine components into a cohesive response
# #             response = " ".join(components)
            
# #             logger.info(f"Generated therapeutic response using techniques: {response[:30]}...")
# #             return response
            
# #         except Exception as e:
# #             logger.error(f"Error generating response: {e}")
# #             return "I'm here to listen. Could you tell me more about how you're feeling?"










# # # # response_generator/generate.py
# # # import json
# # # import random
# # # import logging
# # # import re
# # # import os
# # # from typing import Dict, List, Any
# # # from collections import Counter

# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # class ResponseGenerator:
# # #     def __init__(self, api_key=None, model="gpt-3.5-turbo", templates_path="response_generator/prompt_templates.json"):
# # #         """
# # #         Enhanced response generator that creates thoughtful, contextual responses
# # #         without relying on external APIs.
# # #         """
# # #         # Store API key (not used)
# # #         self.api_key = api_key
        
# # #         # Load prompt templates
# # #         try:
# # #             with open(templates_path, 'r') as f:
# # #                 self.templates = json.load(f)
# # #                 logger.info(f"Loaded templates from {templates_path}")
# # #         except (FileNotFoundError, json.JSONDecodeError) as e:
# # #             logger.warning(f"Error loading templates: {e}")
# # #             # Fallback to default templates
# # #             self.templates = self._default_templates()
            
# # #             # Try to save default templates
# # #             try:
# # #                 os.makedirs(os.path.dirname(templates_path), exist_ok=True)
# # #                 with open(templates_path, 'w') as f:
# # #                     json.dump(self.templates, f, indent=2)
# # #                 logger.info(f"Created default templates at {templates_path}")
# # #             except Exception as e:
# # #                 logger.error(f"Could not save templates: {e}")
        
# # #         # Initialize conversation memory
# # #         self.conversation_memory = []
        
# # #         # Load therapy techniques
# # #         self.therapy_techniques = {
# # #             "reflection": self._reflection_technique,
# # #             "validation": self._validation_technique,
# # #             "reframing": self._reframing_technique,
# # #             "empathy": self._empathy_technique,
# # #             "open_question": self._open_question_technique,
# # #             "suggestion": self._suggestion_technique
# # #         }
    
# # #     def _default_templates(self) -> Dict[str, List[str]]:
# # #         """Create default templates with therapeutic principles"""
# # #         return {
# # #             "joy": [
# # #                 "It's wonderful to hear you're feeling positive. What specific moments brought you this joy?",
# # #                 "I'm glad to hear you're experiencing joy. These positive emotions are valuable to recognize and savor.",
# # #                 "Your happiness comes through clearly. What do you think contributed most to these positive feelings?"
# # #             ],
# # #             "sadness": [
# # #                 "I notice you're feeling down. It takes courage to acknowledge these feelings, and I'm here to listen.",
# # #                 "I can hear that you're experiencing sadness. Would you like to explore what might be behind these feelings?",
# # #                 "When you say you're feeling sad, I wonder if there are specific situations or thoughts contributing to this emotion."
# # #             ],
# # #             "anger": [
# # #                 "I can sense your frustration. Anger often signals that something important to us has been affected.",
# # #                 "Your feelings of anger are valid. Sometimes exploring what's underneath can help us understand our reactions better.",
# # #                 "I notice you're feeling angry. Would it help to discuss what triggered these emotions?"
# # #             ],
# # #             "fear": [
# # #                 "I hear that you're feeling anxious or afraid. These feelings often come up when facing uncertainty.",
# # #                 "Your concerns are understandable. Fear is a natural response when we perceive threats to things we value.",
# # #                 "It sounds like you're experiencing some worry or anxiety. Would you like to explore ways to work with these feelings?"
# # #             ],
# # #             "default": [
# # #                 "Thank you for sharing your thoughts with me. I'm interested in understanding more about your experience.",
# # #                 "I appreciate you opening up. Would you like to explore these feelings further?",
# # #                 "I'm here to listen and support you. Is there a particular aspect of this you'd like to focus on?"
# # #             ]
# # #         }
    
# # #     def _extract_key_elements(self, text: str) -> Dict[str, Any]:
# # #         """
# # #         Extract key elements from the user's input to inform response generation
# # #         """
# # #         elements = {
# # #             "keywords": [],
# # #             "sentiment": "neutral",
# # #             "topics": [],
# # #             "people": [],
# # #             "concerns": [],
# # #             "time_references": []
# # #         }
        
# # #         # Extract keywords (simple approach)
# # #         words = re.findall(r'\b\w+\b', text.lower())
# # #         word_freq = Counter(words)
# # #         common_words = set(['i', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
# # #         elements["keywords"] = [word for word, freq in word_freq.most_common(5) if word not in common_words]
        
# # #         # Basic sentiment analysis
# # #         positive_words = {'good', 'great', 'happy', 'joy', 'excited', 'wonderful', 'love', 'like', 'positive'}
# # #         negative_words = {'bad', 'sad', 'angry', 'upset', 'worried', 'anxious', 'hate', 'dislike', 'negative', 'stress'}
        
# # #         pos_count = sum(1 for word in words if word in positive_words)
# # #         neg_count = sum(1 for word in words if word in negative_words)
        
# # #         if pos_count > neg_count:
# # #             elements["sentiment"] = "positive"
# # #         elif neg_count > pos_count:
# # #             elements["sentiment"] = "negative"
        
# # #         # Identify potential topics
# # #         topic_indicators = {
# # #             "work": {'job', 'work', 'boss', 'career', 'office', 'manager', 'colleague', 'coworker', 'promotion'},
# # #             "relationships": {'friend', 'partner', 'spouse', 'wife', 'husband', 'boyfriend', 'girlfriend', 'relationship', 'family', 'mother', 'father', 'parent', 'child'},
# # #             "health": {'health', 'sick', 'illness', 'doctor', 'hospital', 'pain', 'symptom', 'disease', 'medication'},
# # #             "mental_health": {'anxiety', 'depression', 'stress', 'therapy', 'mental', 'emotional', 'feeling', 'mood', 'therapist'},
# # #             "future": {'future', 'plan', 'goal', 'dream', 'aspiration', 'hope', 'expect', 'anticipate'}
# # #         }
        
# # #         for topic, indicators in topic_indicators.items():
# # #             if any(word in indicators for word in words):
# # #                 elements["topics"].append(topic)
        
# # #         # Extract people mentions
# # #         people_indicators = {'i', 'me', 'my', 'mine', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'we', 'us', 'our'}
# # #         if any(word in people_indicators for word in words):
# # #             if 'manager' in words or 'boss' in words:
# # #                 elements["people"].append("manager/boss")
# # #             if any(word in {'friend', 'friends'} for word in words):
# # #                 elements["people"].append("friend")
# # #             # Add more person detection here
        
# # #         # Detect concerns
# # #         concern_indicators = {'worried', 'concern', 'stress', 'problem', 'issue', 'trouble', 'difficult', 'challenging', 'scared', 'afraid', 'fear'}
# # #         elements["concerns"] = [word for word in words if word in concern_indicators]
        
# # #         # Time references
# # #         time_indicators = {
# # #             "past": {'yesterday', 'last', 'before', 'ago', 'previous', 'used to', 'earlier', 'once'},
# # #             "present": {'today', 'now', 'currently', 'at the moment', 'presently', 'right now'},
# # #             "future": {'tomorrow', 'next', 'will', 'going to', 'plan', 'future', 'soon', 'later'}
# # #         }
        
# # #         for time_frame, indicators in time_indicators.items():
# # #             if any(indicator in text.lower() for indicator in indicators):
# # #                 elements["time_references"].append(time_frame)
        
# # #         return elements
    
# # #     def _reflection_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Create a reflective response that mirrors back the user's feelings and concerns"""
# # #         reflections = [
# # #             f"I hear that you're feeling {elements['sentiment']} about {' and '.join(elements['topics']) if elements['topics'] else 'this'}.",
# # #             f"It sounds like you're expressing {elements['sentiment']} emotions related to {' and '.join(elements['topics']) if elements['topics'] else 'your situation'}.",
# # #             f"From what you've shared, I understand you're experiencing some {elements['sentiment']} feelings about {' and '.join(elements['topics']) if elements['topics'] else 'this'}."
# # #         ]
        
# # #         # Add specific reflections based on extracted elements
# # #         if 'work' in elements['topics']:
# # #             reflections.extend([
# # #                 "Your work situation seems to be affecting you significantly.",
# # #                 "I notice that your job is playing an important role in how you're feeling."
# # #             ])
        
# # #         if 'manager/boss' in elements['people']:
# # #             reflections.extend([
# # #                 "Your relationship with your manager appears to be a significant factor here.",
# # #                 "The way your boss interacts with you seems to have a real impact on your wellbeing."
# # #             ])
        
# # #         return random.choice(reflections)
    
# # #     def _validation_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Create a validating response that normalizes and accepts the user's feelings"""
# # #         validations = [
# # #             "Your feelings are completely valid and understandable given the situation.",
# # #             "It makes sense that you would feel this way, considering what you're going through.",
# # #             "Many people would feel similarly in your position.",
# # #             "Those emotions are a natural response to what you're experiencing."
# # #         ]
        
# # #         # Add specific validations based on emotion and situation
# # #         if elements['sentiment'] == 'negative':
# # #             validations.extend([
# # #                 "Difficult emotions like this are a normal part of the human experience.",
# # #                 "It's okay to feel this way - these emotions are telling you something important."
# # #             ])
        
# # #         return random.choice(validations)
    
# # #     def _reframing_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Offer a gentle reframe or different perspective"""
# # #         reframes = [
# # #             "I wonder if there might be another way to look at this situation?",
# # #             "Sometimes shifting our perspective can reveal new possibilities.",
# # #             "While acknowledging these feelings, I'm curious if there are aspects of this situation that might contain opportunities for growth?"
# # #         ]
        
# # #         # Add specific reframes based on the topics
# # #         if 'work' in elements['topics'] and elements['sentiment'] == 'negative':
# # #             reframes.extend([
# # #                 "Work challenges, while difficult, sometimes reveal our strengths and resilience.",
# # #                 "Difficult workplace situations can sometimes be opportunities to establish boundaries or clarify your values."
# # #             ])
        
# # #         return random.choice(reframes)
    
# # #     def _empathy_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Create an empathetic response that connects with the user's emotional experience"""
# # #         empathy = [
# # #             "That sounds really challenging to go through.",
# # #             "I can imagine that must be difficult to experience.",
# # #             "It makes sense that you would feel that way in this situation."
# # #         ]
        
# # #         # Add specific empathy based on extracted elements
# # #         if 'manager/boss' in elements['people'] and elements['sentiment'] == 'negative':
# # #             empathy.extend([
# # #                 "Difficult interactions with managers can be particularly stressful, since they affect both your daily experience and your sense of security.",
# # #                 "It can be especially challenging when work relationships are strained, as we spend so much of our time in that environment."
# # #             ])
        
# # #         return random.choice(empathy)
    
# # #     def _open_question_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Create an open-ended question to encourage exploration"""
# # #         questions = [
# # #             "What aspects of this situation feel most important to you right now?",
# # #             "How have you been taking care of yourself through this?",
# # #             "What would feel supportive or helpful to you in this moment?"
# # #         ]
        
# # #         # Add specific questions based on the topics and sentiment
# # #         if 'work' in elements['topics']:
# # #             questions.extend([
# # #                 "How does this work situation align with your longer-term goals or values?",
# # #                 "What would an ideal resolution to this work scenario look like for you?"
# # #             ])
        
# # #         if 'manager/boss' in elements['people'] and elements['sentiment'] == 'negative':
# # #             questions.extend([
# # #                 "What specifically about your manager's approach is most challenging for you?",
# # #                 "Have there been any moments where you felt better about your interactions with your manager?"
# # #             ])
        
# # #         return random.choice(questions)
    
# # #     def _suggestion_technique(self, user_input: str, elements: Dict[str, Any]) -> str:
# # #         """Offer gentle suggestions or ideas, phrased tentatively"""
# # #         suggestions = [
# # #             "Some people find it helpful to journal about difficult emotions like this.",
# # #             "Taking small breaks throughout the day might create some space for reflection.",
# # #             "Sometimes talking with trusted friends can offer new perspectives."
# # #         ]
        
# # #         # Add specific suggestions based on the topics
# # #         if 'work' in elements['topics'] and elements['sentiment'] == 'negative':
# # #             suggestions.extend([
# # #                 "Setting small boundaries at work sometimes helps create a sense of agency in challenging situations.",
# # #                 "Some find it helpful to clearly separate work time from personal time to maintain perspective."
# # #             ])
        
# # #         return random.choice(suggestions)
    




# # #         # Update your generate.py with these therapeutic techniques
# # # # response_generator/generate.py
# # # # Add this method or update the existing _add_therapeutic_techniques method

# # #     def _add_therapeutic_techniques(self, user_input, emotion):
# # #         """Add therapeutic techniques to response generation"""
# # #         techniques = {
# # #             "reflection": self._reflection_technique,
# # #             "validation": self._validation_technique,
# # #             "reframing": self._reframing_technique,
# # #             "empathy": self._empathy_technique,
# # #             "open_question": self._open_question_technique,
# # #             "celebration": self._celebration_technique  # Add a new technique for positive emotions
# # #         }
        
# # #         # Categorize emotions
# # #         positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", 
# # #                             "love", "optimism", "pride", "approval", "caring", "desire", "relief"]
# # #         negative_emotions = ["sadness", "anger", "fear", "disappointment", "grief", "remorse", 
# # #                             "embarrassment", "annoyance", "disapproval", "disgust"]
        
# # #         # Select appropriate techniques based on emotion category
# # #         if emotion in positive_emotions:
# # #             technique_list = ["celebration", "reflection", "open_question"]
# # #         elif emotion in negative_emotions:
# # #             technique_list = ["empathy", "validation", "reframing"]
# # #         else:
# # #             technique_list = ["reflection", "open_question", "empathy"]
        
# # #         # Generate response components
# # #         components = []
# # #         for technique in technique_list:
# # #             if technique in techniques:
# # #                 components.append(techniques[technique](user_input, emotion))
        
# # #         return " ".join(components)

# # #     # Add this new method for celebrating positive emotions
# # #     def _celebration_technique(self, user_input, emotion):
# # #         """Create a response that celebrates and reinforces positive experiences"""
# # #         celebrations = [
# # #             "It's wonderful to hear you're having such a positive experience!",
# # #             "I'm so glad to hear about your great day - those moments are worth savoring.",
# # #             "That's fantastic news! It's important to recognize and celebrate these positive experiences."
# # #         ]
# # #         return random.choice(celebrations)

# # #     # Update the reflection technique to better handle positive emotions
# # #     def _reflection_technique(self, user_input, emotion):
# # #         """Create a reflective response that mirrors back the user's feelings"""
# # #         # Categorize emotions
# # #         positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", 
# # #                             "love", "optimism", "pride", "approval", "caring", "desire", "relief"]
# # #         negative_emotions = ["sadness", "anger", "fear", "disappointment", "grief", "remorse", 
# # #                             "embarrassment", "annoyance", "disapproval", "disgust"]
        
# # #         if emotion in positive_emotions:
# # #             reflections = [
# # #                 f"I can see that you're feeling {emotion} and positive about this situation.",
# # #                 f"It sounds like you're experiencing {emotion} and happiness right now.",
# # #                 f"I'm hearing that this is bringing up feelings of {emotion} and satisfaction for you."
# # #             ]
# # #         elif emotion in negative_emotions:
# # #             reflections = [
# # #                 f"I can see that you're feeling {emotion} about this situation.",
# # #                 f"It sounds like you're experiencing {emotion} right now.",
# # #                 f"I'm hearing that this is bringing up feelings of {emotion} for you."
# # #             ]
# # #         else:
# # #             reflections = [
# # #                 f"I notice that you're feeling {emotion} about this.",
# # #                 f"It seems like this situation is bringing up {emotion} for you.",
# # #                 f"I'm hearing that you're experiencing {emotion} right now."
# # #             ]
        
# # #         return random.choice(reflections)

# # #     # Update the empathy technique to be appropriate for the emotion
# # #     def _empathy_technique(self, user_input, emotion):
# # #         """Create an empathetic response that connects with the user's emotional experience"""
# # #         # Categorize emotions
# # #         positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", 
# # #                             "love", "optimism", "pride", "approval", "caring", "desire", "relief"]
# # #         negative_emotions = ["sadness", "anger", "fear", "disappointment", "grief", "remorse", 
# # #                             "embarrassment", "annoyance", "disapproval", "disgust"]
        
# # #         if emotion in positive_emotions:
# # #             empathy = [
# # #                 "Those moments of recognition and praise can be really meaningful.",
# # #                 "It's so affirming when others appreciate our efforts and contributions.",
# # #                 "That kind of positive experience can really brighten your day and outlook."
# # #             ]
# # #         elif emotion in negative_emotions:
# # #             empathy = [
# # #                 "That sounds really challenging to go through.",
# # #                 "I can imagine that must be difficult to experience.",
# # #                 "It makes sense that you would feel that way in this situation."
# # #             ]
# # #         else:
# # #             empathy = [
# # #                 "I appreciate you sharing this experience.",
# # #                 "Thank you for letting me know how you're feeling about this.",
# # #                 "I value your openness about what you're going through."
# # #             ]
        
# # #         return random.choice(empathy)

# # #     # Update open question technique for positive emotions
# # #     def _open_question_technique(self, user_input, emotion):
# # #         """Create an open-ended question to encourage exploration"""
# # #         # Categorize emotions
# # #         positive_emotions = ["joy", "admiration", "amusement", "excitement", "gratitude", 
# # #                             "love", "optimism", "pride", "approval", "caring", "desire", "relief"]
# # #         negative_emotions = ["sadness", "anger", "fear", "disappointment", "grief", "remorse", 
# # #                             "embarrassment", "annoyance", "disapproval", "disgust"]
        
# # #         if emotion in positive_emotions:
# # #             questions = [
# # #                 "What aspects of this positive experience feel most meaningful to you?",
# # #                 "How might you build on this good feeling going forward?",
# # #                 "What does this success or positive moment tell you about yourself?"
# # #             ]
# # #         elif emotion in negative_emotions:
# # #             questions = [
# # #                 "What aspects of this situation feel most important to you right now?",
# # #                 "How have you been taking care of yourself through this?",
# # #                 "What would feel supportive or helpful to you in this moment?"
# # #             ]
# # #         else:
# # #             questions = [
# # #                 "What more would you like to explore about this experience?",
# # #                 "How are you making sense of this situation?",
# # #                 "What feels most important about this to you right now?"
# # #             ]
        
# # #         return random.choice(questions)
    
# # #     def generate_response(self, user_input, emotion):
# # #         """
# # #         Generate a response based on user input and detected emotion.
# # #         Now enhanced with therapeutic techniques.
# # #         """
# # #         try:
# # #             logger.info(f"Generating response for emotion: {emotion}")
            
# # #             # Add analysis of user input
# # #             from utils.content_analyzer import ContentAnalyzer
# # #             analyzer = ContentAnalyzer()
# # #             keywords = analyzer.extract_keywords(user_input)
# # #             topics = analyzer.analyze_topics(user_input)
            
# # #             # Generate therapeutic response
# # #             response = self._add_therapeutic_techniques(user_input, emotion)
            
# # #             # Add a personal touch by incorporating a key phrase from the user's input
# # #             if len(user_input.split()) > 5:  # Only if the input is substantial
# # #                 # Find meaningful phrases to reflect back
# # #                 words = user_input.split()
# # #                 start_idx = random.randint(0, max(0, len(words) - 4))
# # #                 phrase_length = min(random.randint(3, 5), len(words) - start_idx)
# # #                 phrase = " ".join(words[start_idx:start_idx + phrase_length])
                
# # #                 # Add the reflected phrase in a natural way
# # #                 reflection_templates = [
# # #                     f"I'm interested in what you said about '{phrase}'.",
# # #                     f"When you mention '{phrase}', what specifically comes to mind?",
# # #                     f"I notice you used the phrase '{phrase}'. Can you tell me more about that?"
# # #                 ]
                
# # #                 # Add the reflection at the end 50% of the time
# # #                 if random.random() > 0.5:
# # #                     response += " " + random.choice(reflection_templates)
            
# # #             logger.info(f"Generated therapeutic response using techniques: {response[:30]}...")
# # #             return response
            
# # #         except Exception as e:
# # #             logger.error(f"Error generating response: {e}")
# # #             return "I'm here to listen. Could you tell me more about how you're feeling?"
        
# # #     # def generate_response(self, user_input: str, emotion: str) -> str:
# # #     #     """
# # #     #     Generate a thoughtful, personalized response based on the user's input and detected emotion
# # #     #     """
# # #     #     try:
# # #     #         logger.info(f"Generating response for emotion: {emotion}")
            
# # #     #         # Extract key elements from user input
# # #     #         elements = self._extract_key_elements(user_input)
# # #     #         elements["detected_emotion"] = emotion
            
# # #     #         # Store conversation for context (not used in this simplified version)
# # #     #         self.conversation_memory.append({
# # #     #             "user_input": user_input,
# # #     #             "emotion": emotion,
# # #     #             "elements": elements
# # #     #         })
            
# # #     #         # Select therapy techniques to apply
# # #     #         # Choose 2-3 techniques based on the emotion and content
# # #     #         available_techniques = list(self.therapy_techniques.keys())
            
# # #     #         # Always include empathy and reflection for negative emotions
# # #     #         if emotion in ["sadness", "anger", "fear", "disappointment", "grief", "remorse"]:
# # #     #             primary_techniques = ["empathy", "validation"]
# # #     #             remaining = [t for t in available_techniques if t not in primary_techniques]
# # #     #             techniques_to_use = primary_techniques + random.sample(remaining, 1)
# # #     #         else:
# # #     #             # For positive or neutral emotions, use a different mix
# # #     #             techniques_to_use = random.sample(available_techniques, 3)
            
# # #     #         # Generate response components using selected techniques
# # #     #         response_components = []
# # #     #         for technique in techniques_to_use:
# # #     #             component = self.therapy_techniques[technique](user_input, elements)
# # #     #             response_components.append(component)
            
# # #     #         # Create personalized response using the extracted elements and selected techniques
# # #     #         response = " ".join(response_components)
            
# # #     #         # Add a personal touch by incorporating key phrases from the user's input
# # #     #         if len(user_input.split()) > 5:  # Only if the input is substantial
# # #     #             # Find meaningful phrases to reflect back
# # #     #             words = user_input.split()
# # #     #             start_idx = random.randint(0, max(0, len(words) - 4))
# # #     #             phrase_length = min(random.randint(3, 5), len(words) - start_idx)
# # #     #             phrase = " ".join(words[start_idx:start_idx + phrase_length])
                
# # #     #             # Add the reflected phrase in a natural way
# # #     #             reflection_templates = [
# # #     #                 f"I'm interested in what you said about '{phrase}'.",
# # #     #                 f"When you mention '{phrase}', what specifically comes to mind?",
# # #     #                 f"I notice you used the phrase '{phrase}'. Can you tell me more about that?"
# # #     #             ]
                
# # #     #             # Add the reflection at the end 50% of the time
# # #     #             if random.random() > 0.5:
# # #     #                 response += " " + random.choice(reflection_templates)
            
# # #     #         logger.info(f"Generated therapeutic response using techniques: {techniques_to_use}")
# # #     #         return response
            
# # #     #     except Exception as e:
# # #     #         logger.error(f"Error generating response: {e}")
# # #     #         return "I'm here to listen. Could you tell me more about how you're feeling?"




# # # # # response_generator/generate.py

# # # # import os
# # # # import json
# # # # import random
# # # # import time
# # # # import logging
# # # # from typing import Dict, List, Optional

# # # # import requests
# # # # from requests.exceptions import RequestException

# # # # # Configure logging
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# # # # )
# # # # logger = logging.getLogger(__name__)

# # # # class ResponseGenerator:
# # # #     """Generate therapeutic responses to user journal entries based on detected emotions."""
# # # #     # Add this to your ResponseGenerator class
# # # #     def _rate_limited_api_call(self, chat_history, emotion):
# # # #         """
# # # #         Call the OpenAI API with strict client-side rate limiting
# # # #         """
# # # #         # Enforce a minimum delay between API calls
# # # #         if hasattr(self, 'last_api_call_time'):
# # # #             elapsed = time.time() - self.last_api_call_time
# # # #             if elapsed < 10:  # At least 10 seconds between calls
# # # #                 time.sleep(10 - elapsed)
        
# # # #         # Make the API call
# # # #         result = self._call_openai_api(chat_history, emotion)
        
# # # #         # Update the timestamp
# # # #         self.last_api_call_time = time.time()
        
# # # #         return result
    

# # # #     def __init__(
# # # #         self, 
# # # #         api_key=None, 
# # # #         model="gpt-3.5-turbo-16k", 
# # # #         templates_path="response_generator/prompt_templates.json"
# # # #     ):
# # # #         """
# # # #         Initialize the response generator with OpenAI API key (if available) and templates.
        
# # # #         Args:
# # # #             api_key: OpenAI API key (optional)
# # # #             model: OpenAI model to use
# # # #             templates_path: Path to JSON file with prompt templates
# # # #         """
# # # #         # Store API key and model
# # # #         self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
# # # #         self.model = model
        
# # # #         # Load templates
# # # #         try:
# # # #             with open(templates_path, 'r') as f:
# # # #                 self.templates = json.load(f)
# # # #             logger.info(f"Loaded templates from {templates_path}")
# # # #         except (FileNotFoundError, json.JSONDecodeError) as e:
# # # #             logger.warning(f"Error loading templates: {e}")
# # # #             # Fallback to default templates
# # # #             self.templates = self._default_templates()
            
# # # #             # Try to create the directory and save default templates
# # # #             try:
# # # #                 os.makedirs(os.path.dirname(templates_path), exist_ok=True)
# # # #                 with open(templates_path, 'w') as f:
# # # #                     json.dump(self.templates, f, indent=2)
# # # #                 logger.info(f"Created default templates file at {templates_path}")
# # # #             except Exception as e:
# # # #                 logger.warning(f"Could not save default templates: {e}")
        
# # # #         # Define fallback responses for different emotions
# # # #         self.responses = {
# # # #             "joy": [
# # # #                 "I'm glad to hear you're feeling positive! What's bringing you joy today?",
# # # #                 "That's wonderful to hear! Positive emotions are worth savoring. What's making you feel this way?",
# # # #                 "It's great that you're feeling good. Is there something specific that brightened your day?"
# # # #             ],
# # # #             "sadness": [
# # # #                 "I'm here to listen. Could you tell me more about what's making you feel sad?",
# # # #                 "I'm sorry you're feeling down. Sometimes talking about it can help. Would you like to share more?",
# # # #                 "It's okay to feel sad sometimes. Would you like to talk about what's on your mind?"
# # # #             ],
# # # #             "anger": [
# # # #                 "I can understand feeling frustrated. What happened that triggered these feelings?",
# # # #                 "It sounds like something's bothering you. Would you like to talk about what's causing these feelings?",
# # # #                 "When we're angry, it can help to take a deep breath. What's on your mind right now?"
# # # #             ],
# # # #             "fear": [
# # # #                 "It's natural to feel anxious sometimes. What's causing you concern?",
# # # #                 "I'm here to listen. Would you like to talk about what's making you feel afraid?",
# # # #                 "Feeling scared can be overwhelming. Is there something specific that's worrying you?"
# # # #             ],
# # # #             "default": [
# # # #                 "I'm here to listen. Could you tell me more about how you're feeling?",
# # # #                 "Thank you for sharing. Would you like to explore these feelings a bit more?",
# # # #                 "I appreciate you opening up. Is there anything specific you'd like to talk about today?"
# # # #             ]
# # # #         }
        
# # # #         # Log initialization info
# # # #         if self.api_key:
# # # #             logger.info(f"ResponseGenerator initialized with API key for model {model}")
# # # #         else:
# # # #             logger.info("ResponseGenerator initialized with fallback responses only (no API key)")
    
# # # #     def _default_templates(self):
# # # #         """
# # # #         Default templates in case the template file is not found.
        
# # # #         Returns:
# # # #             Dict mapping emotion categories to lists of template strings
# # # #         """
# # # #         return {
# # # #             "sadness": [
# # # #                 "The user is feeling sad and wrote: '{user_input}'. Provide a compassionate and supportive response that acknowledges their feelings and offers gentle encouragement.",
# # # #                 "Respond with empathy to someone who's feeling down. They wrote: '{user_input}'. Give a validating response that shows understanding."
# # # #             ],
# # # #             "joy": [
# # # #                 "The user is feeling joy and wrote: '{user_input}'. Respond in a way that celebrates their positive feelings and encourages them to savor this moment.",
# # # #                 "Share in the user's happiness about: '{user_input}'. Give a warm response that mirrors their positive emotion."
# # # #             ],
# # # #             "anger": [
# # # #                 "The user is feeling angry and wrote: '{user_input}'. Provide a calming response that validates their feelings without reinforcing negative thoughts.",
# # # #                 "Respond with understanding to someone feeling frustrated. They wrote: '{user_input}'. Offer a balanced perspective that acknowledges their feelings."
# # # #             ],
# # # #             "fear": [
# # # #                 "The user is feeling anxious or fearful and wrote: '{user_input}'. Provide a grounding response that helps them feel safer and offers perspective.",
# # # #                 "Respond with reassurance to someone feeling afraid. They wrote: '{user_input}'. Offer support and gentle coping suggestions."
# # # #             ],
# # # #             "default": [
# # # #                 "Respond with empathy to the following journal entry: '{user_input}'. Provide a thoughtful, supportive response that a compassionate therapist might give.",
# # # #                 "Here's what someone wrote in their journal: '{user_input}'. Give a kind, thoughtful response that shows understanding."
# # # #             ]
# # # #         }
    
# # # #     def _get_template_for_emotion(self, emotion):
# # # #         """
# # # #         Get an appropriate template for the detected emotion.
        
# # # #         Args:
# # # #             emotion: The detected emotion
            
# # # #         Returns:
# # # #             A template string for the given emotion
# # # #         """
# # # #         # Map specific emotions to broader categories for template selection
# # # #         emotion_mapping = {
# # # #             # Joy-related emotions
# # # #             "joy": "joy", "amusement": "joy", "excitement": "joy", 
# # # #             "gratitude": "joy", "love": "joy", "optimism": "joy",
# # # #             "pride": "joy", "admiration": "joy", "approval": "joy",
# # # #             "caring": "joy", "desire": "joy", "relief": "joy",
            
# # # #             # Sadness-related emotions
# # # #             "sadness": "sadness", "disappointment": "sadness", 
# # # #             "grief": "sadness", "remorse": "sadness", 
# # # #             "embarrassment": "sadness",
            
# # # #             # Anger-related emotions
# # # #             "anger": "anger", "annoyance": "anger", 
# # # #             "disapproval": "anger", "disgust": "anger",
            
# # # #             # Fear-related emotions
# # # #             "fear": "fear", "nervousness": "fear", "confusion": "fear",
            
# # # #             # Other emotions
# # # #             "surprise": "default", "realization": "default", 
# # # #             "curiosity": "default", "neutral": "default"
# # # #         }
        
# # # #         # Get the template category
# # # #         template_category = emotion_mapping.get(emotion.lower(), "default")
        
# # # #         # Get templates for this category
# # # #         templates = self.templates.get(template_category, self.templates["default"])
        
# # # #         # Return a random template from the list
# # # #         return random.choice(templates)
    
# # # #     def _call_openai_api(self, chat_history, emotion):
# # # #         """
# # # #         Call the OpenAI API with improved error handling and rate limit management.
        
# # # #         Args:
# # # #             chat_history: List of message dictionaries (role, content)
# # # #             emotion: The detected emotion for better therapeutic guidance
            
# # # #         Returns:
# # # #             The generated response or None if the API call fails
# # # #         """
# # # #         if not self.api_key:
# # # #             logger.warning("No API key available for OpenAI call")
# # # #             return None

# # # #         url = "https://api.openai.com/v1/chat/completions"
# # # #         headers = {
# # # #             "Authorization": f"Bearer {self.api_key}",
# # # #             "Content-Type": "application/json",
# # # #         }

# # # #         # Enhanced system prompt for better therapeutic responses
# # # #         system_prompt = f"""You are a compassionate, empathetic therapist providing emotional support and practical advice.

# # # # The user is experiencing the emotion: {emotion.upper()}

# # # # Your guidelines:
# # # # 1. Acknowledge their feelings and validate their emotional experience
# # # # 2. Demonstrate deep empathy and understanding for their situation
# # # # 3. Provide specific, actionable advice or coping strategies
# # # # 4. Be warm, supportive, and conversational - like a real human therapist
# # # # 5. Ask thoughtful questions that encourage self-reflection
# # # # 6. Offer perspective that might help them see their situation differently
# # # # 7. Suggest healthy ways to process their emotions
# # # # 8. Keep responses conversational and natural, not clinical or distant

# # # # Your goal is to be genuinely helpful - offering both emotional support AND practical guidance."""

# # # #         # Prepend the enhanced system prompt to the chat history
# # # #         messages = [
# # # #             {"role": "system", "content": system_prompt}
# # # #         ] + chat_history

# # # #         payload = {
# # # #             "model": self.model,
# # # #             "messages": messages,
# # # #             "temperature": 0.6,  # Slightly lower temperature for more focused responses
# # # #             "max_tokens": 300,  # Increased token count for more detailed therapeutic responses
# # # #         }

# # # #         # Define wait times for retries with exponential backoff
# # # #         wait_times = [5, 15, 30]  # Wait times in seconds

# # # #         # Retry up to 3 times with better handling
# # # #         for attempt in range(3):
# # # #             try:
# # # #                 logger.info(f"Calling OpenAI API (attempt {attempt + 1}/3)")
                
# # # #                 # Add a small delay before each retry
# # # #                 if attempt > 0:
# # # #                     time.sleep(1)
                
# # # #                 resp = requests.post(url, headers=headers, json=payload, timeout=30)
                
# # # #                 # Print detailed response info for debugging
# # # #                 logger.info(f"OpenAI API Status: {resp.status_code}")
                
# # # #                 # Handle rate limiting specifically
# # # #                 if resp.status_code == 429:
# # # #                     retry_after = int(resp.headers.get('Retry-After', wait_times[min(attempt, len(wait_times)-1)]))
# # # #                     logger.warning(f"Rate limited by OpenAI API, waiting {retry_after}s before retry")
# # # #                     time.sleep(retry_after)
# # # #                     continue
                
# # # #                 # Raise exception for other status codes
# # # #                 resp.raise_for_status()
                
# # # #                 response = resp.json()["choices"][0]["message"]["content"].strip()
# # # #                 logger.info("OpenAI API call successful")
# # # #                 return response
                
# # # #             except RequestException as e:
# # # #                 error_msg = str(e)
# # # #                 logger.error(f"Error calling OpenAI API: {error_msg}")
                
# # # #                 # Print response text if available
# # # #                 if hasattr(e, 'response') and e.response is not None:
# # # #                     logger.error(f"Error response: {e.response.text}")
                
# # # #                 if attempt < 2:
# # # #                     wait_time = wait_times[attempt]
# # # #                     logger.info(f"Retrying in {wait_time} seconds...")
# # # #                     time.sleep(wait_time)
# # # #                     continue
# # # #                 break
                
# # # #             except Exception as e:
# # # #                 logger.error(f"Unexpected error calling OpenAI API: {str(e)}")
# # # #                 if attempt < 2:
# # # #                     wait_time = wait_times[attempt]
# # # #                     logger.info(f"Retrying in {wait_time} seconds...")
# # # #                     time.sleep(wait_time)
# # # #                     continue
# # # #                 break

# # # #         logger.warning("All OpenAI API attempts failed")
# # # #         return None
    
# # # #     def _get_fallback_response(self, emotion):
# # # #         """
# # # #         Get a fallback response for the given emotion.
        
# # # #         Args:
# # # #             emotion: The detected emotion
            
# # # #         Returns:
# # # #             A fallback response string
# # # #         """
# # # #         # Map to broader emotion categories
# # # #         emotion_mapping = {
# # # #             # Joy-related emotions
# # # #             "joy": "joy", "amusement": "joy", "excitement": "joy", 
# # # #             "gratitude": "joy", "love": "joy", "optimism": "joy",
# # # #             "pride": "joy", "admiration": "joy", "approval": "joy",
# # # #             "caring": "joy", "desire": "joy", "relief": "joy",
            
# # # #             # Sadness-related emotions
# # # #             "sadness": "sadness", "disappointment": "sadness", 
# # # #             "grief": "sadness", "remorse": "sadness", 
# # # #             "embarrassment": "sadness",
            
# # # #             # Anger-related emotions
# # # #             "anger": "anger", "annoyance": "anger", 
# # # #             "disapproval": "anger", "disgust": "anger",
            
# # # #             # Fear-related emotions
# # # #             "fear": "fear", "nervousness": "fear", "confusion": "fear",
            
# # # #             # Default for other emotions
# # # #             "surprise": "default", "realization": "default", 
# # # #             "curiosity": "default", "neutral": "default"
# # # #         }
        
# # # #         # Get the response category
# # # #         response_category = emotion_mapping.get(emotion.lower(), "default")
        
# # # #         # Get responses for this category
# # # #         responses = self.responses.get(response_category, self.responses["default"])
        
# # # #         # Return a random response from the list
# # # #         return random.choice(responses)
    
# # # #     def generate_response(self, user_input, emotion):
# # # #         """
# # # #         Generate a response based on the user input and detected emotion.
        
# # # #         This method tries to use the OpenAI API first if available,
# # # #         and falls back to template-based responses if the API is unavailable or fails.
        
# # # #         Args:
# # # #             user_input: The user's journal entry
# # # #             emotion: The detected emotion
            
# # # #         Returns:
# # # #             A therapeutic response string
# # # #         """
# # # #         try:
# # # #             logger.info(f"Generating response for emotion: {emotion}")
            
# # # #             # Try using OpenAI API if key is available
# # # #             if self.api_key:
# # # #                 # Create a simple chat history with the user's input
# # # #                 chat_history = [{"role": "user", "content": user_input}]
                
# # # #                 # Call OpenAI API with the emotion parameter
# # # #                 api_response = self._call_openai_api(chat_history, emotion)
                
# # # #                 # If API call successful, return the response
# # # #                 if api_response:
# # # #                     return api_response
                
# # # #                 # Log fallback to templates
# # # #                 logger.info("Falling back to template-based response")
            
# # # #             # Get a template for the emotion
# # # #             template = self._get_template_for_emotion(emotion)
            
# # # #             # If template contains formatting placeholders, format it with user input
# # # #             if "{user_input}" in template:
# # # #                 formatted_template = template.format(user_input=user_input)
# # # #                 # In a real-world scenario, we would process this template further
# # # #                 # or use an alternative LLM, but for this assignment, we'll use a fallback
# # # #                 return self._get_fallback_response(emotion)
# # # #             else:
# # # #                 # If template doesn't contain placeholders, use it directly
# # # #                 return template
            
# # # #         except Exception as e:
# # # #             # Log the error and return a generic response
# # # #             logger.error(f"Error generating response: {e}")
# # # #             return "I'm here to listen. Could you tell me more about how you're feeling?"







# # # # # import json
# # # # # import random
# # # # # import os

# # # # # class ResponseGenerator:
# # # # #     def __init__(self, api_key=None, model="gpt-3.5-turbo", templates_path="response_generator/prompt_templates.json"):

# # # # #         """
# # # # #         Initialize the response generator with templates
# # # # #         """
# # # # #         # Load prompt templates
# # # # #         try:
# # # # #             with open(templates_path, 'r') as f:
# # # # #                 self.templates = json.load(f)
# # # # #         except (FileNotFoundError, json.JSONDecodeError) as e:
# # # # #             print(f"Error loading templates: {e}")
# # # # #             # Fallback to default templates
# # # # #             self.templates = self._default_templates()
        
# # # # #         # Define hardcoded responses for different emotions
# # # # #         self.responses = {
# # # # #             "joy": [
# # # # #                 "I'm glad to hear you're feeling positive! What's bringing you joy today?",
# # # # #                 "That's wonderful to hear! Positive emotions are worth savoring. What's making you feel this way?",
# # # # #                 "It's great that you're feeling good. Is there something specific that brightened your day?"
# # # # #             ],
# # # # #             "sadness": [
# # # # #                 "I'm here to listen. Could you tell me more about what's making you feel sad?",
# # # # #                 "I'm sorry you're feeling down. Sometimes talking about it can help. Would you like to share more?",
# # # # #                 "It's okay to feel sad sometimes. Would you like to talk about what's on your mind?"
# # # # #             ],
# # # # #             "anger": [
# # # # #                 "I can understand feeling frustrated. What happened that triggered these feelings?",
# # # # #                 "It sounds like something's bothering you. Would you like to talk about what's causing these feelings?",
# # # # #                 "When we're angry, it can help to take a deep breath. What's on your mind right now?"
# # # # #             ],
# # # # #             "fear": [
# # # # #                 "It's natural to feel anxious sometimes. What's causing you concern?",
# # # # #                 "I'm here to listen. Would you like to talk about what's making you feel afraid?",
# # # # #                 "Feeling scared can be overwhelming. Is there something specific that's worrying you?"
# # # # #             ],
# # # # #             "default": [
# # # # #                 "I'm here to listen. Could you tell me more about how you're feeling?",
# # # # #                 "Thank you for sharing. Would you like to explore these feelings a bit more?",
# # # # #                 "I appreciate you opening up. Is there anything specific you'd like to talk about today?"
# # # # #             ]
# # # # #         }
    
# # # # #     def _default_templates(self):
# # # # #         """
# # # # #         Default templates in case the template file is not found
# # # # #         """
# # # # #         return {
# # # # #             "sadness": [
# # # # #                 "The user is feeling sad and wrote: '{user_input}'. Provide a compassionate and supportive response that acknowledges their feelings and offers gentle encouragement.",
# # # # #                 "Respond with empathy to someone who's feeling down. They wrote: '{user_input}'. Give a validating response that shows understanding."
# # # # #             ],
# # # # #             "joy": [
# # # # #                 "The user is feeling joy and wrote: '{user_input}'. Respond in a way that celebrates their positive feelings and encourages them to savor this moment.",
# # # # #                 "Share in the user's happiness about: '{user_input}'. Give a warm response that mirrors their positive emotion."
# # # # #             ],
# # # # #             "anger": [
# # # # #                 "The user is feeling angry and wrote: '{user_input}'. Provide a calming response that validates their feelings without reinforcing negative thoughts.",
# # # # #                 "Respond with understanding to someone feeling frustrated. They wrote: '{user_input}'. Offer a balanced perspective that acknowledges their feelings."
# # # # #             ],
# # # # #             "fear": [
# # # # #                 "The user is feeling anxious or fearful and wrote: '{user_input}'. Provide a grounding response that helps them feel safer and offers perspective.",
# # # # #                 "Respond with reassurance to someone feeling afraid. They wrote: '{user_input}'. Offer support and gentle coping suggestions."
# # # # #             ],
# # # # #             "default": [
# # # # #                 "Respond with empathy to the following journal entry: '{user_input}'. Provide a thoughtful, supportive response that a compassionate therapist might give.",
# # # # #                 "Here's what someone wrote in their journal: '{user_input}'. Give a kind, thoughtful response that shows understanding."
# # # # #             ]
# # # # #         }
    
# # # # #     def _get_template_for_emotion(self, emotion):
# # # # #         """
# # # # #         Get an appropriate template for the detected emotion
# # # # #         """
# # # # #         # Map specific emotions to broader categories for template selection
# # # # #         emotion_mapping = {
# # # # #             # Joy-related emotions
# # # # #             "joy": "joy",
# # # # #             "amusement": "joy",
# # # # #             "excitement": "joy",
# # # # #             "gratitude": "joy",
# # # # #             "love": "joy",
# # # # #             "optimism": "joy",
# # # # #             "pride": "joy",
# # # # #             "admiration": "joy",
# # # # #             "approval": "joy",
# # # # #             "caring": "joy",
# # # # #             "desire": "joy",
# # # # #             "relief": "joy",
            
# # # # #             # Sadness-related emotions
# # # # #             "sadness": "sadness",
# # # # #             "disappointment": "sadness",
# # # # #             "grief": "sadness",
# # # # #             "remorse": "sadness",
# # # # #             "embarrassment": "sadness",
            
# # # # #             # Anger-related emotions
# # # # #             "anger": "anger",
# # # # #             "annoyance": "anger",
# # # # #             "disapproval": "anger",
# # # # #             "disgust": "anger",
            
# # # # #             # Fear-related emotions
# # # # #             "fear": "fear",
# # # # #             "nervousness": "fear",
# # # # #             "confusion": "fear",
            
# # # # #             # Other emotions
# # # # #             "surprise": "default",
# # # # #             "realization": "default",
# # # # #             "curiosity": "default",
# # # # #             "neutral": "default"
# # # # #         }
        
# # # # #         # Get the template category
# # # # #         template_category = emotion_mapping.get(emotion.lower(), "default")
        
# # # # #         # Get templates for this category
# # # # #         templates = self.templates.get(template_category, self.templates["default"])
        
# # # # #         # Return a random template from the list
# # # # #         return random.choice(templates)
    
# # # # #     def generate_response(self, user_input, emotion):
# # # # #         """
# # # # #         Generate a simple response based on detected emotion without using OpenAI API
# # # # #         """
# # # # #         try:
# # # # #             # Map to broader emotion categories
# # # # #             emotion_mapping = {
# # # # #                 # Joy-related emotions
# # # # #                 "joy": "joy", "amusement": "joy", "excitement": "joy", 
# # # # #                 "gratitude": "joy", "love": "joy", "optimism": "joy",
# # # # #                 "pride": "joy", "admiration": "joy", "approval": "joy",
# # # # #                 "caring": "joy", "desire": "joy", "relief": "joy",
                
# # # # #                 # Sadness-related emotions
# # # # #                 "sadness": "sadness", "disappointment": "sadness", 
# # # # #                 "grief": "sadness", "remorse": "sadness", 
# # # # #                 "embarrassment": "sadness",
                
# # # # #                 # Anger-related emotions
# # # # #                 "anger": "anger", "annoyance": "anger", 
# # # # #                 "disapproval": "anger", "disgust": "anger",
                
# # # # #                 # Fear-related emotions
# # # # #                 "fear": "fear", "nervousness": "fear", "confusion": "fear",
                
# # # # #                 # Default for other emotions
# # # # #                 "surprise": "default", "realization": "default", 
# # # # #                 "curiosity": "default", "neutral": "default"
# # # # #             }
            
# # # # #             # Get the response category
# # # # #             response_category = emotion_mapping.get(emotion.lower(), "default")
            
# # # # #             # Get responses for this category
# # # # #             responses = self.responses.get(response_category, self.responses["default"])
            
# # # # #             # Return a random response from the list
# # # # #             return random.choice(responses)
            
# # # # #         except Exception as e:
# # # # #             print(f"Error generating response: {e}")
# # # # #             return "I'm here to listen. Could you tell me more about how you're feeling?"



# # # # # # import os
# # # # # # import openai
# # # # # # import json
# # # # # # import random

# # # # # # class ResponseGenerator:
# # # # # #     def __init__(self, api_key=None, model="gpt-3.5-turbo", templates_path="response_generator/prompt_templates.json"):
# # # # # #         """
# # # # # #         Initialize the response generator with OpenAI API key and templates
# # # # # #         """
# # # # # #         # Set OpenAI API key from environment variable if not provided
# # # # # #         self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
# # # # # #         if not self.api_key:
# # # # # #             raise ValueError("OpenAI API key is required. Set it via the api_key parameter or OPENAI_API_KEY environment variable.")
        
# # # # # #         openai.api_key = self.api_key
# # # # # #         self.model = model
        
# # # # # #         # Load prompt templates
# # # # # #         try:
# # # # # #             with open(templates_path, 'r') as f:
# # # # # #                 self.templates = json.load(f)
# # # # # #         except (FileNotFoundError, json.JSONDecodeError) as e:
# # # # # #             print(f"Error loading templates: {e}")
# # # # # #             # Fallback to default templates
# # # # # #             self.templates = self._default_templates()
    
# # # # # #     def _default_templates(self):
# # # # # #         """
# # # # # #         Default templates in case the template file is not found
# # # # # #         """
# # # # # #         return {
# # # # # #             "sadness": [
# # # # # #                 "The user is feeling sad and wrote: '{user_input}'. Provide a compassionate and supportive response that acknowledges their feelings and offers gentle encouragement.",
# # # # # #                 "Respond with empathy to someone who's feeling down. They wrote: '{user_input}'. Give a validating response that shows understanding."
# # # # # #             ],
# # # # # #             "joy": [
# # # # # #                 "The user is feeling joy and wrote: '{user_input}'. Respond in a way that celebrates their positive feelings and encourages them to savor this moment.",
# # # # # #                 "Share in the user's happiness about: '{user_input}'. Give a warm response that mirrors their positive emotion."
# # # # # #             ],
# # # # # #             "anger": [
# # # # # #                 "The user is feeling angry and wrote: '{user_input}'. Provide a calming response that validates their feelings without reinforcing negative thoughts.",
# # # # # #                 "Respond with understanding to someone feeling frustrated. They wrote: '{user_input}'. Offer a balanced perspective that acknowledges their feelings."
# # # # # #             ],
# # # # # #             "fear": [
# # # # # #                 "The user is feeling anxious or fearful and wrote: '{user_input}'. Provide a grounding response that helps them feel safer and offers perspective.",
# # # # # #                 "Respond with reassurance to someone feeling afraid. They wrote: '{user_input}'. Offer support and gentle coping suggestions."
# # # # # #             ],
# # # # # #             "default": [
# # # # # #                 "Respond with empathy to the following journal entry: '{user_input}'. Provide a thoughtful, supportive response that a compassionate therapist might give.",
# # # # # #                 "Here's what someone wrote in their journal: '{user_input}'. Give a kind, thoughtful response that shows understanding."
# # # # # #             ]
# # # # # #         }
    
# # # # # #     def _get_template_for_emotion(self, emotion):
# # # # # #         """
# # # # # #         Get an appropriate template for the detected emotion
# # # # # #         """
# # # # # #         # Map specific emotions to broader categories for template selection
# # # # # #         emotion_mapping = {
# # # # # #             # Joy-related emotions
# # # # # #             "joy": "joy",
# # # # # #             "amusement": "joy",
# # # # # #             "excitement": "joy",
# # # # # #             "gratitude": "joy",
# # # # # #             "love": "joy",
# # # # # #             "optimism": "joy",
# # # # # #             "pride": "joy",
# # # # # #             "admiration": "joy",
# # # # # #             "approval": "joy",
# # # # # #             "caring": "joy",
# # # # # #             "desire": "joy",
# # # # # #             "relief": "joy",
            
# # # # # #             # Sadness-related emotions
# # # # # #             "sadness": "sadness",
# # # # # #             "disappointment": "sadness",
# # # # # #             "grief": "sadness",
# # # # # #             "remorse": "sadness",
# # # # # #             "embarrassment": "sadness",
            
# # # # # #             # Anger-related emotions
# # # # # #             "anger": "anger",
# # # # # #             "annoyance": "anger",
# # # # # #             "disapproval": "anger",
# # # # # #             "disgust": "anger",
            
# # # # # #             # Fear-related emotions
# # # # # #             "fear": "fear",
# # # # # #             "nervousness": "fear",
# # # # # #             "confusion": "fear",
            
# # # # # #             # Other emotions
# # # # # #             "surprise": "default",
# # # # # #             "realization": "default",
# # # # # #             "curiosity": "default",
# # # # # #             "neutral": "default"
# # # # # #         }
        
# # # # # #         # Get the template category
# # # # # #         template_category = emotion_mapping.get(emotion.lower(), "default")
        
# # # # # #         # Get templates for this category
# # # # # #         templates = self.templates.get(template_category, self.templates["default"])
        
# # # # # #         # Return a random template from the list
# # # # # #         return random.choice(templates)
    
    





# # # # # # # import openai
# # # # # # # import json
# # # # # # # from emotion_detector.predict import predict_emotion

# # # # # # # openai.api_key = "your-api-key-here"  # Replace with your actual key

# # # # # # # with open("response_generator/prompt_templates.json", "r") as f:
# # # # # # #     templates = json.load(f)

# # # # # # # def generate_response(user_input):
# # # # # # #     emotion, _ = predict_emotion(user_input)
# # # # # # #     prompt = templates.get(emotion, "I understand. Can you tell me more?")
# # # # # # #     full_prompt = f"User: {user_input}\nTherapist: {prompt}"
# # # # # # #     response = openai.Completion.create(
# # # # # # #         engine="text-davinci-003",
# # # # # # #         prompt=full_prompt,
# # # # # # #         max_tokens=50
# # # # # # #     )
# # # # # # #     return emotion, response.choices[0].text.strip()

# # # # # # # if __name__ == '__main__':
# # # # # # #     text = "I'm feeling very anxious about my future."
# # # # # # #     emotion, reply = generate_response(text)
# # # # # # #     print(f"Emotion: {emotion}\nResponse: {reply}")
