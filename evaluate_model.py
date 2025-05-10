import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from emotion_detector.predict import EmotionDetector
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_emotion_detector(data_path="data/goemotions.csv", model_path="models/emotion_detector", 
                              test_size=0.2, random_seed=42):
    """
    Evaluate the emotion detector model and calculate metrics including F1 score
    
    Args:
        data_path: Path to the GoEmotions dataset
        model_path: Path to the fine-tuned model
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading data from {data_path}...")
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded with {len(df)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Initialize the emotion detector
    print(f"Loading emotion detector model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Using default pre-trained model.")
    
    detector = EmotionDetector(model_path=model_path)
    
    # Split the data into train and test sets
    np.random.seed(random_seed)
    test_indices = np.random.choice(df.index, size=int(len(df) * test_size), replace=False)
    test_df = df.iloc[test_indices]
    
    print(f"Evaluating on {len(test_df)} test samples...")
    
    # Make predictions on the test set
    true_labels = []
    pred_labels = []
    
    for _, row in test_df.iterrows():
        text = row['text']
        true_label = row['label']
        
        # Get prediction
        try:
            pred_label, confidence = detector.predict(text)
            true_labels.append(true_label)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"Error predicting for text '{text[:50]}...': {e}")
    
    # Calculate metrics
    print("Calculating metrics...")
    
    # Get unique labels from both true and predicted labels
    all_labels = sorted(list(set(true_labels + pred_labels)))
    
    # Calculate overall metrics
    report = classification_report(true_labels, pred_labels, labels=all_labels, output_dict=True)
    
    # Extract overall metrics
    overall_metrics = {
        'accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
    
    # Get per-class metrics
    class_metrics = {}
    for label in all_labels:
        if label in report:
            class_metrics[label] = {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
                'support': report[label]['support']
            }
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    # Plot F1 scores by emotion
    f1_scores = {label: metrics['f1'] for label, metrics in class_metrics.items()}
    plt.figure(figsize=(14, 8))
    plt.bar(f1_scores.keys(), f1_scores.values())
    plt.xticks(rotation=90)
    plt.xlabel('Emotion')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Emotion')
    plt.tight_layout()
    f1_plot_path = "f1_scores_by_emotion.png"
    plt.savefig(f1_plot_path)
    print(f"F1 scores plot saved to {f1_plot_path}")
    
    # Return results
    results = {
        'overall': overall_metrics,
        'class': class_metrics,
        'confusion_matrix': cm,
        'all_labels': all_labels
    }
    
    # Print summary
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro Avg F1: {overall_metrics['macro_avg_f1']:.4f}")
    print(f"Weighted Avg F1: {overall_metrics['weighted_avg_f1']:.4f}")
    print("\nTop 5 performing emotions:")
    top_emotions = sorted([(label, metrics['f1']) for label, metrics in class_metrics.items()], 
                          key=lambda x: x[1], reverse=True)[:5]
    for emotion, f1 in top_emotions:
        print(f"  {emotion}: F1 = {f1:.4f}")
    
    print("\nBottom 5 performing emotions:")
    bottom_emotions = sorted([(label, metrics['f1']) for label, metrics in class_metrics.items()], 
                            key=lambda x: x[1])[:5]
    for emotion, f1 in bottom_emotions:
        print(f"  {emotion}: F1 = {f1:.4f}")
    
    return results

if __name__ == "__main__":
    # You can adjust these paths to match your setup
    results = evaluate_emotion_detector(
        data_path="data/goemotions.csv",
        model_path="models/emotion_detector"
    )
    
    if results:
        print("\nEvaluation completed successfully!")