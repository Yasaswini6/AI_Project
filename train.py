import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight       # === CHANGED ===
import os
import argparse
import time

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_emotion_model(data_path, output_dir, epochs=7, batch_size=16, learning_rate=5e-5):
    """
    Fine-tune a DistilBERT model on the GoEmotions dataset,
    with class-weighted loss and warmup scheduler.
    """
    start_time = time.time()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare texts and labels
    texts = df['text'].tolist()
    if 'label' in df.columns:
        labels = df['label'].tolist()
        if isinstance(labels[0], str):
            unique_labels = sorted(list(set(labels)))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = [label_map[label] for label in labels]
            print(f"Converted string labels to integers. Found {len(unique_labels)} unique labels.")
    else:
        emotion_columns = [col for col in df.columns if col != 'text']
        labels = df[emotion_columns].values.argmax(axis=1).tolist()
    
    print(f"Dataset loaded with {len(texts)} samples")
    
    # Compute class weights to handle imbalance
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Computed class weights: {dict(zip(classes, weights))}")
    
    # Split into train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    print(f"Training set: {len(train_texts)} samples, Validation set: {len(val_texts)} samples")
    
    # Load tokenizer & model
    print("Loading pre-trained DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=max(labels) + 1,
        problem_type="single_label_classification"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Prepare DataLoaders
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset   = EmotionDataset(val_texts,   val_labels,   tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size)
    
    # Optimizer & scheduler with warmup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = max(1, total_steps // 20)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Scheduler: {warmup_steps} warmup steps out of {total_steps} total")
    
    best_accuracy = 0
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            
            # Use class-weighted loss
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels_batch)
            
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_dataloader)} Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done in {epoch_time:.1f}s, Avg Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        correct_preds = 0
        total_eval_loss = 0
        
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                logits = outputs.logits
                # same weighted loss on validation
                loss = torch.nn.CrossEntropyLoss(weight=class_weights)(logits, labels_batch)
                
                total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_preds += (preds == labels_batch).sum().item()
        
        avg_val_loss = total_eval_loss / len(val_dataloader)
        accuracy = correct_preds / len(val_dataset)
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model (Accuracy: {accuracy:.4f}) to {output_dir}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} min, Best Accuracy: {best_accuracy:.4f}")
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion detection model")
    parser.add_argument("--data_path", type=str, default="data/goemotions.csv")
    parser.add_argument("--output_dir", type=str, default="models/emotion_detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_emotion_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )



# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
# from transformers import get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os
# import argparse
# import time

# class EmotionDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
        
#         encoding = self.tokenizer(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors="pt"
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(label, dtype=torch.long)
#         }

# def train_emotion_model(data_path, output_dir, epochs=3, batch_size=16, learning_rate=5e-5):
#     """
#     Fine-tune a DistilBERT model on the GoEmotions dataset
#     """
#     # Start timing
#     start_time = time.time()
    
#     # Check if data exists
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")
    
#     print(f"Loading dataset from {data_path}")
    
#     # Load data
#     df = pd.read_csv(data_path)
    
#     # Prepare labels and texts
#     texts = df['text'].tolist()
    
#     # Assuming 'label' column contains the emotion index
#     # If the dataset has one-hot encoded emotions, convert to indices
#     if 'label' in df.columns:
#         labels = df['label'].tolist()
        
#         # If labels are strings, convert to integers
#         if isinstance(labels[0], str):
#             # Get unique labels
#             unique_labels = sorted(list(set(labels)))
#             label_map = {label: i for i, label in enumerate(unique_labels)}
#             labels = [label_map[label] for label in labels]
            
#             print(f"Converted string labels to integers. Found {len(unique_labels)} unique labels.")
#     else:
#         # Assume one-hot encoding for emotions in separate columns
#         emotion_columns = [col for col in df.columns if col != 'text']
#         labels = df[emotion_columns].values.argmax(axis=1).tolist()
    
#     print(f"Dataset loaded with {len(texts)} samples")
    
#     # Split dataset
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         texts, labels, test_size=0.1, random_state=42
#     )
    
#     print(f"Training set: {len(train_texts)} samples")
#     print(f"Validation set: {len(val_texts)} samples")
    
#     # Load tokenizer and model
#     print("Loading pre-trained DistilBERT model...")
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertForSequenceClassification.from_pretrained(
#         'distilbert-base-uncased',
#         num_labels=max(labels) + 1,  # Number of unique labels
#         problem_type="single_label_classification"
#     )
    
#     # Prepare datasets
#     print("Preparing datasets...")
#     train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
#     val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
#     # DataLoaders
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # Training setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model.to(device)
    
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     total_steps = len(train_dataloader) * epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, 
#         num_warmup_steps=0,
#         num_training_steps=total_steps
#     )
    
#     # Training loop
#     best_accuracy = 0
    
#     print(f"Starting training for {epochs} epochs...")
    
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         total_train_loss = 0
#         epoch_start_time = time.time()
        
#         for batch_idx, batch in enumerate(train_dataloader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
            
#             model.zero_grad()
            
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
            
#             loss = outputs.loss
#             total_train_loss += loss.item()
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
#             optimizer.step()
#             scheduler.step()
            
#             # Print progress
#             if (batch_idx + 1) % 100 == 0:
#                 print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
        
#         avg_train_loss = total_train_loss / len(train_dataloader)
#         epoch_time = time.time() - epoch_start_time
#         print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - Avg. training loss: {avg_train_loss:.4f}")
        
#         # Validation
#         model.eval()
#         correct_predictions = 0
#         total_eval_loss = 0
        
#         for batch in val_dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
            
#             with torch.no_grad():
#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )
                
#                 loss = outputs.loss
#                 logits = outputs.logits
                
#                 total_eval_loss += loss.item()
                
#                 preds = torch.argmax(logits, dim=1)
#                 correct_predictions += (preds == labels).sum().item()
        
#         avg_val_loss = total_eval_loss / len(val_dataloader)
#         accuracy = correct_predictions / len(val_dataset)
        
#         print(f"Validation Loss: {avg_val_loss:.4f}")
#         print(f"Accuracy: {accuracy:.4f}")
        
#         # Save best model
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
            
#             # Create output directory if it doesn't exist
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
                
#             print(f"Saving best model to {output_dir}")
#             model.save_pretrained(output_dir)
#             tokenizer.save_pretrained(output_dir)
            
#             # Save label mapping
#             if isinstance(df['label'].iloc[0], str):
#                 with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
#                     import json
#                     json.dump(label_map, f)
    
#     total_time = time.time() - start_time
#     print(f"Training completed in {total_time:.2f}s")
#     print(f"Best validation accuracy: {best_accuracy:.4f}")
    
#     return model, tokenizer


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train emotion detection model")
#     parser.add_argument("--data_path", type=str, default="data/goemotions.csv", help="Path to the dataset")
#     parser.add_argument("--output_dir", type=str, default="models/emotion_detector", help="Directory to save the model")
#     parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
#     parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
    
#     train_emotion_model(
#         data_path=args.data_path,
#         output_dir=args.output_dir,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         learning_rate=args.learning_rate
#     )




# # import torch
# # from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
# # from transformers import get_linear_schedule_with_warmup
# # from torch.utils.data import DataLoader, Dataset
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # import os

# # class EmotionDataset(Dataset):
# #     def __init__(self, texts, labels, tokenizer, max_length=128):
# #         self.texts = texts
# #         self.labels = labels
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length
    
# #     def __len__(self):
# #         return len(self.texts)
    
# #     def __getitem__(self, idx):
# #         text = self.texts[idx]
# #         label = self.labels[idx]
        
# #         encoding = self.tokenizer(
# #             text,
# #             add_special_tokens=True,
# #             max_length=self.max_length,
# #             return_token_type_ids=False,
# #             padding="max_length",
# #             truncation=True,
# #             return_attention_mask=True,
# #             return_tensors="pt"
# #         )
        
# #         return {
# #             'input_ids': encoding['input_ids'].flatten(),
# #             'attention_mask': encoding['attention_mask'].flatten(),
# #             'label': torch.tensor(label, dtype=torch.long)
# #         }

# # def train_emotion_model(data_path, output_dir, epochs=3, batch_size=16):
# #     """
# #     Fine-tune a DistilBERT model on the GoEmotions dataset
# #     """
# #     # Load data
# #     df = pd.read_csv(data_path)
    
# #     # Prepare labels and texts
# #     texts = df['text'].tolist()
    
# #     # Assuming 'label' column contains the emotion index
# #     # If the dataset has one-hot encoded emotions, convert to indices
# #     if 'label' in df.columns:
# #         labels = df['label'].tolist()
# #     else:
# #         # Assume one-hot encoding for emotions in separate columns
# #         emotion_columns = [col for col in df.columns if col != 'text']
# #         labels = df[emotion_columns].values.argmax(axis=1).tolist()
    
# #     # Split dataset
# #     train_texts, val_texts, train_labels, val_labels = train_test_split(
# #         texts, labels, test_size=0.1, random_state=42
# #     )
    
# #     # Load tokenizer and model
# #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# #     model = DistilBertForSequenceClassification.from_pretrained(
# #         'distilbert-base-uncased',
# #         num_labels=28,  # 27 emotions + neutral
# #         problem_type="single_label_classification"
# #     )
    
# #     # Prepare datasets
# #     train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
# #     val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
# #     # DataLoaders
# #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# #     val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
# #     # Training setup
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)
    
# #     optimizer = AdamW(model.parameters(), lr=5e-5)
# #     total_steps = len(train_dataloader) * epochs
# #     scheduler = get_linear_schedule_with_warmup(
# #         optimizer, 
# #         num_warmup_steps=0,
# #         num_training_steps=total_steps
# #     )
    
# #     # Training loop
# #     best_accuracy = 0
    
# #     for epoch in range(epochs):
# #         model.train()
# #         total_train_loss = 0
        
# #         for batch in train_dataloader:
# #             input_ids = batch['input_ids'].to(device)
# #             attention_mask = batch['attention_mask'].to(device)
# #             labels = batch['label'].to(device)
            
# #             model.zero_grad()
            
# #             outputs = model(
# #                 input_ids=input_ids,
# #                 attention_mask=attention_mask,
# #                 labels=labels
# #             )
            
# #             loss = outputs.loss
# #             total_train_loss += loss.item()
            
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
# #             optimizer.step()
# #             scheduler.step()
        
# #         avg_train_loss = total_train_loss / len(train_dataloader)
# #         print(f"Epoch {epoch+1}/{epochs} - Avg. training loss: {avg_train_loss:.4f}")
        
# #         # Validation
# #         model.eval()
# #         correct_predictions = 0
# #         total_eval_loss = 0
        
# #         for batch in val_dataloader:
# #             input_ids = batch['input_ids'].to(device)
# #             attention_mask = batch['attention_mask'].to(device)
# #             labels = batch['label'].to(device)
            
# #             with torch.no_grad():
# #                 outputs = model(
# #                     input_ids=input_ids,
# #                     attention_mask=attention_mask,
# #                     labels=labels
# #                 )
                
# #                 loss = outputs.loss
# #                 logits = outputs.logits
                
# #                 total_eval_loss += loss.item()
                
# #                 preds = torch.argmax(logits, dim=1)
# #                 correct_predictions += (preds == labels).sum().item()
        
# #         avg_val_loss = total_eval_loss / len(val_dataloader)
# #         accuracy = correct_predictions / len(val_dataset)
        
# #         print(f"Validation Loss: {avg_val_loss:.4f}")
# #         print(f"Accuracy: {accuracy:.4f}")
        
# #         # Save best model
# #         if accuracy > best_accuracy:
# #             best_accuracy = accuracy
            
# #             # Create output directory if it doesn't exist
# #             if not os.path.exists(output_dir):
# #                 os.makedirs(output_dir)
                
# #             print(f"Saving best model to {output_dir}")
# #             model.save_pretrained(output_dir)
# #             tokenizer.save_pretrained(output_dir)
    
# #     return model, tokenizer

# # if __name__ == "__main__":
# #     # Example usage
# #     train_emotion_model(
# #         data_path="path/to/goemotions.csv",
# #         output_dir="models/emotion_detector"
# #     )


# # # from datasets import load_dataset
# # # from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
# # # import torch

# # # dataset = load_dataset("go_emotions")
# # # train_data = dataset['train']
# # # test_data = dataset['test']

# # # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# # # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=28)

# # # def tokenize(example):
# # #     return tokenizer(example['text'], padding="max_length", truncation=True)

# # # tokenized_train = train_data.map(tokenize, batched=True)
# # # tokenized_test = test_data.map(tokenize, batched=True)

# # # tokenized_train = tokenized_train.rename_column("labels", "label")
# # # tokenized_train.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
# # # tokenized_test.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

# # # training_args = TrainingArguments(
# # #     output_dir="./emotion_detector/model",
# # #     evaluation_strategy="epoch",
# # #     save_strategy="epoch",
# # #     per_device_train_batch_size=16,
# # #     per_device_eval_batch_size=64,
# # #     num_train_epochs=3,
# # #     weight_decay=0.01,
# # # )

# # # trainer = Trainer(
# # #     model=model,
# # #     args=training_args,
# # #     train_dataset=tokenized_train,
# # #     eval_dataset=tokenized_test,
# # # )

# # # trainer.train()
# # # model.save_pretrained('./emotion_detector/model')
# # # tokenizer.save_pretrained('./emotion_detector/model')
