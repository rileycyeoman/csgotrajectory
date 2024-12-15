import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, Dataset
from data import DemoParser






model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)




class SplineLoss(nn.Module):
    def __init__(self, alpha=1.0, frame_interval=10):
        super(SplineLoss, self).__init__()
        self.alpha = alpha
        self.frame_interval = frame_interval
    
    def forward(self, true_frames, predicted_frames):
        batch_size, num_frames, frame_dims = true_frames.shape
        
        loss = 0.0
        
        for i in range(batch_size):
            true_frames_batch = true_frames[i].detach().cpu().numpy()  # Convert to numpy for spline fitting
            predicted_frames_batch = predicted_frames[i].detach().cpu().numpy()

            time_steps = np.arange(num_frames)
            true_spline = CubicSpline(time_steps, true_frames_batch, axis=0)
            predicted_spline = CubicSpline(time_steps, predicted_frames_batch, axis=0)


            sample_times = np.arange(0, num_frames, self.frame_interval)

            true_samples = true_spline(sample_times)  # Shape: (num_samples, frame_dims)
            predicted_samples = predicted_spline(sample_times)  # Shape: (num_samples, frame_dims)

            mse_loss = np.mean((true_samples - predicted_samples) ** 2)


            loss += mse_loss

        return torch.tensor(loss / batch_size, requires_grad=True).to(true_frames.device)
    

class MSEFrameLoss(nn.Module):
    def __init__(self, alpha=1.0, frame_interval=10):
        super(MSEFrameLoss, self).__init__()
        self.alpha = alpha
        self.frame_interval = frame_interval
        self.base_loss = nn.MSELoss(alpha=alpha)
    
    def forward(self, true_frames, predicted_frames):
        batch_size, num_frames, frame_dims = true_frames.shape
        
        loss = 0.0
        
        for i in range(batch_size):
            true_frames_batch = true_frames[i].detach().cpu().numpy()  # Convert to numpy for spline fitting
            predicted_frames_batch = predicted_frames[i].detach().cpu().numpy()

            sample_times = np.arange(0, num_frames, self.frame_interval)
            # TODO: comment out the to numpy and use torch.arange() instead?

            true_samples = true_frames_batch[sample_times, :]  # Shape: (num_samples, frame_dims)
            predicted_samples = predicted_frames_batch[sample_times, :]  # Shape: (num_samples, frame_dims)

            mse_loss = self.base_loss(true_samples, predicted_samples)

            loss += mse_loss

        return torch.tensor(loss / batch_size, requires_grad=True).to(true_frames.device)


def prepare_sequences(df, sequence_length):
    input_sequences = []
    output_sequences = []

    for i in range(len(df) - sequence_length):
        input_seq = df.iloc[i:i+sequence_length].values.flatten().tolist()
        output_seq = df.iloc[i+sequence_length].values.flatten().tolist()
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    
    return input_sequences, output_sequences

def tokenize_data(input_sequences, output_sequences, tokenizer, max_length=128):
    input_encodings = tokenizer(input_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    output_encodings = tokenizer(output_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Shift the labels for autoregressive language models (causal language modeling)
    labels = output_encodings['input_ids'].clone()
    return input_encodings, labels


class Seq2SeqDataset(Dataset):
    def __init__(self, input_encodings, labels):
        self.input_encodings = input_encodings
        self.labels = labels

    def __len__(self):
        return len(self.input_encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_encodings['input_ids'][idx],
            'attention_mask': self.input_encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        
input_sequences, output_sequences = prepare_sequences(lan_frames_df, sequence_length=3)  # Customize sequence length
input_encodings, labels = tokenize_data(input_sequences, output_sequences, tokenizer)


dataset = Seq2SeqDataset(input_encodings, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
spline_loss_fn = SplineLoss(alpha=0.5, frame_interval=10)
mse_loss_fn = MSEFrameLoss(alpha = 0.5, frame_interval=10)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.train()
for epoch in range(3): 
    for batch in train_loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        #loss = spline_loss_fn(labels, logits)

        criterion1 = spline_loss_fn(labels, logits)
        criterion2 = mse_loss_fn(labels, logits) #TODO: need to compare the output to the correct next frame...


        # Parameters for combined loss
        alpha = 0.5
        beta = 0.5

        # Calculate the combined loss
        def combined_loss(output, target):
            loss1 = criterion1(labels, logits)
            loss2 = criterion2(labels, logits)
            return alpha * loss1 + beta * loss2

        loss = combined_loss(labels, logits)


        loss.backward()


        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')



path = ''
parser = DemoParser(path)


parser.parse_demos()


parser.save_to_pickle('lan_frames_df.pkl')


lan_frames_df = parser.load_from_pickle('lan_frames_df.pkl')

print(parser.dataframe_shape())
print(parser.dataframe_head())


demoID = '0013db25-4444-452b-980b-7702dc6fb810'
roundNum = 1
filtered_demo = parser.demo_stats(demoID, roundNum)
print(filtered_demo)



    
model.eval()

def predict_next_cell(input_sequence, model, tokenizer):
    input_encodings = tokenizer(input_sequence, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_encodings['input_ids'], attention_mask=input_encodings['attention_mask'], max_length=50)
    predicted_cell = tokenizer.decode(output[0], skip_special_tokens=True)
    return predicted_cell

# Example: predict the next cell after an input sequence
input_sequence = lan_frames_df.iloc[:3].values.flatten().tolist()  # Example input sequence
predicted_cell = predict_next_cell(input_sequence, model, tokenizer)
print(f'Predicted next cell: {predicted_cell}')