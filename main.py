import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(embedding_size, nhead=8, num_encoder_layers=num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, src):
        src = self.embedding(src)
        out = self.transformer(src, src)
        out = self.fc(out)
        return out

# Open the text file and read the training data conversations
with open("training_data.txt", "r") as file:
    conversations = file.read().split("\n\n")  # Split conversations by double newline

# Preprocessing: Create vocabulary and convert text to numerical data
word_to_index = {}
index_to_word = {}
index = 0  # Initialize index

data = []  # List to store alternating prompts and replies
for conversation in conversations:
    lines = conversation.split("\n")
    for i in range(0, len(lines), 2):
        user_prompt = lines[i].replace("User: ", "")
        ai_reply = lines[i + 1].replace("AI: ", "")
        data.append(user_prompt)
        data.append(ai_reply)

        # Process words in user prompt and AI reply
        for word in user_prompt.split() + ai_reply.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(index_to_word)] = word

# Manually assign an index to the period symbol
word_to_index["."] = len(word_to_index)
index_to_word[len(index_to_word)] = "."


# Batching the data with padding
X_batches = []

batch_size = 4
num_batches = (len(data) + batch_size - 1) // batch_size

max_sentence_length = max(len(sentence.split()) for sentence in data)  # Find the max sentence length

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_data = data[start_idx:end_idx]

    batch_X = []
    for sentence in batch_data:
        sentence_indices = [word_to_index[word] for word in sentence.split()]
        # Pad the sentence to the max length
        padded_sentence_indices = sentence_indices + [word_to_index["."]] * (max_sentence_length - len(sentence_indices))
        batch_X.append(torch.tensor(padded_sentence_indices))

    batch_X = nn.utils.rnn.pad_sequence(batch_X, batch_first=True)
    X_batches.append(batch_X)

X_batches = torch.stack(X_batches)

vocab_size = len(word_to_index)
embedding_size = 256
num_layers = 10

device = torch.device("cuda")  # Use CPU

model = GPT(vocab_size, embedding_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

accumulation_steps = 8
clip_value = 1.0

for epoch in range(200):
    total_loss = 0
    accumulated_loss = 0
    model.train()
    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        batch_X = X_batches[batch_idx].to(device)
        targets = batch_X.clone()
        targets[:, :-1] = batch_X[:, 1:]
        outputs = model(batch_X)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss /= accumulation_steps
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value)
        accumulated_loss += loss.item()
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Move scheduler step here
            total_loss += accumulated_loss
            accumulated_loss = 0

    print("Epoch {}: average loss={}".format(epoch, total_loss / num_batches))

    # Generate Text
    def generate_text(model, conversation_history, max_length=50):
        conversation_context = "\n".join(conversation_history)
        conversation_context = conversation_context.replace("User:", "").replace("AI:", "")

        prompt_tensor = torch.tensor([word_to_index[word] for word in conversation_context.split()]).to(device)

        output = torch.tensor([], dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(max_length):
                next_word_logits = model(prompt_tensor.unsqueeze(0).to(device))[-1]
                next_word_probs = torch.softmax(next_word_logits, dim=-1)
                next_word_probs = next_word_probs.view(-1)
                next_word_index = torch.multinomial(next_word_probs, 1).item()

                if next_word_index in index_to_word:
                    next_word = index_to_word[next_word_index]
                    output = torch.cat((output, torch.tensor([next_word_index], device=device)))

                    if next_word == ".":
                        break
                else:
                    break

        generated_text = " ".join([index_to_word[index.item()] for index in output])
        return generated_text

# Chat loop
# Print model parameters
print(f"Model structure: {model}")

# Print number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_params:,} total trainable parameters")
print("AI: Hello! How can I assist you today?")
conversation_history = []  # Keep track of conversation history

# Move initial conversation history to the same device
conversation_history = [history.to(device) for history in conversation_history]


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break

    # Convert user input to tensor and move to the device
    user_input_tensor = torch.tensor([word_to_index[word] for word in user_input.split()]).to(device)

    conversation_history.append("User: " + user_input)

    generated_response = generate_text(model, conversation_history)

    print("AI:", generated_response)
    conversation_history.append("AI: " + generated_response)