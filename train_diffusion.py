import torch
from models.diffusion_embedding import DiffusionEmbedding

def treinar_diffusion_model(embeddings, embedding_dim, epochs=100, batch_size=32, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionEmbedding(embedding_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            optimizer.zero_grad()
            noisy_embeddings = model.encode(batch)
            reconstructed = model.decode(noisy_embeddings)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(embeddings):.4f}")
    
    return model