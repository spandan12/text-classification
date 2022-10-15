import tqdm
import torch


def create_embedding(df, tokenizer, model, device):
    embeddings = list()
    for _, (text, _, _, _) in tqdm.tqdm(df.iterrows(), total=len(df)):
        token = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        token_train = {k: v.clone().detach().to(device) for k, v in token.items()}
        with torch.no_grad():
            text_embed_full = model(**token_train)
        text_embed_cls = text_embed_full.last_hidden_state[:, 0, :]
        text_embed_cls = text_embed_cls.clone().detach()
        embeddings.append(text_embed_cls)

    return torch.cat(embeddings).clone().detach().cpu().numpy()
