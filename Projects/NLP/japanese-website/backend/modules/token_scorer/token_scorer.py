import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

class TokenScorer:
    def __init__(self, language="en"):
        if language == "en":
            self.model_name = "bert-base-uncased"
        elif language == "ja":
            self.model_name = "cl-tohoku/bert-base-japanese-v2"
        else:
            raise ValueError("Language not supported. Use 'en' or 'ja'.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()

    def score_tokens(self, sentence, top_k=5):
        encoded = self.tokenizer(sentence, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        results = []

        for i in range(1, len(input_ids) - 1):  # skip [CLS] and [SEP]
            masked_input = input_ids.clone()
            masked_input[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input.unsqueeze(0))
                logits = outputs.logits[0, i]

            probs = torch.softmax(logits, dim=0)
            token_id = input_ids[i]
            token_prob = probs[token_id].item()
            token_score = torch.log(probs[token_id] + 1e-10).item()

            top_ids = torch.topk(probs, top_k).indices
            top_tokens = [self.tokenizer.convert_ids_to_tokens([tid])[0] for tid in top_ids]
            top_probs = [probs[tid].item() for tid in top_ids]

            results.append({
                "position": i,
                "token": tokens[i],
                "score": token_score,
                "probability": token_prob,
                "top_alternatives": list(zip(top_tokens, top_probs))
            })

        return results

if __name__== '__main__':
    # Esempio in inglese
    scorer = TokenScorer(language="en")
    sentence = "She go to the market yesterday."
    results = scorer.score_tokens(sentence)

    for r in results:
        print(f"Token: {r['token']} | Score: {r['score']:.4f} | Prob: {r['probability']:.4f}")
        print(f"Top alternatives: {r['top_alternatives']}\n")

    scorer = TokenScorer(language='ja')
    sentence = "ラーメンがいかがですか"
    results = scorer.score_tokens(sentence)

    for r in results:
        print(f"Token: {r['token']} | Score: {r['score']:.4f} | Prob: {r['probability']:.4f}")
        print(f"Top alternatives: {r['top_alternatives']}\n")
