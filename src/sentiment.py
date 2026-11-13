"""Score the tone of PubMed titles and abstracts with a frozen NLI classifier."""

MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
MODEL_REVISION = "613e8c52c33e2bc0677ada4ad760f693e5e0f581"
LABELS = ("negative", "neutral", "positive")
HYPOTHESES = (
    "The scientific text conveys a negative assessment.",
    "The scientific text conveys a neutral assessment.",
    "The scientific text conveys a positive assessment.",
)


class PubMedSentimentModel:
    """Return negative, neutral and positive scores for each article."""

    def __init__(self, model_path=None, batch_size=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        source = model_path or MODEL_ID
        kwargs = {} if model_path else {"revision": MODEL_REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, use_safetensors=True, **kwargs
        ).eval()
        self.entailment_id = self.model.config.label2id["entailment"]
        self.batch_size = batch_size
        self.hypotheses = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in HYPOTHESES
        ]
        self.chunk_size = (
            min(512, self.tokenizer.model_max_length, self.model.config.max_position_embeddings)
            - self.tokenizer.num_special_tokens_to_add(pair=True)
            - max(map(len, self.hypotheses))
        )

    def score(self, texts):
        """Average the softmax scores of all chunks, weighted by token count."""
        pairs = []
        chunks = []
        for article_index, article in enumerate(texts):
            tokens = self.tokenizer.encode(article, add_special_tokens=False, verbose=False)
            for start in range(0, len(tokens), self.chunk_size):
                chunk = tokens[start:start + self.chunk_size]
                chunks.append((article_index, len(chunk)))
                for hypothesis in self.hypotheses:
                    pairs.append({"input_ids": self.tokenizer.build_inputs_with_special_tokens(
                        chunk, hypothesis
                    )})

        logits = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                inputs = self.tokenizer.pad(
                    pairs[start:start + self.batch_size], return_tensors="pt"
                )
                logits.append(self.model(**inputs).logits[:, self.entailment_id])

        scores = self.torch.cat(logits).reshape(-1, len(LABELS)).softmax(dim=1).tolist()
        totals = [[0.0] * len(LABELS) for _ in texts]
        weights = [0] * len(texts)
        for (article_index, size), scores_for_chunk in zip(chunks, scores):
            weights[article_index] += size
            for label_index, score in enumerate(scores_for_chunk):
                totals[article_index][label_index] += size * score
        return [tuple(score / weight for score in total)
                for total, weight in zip(totals, weights)]
