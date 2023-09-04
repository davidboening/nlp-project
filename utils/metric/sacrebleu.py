import evaluate

class SacreBleu:
    
    def get_mBART_metric(*, tokenizer=None, target_language: str=None):
        assert tokenizer is not None and hasattr(tokenizer, "__call__"), "Object passed is not a valid tokenizer!"
        assert target_language is not None and target_language in ["en", "ja"], "Invalid language."
        metric = evaluate.load("sacrebleu")
        
        if target_language == "ja":
            def compute_metrics(preds):
                preds_ids, labels_ids = preds
                
                preds_ids[preds_ids == -100] = tokenizer.pad_token_id
                labels_ids[labels_ids == -100] = tokenizer.pad_token_id
                references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
                references = [[reference] for reference in references]

                predictions = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

                bleu_output = metric.compute(
                    references=references, 
                    predictions=predictions, 
                    tokenize="ja-mecab"
                )
                return bleu_output
            return compute_metrics
        else:
            def compute_metrics(preds):
                preds_ids, labels_ids = preds

                preds_ids[preds_ids == -100] = tokenizer.pad_token_id
                labels_ids[labels_ids == -100] = tokenizer.pad_token_id
                references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
                references = [[reference] for reference in references]

                predictions = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
                
                bleu_output = metric.compute(
                    references=references, 
                    predictions=predictions
                )
                return bleu_output  
            return compute_metrics