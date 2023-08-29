import os
import argparse
os.environ["HF_HOME"] = r"./.cache"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_count, trainable_bytes = 0, 0
    total_count, total_bytes = 0, 0
    for _, param in model.named_parameters():
        total_count += param.nelement()
        total_bytes += param.nelement() * param.element_size()
        if param.requires_grad:
            trainable_count += param.nelement()
            trainable_bytes += param.nelement() * param.element_size()
    print(
        f"Total params: {total_count:12,} ({(total_bytes / 1024**2):7,.1f}MB) | "
        f"Trainable params: {trainable_count:12,} ({(trainable_bytes / 1024**2):7,.1f}MB) [{100 * trainable_count / total_count:3.1f}%]"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_mbart',
        description='Run training for mBART with fixed hyperparameters',
    )
    parser.add_argument('-l', '--source-language', choices=["en", "ja"], type=str, help='source language')
    parser.add_argument('-d', '--dataset-name', choices=["mixed-500k", "mixed-250k+bt-250k", "news-250k"], type=str, help='dataset name')
    parser.add_argument('--resume', default=True, action=argparse.BooleanOptionalAction, help="resume training from checkpoint (default: True)")
    args = parser.parse_args()
    
    SOURCE_LANG = args.source_language
    TARGET_LANG = "ja" if SOURCE_LANG == "en" else "en"
    DATASET_NAME = args.dataset_name
    RESUME = args.resume
    
    from utils.dataset import EnJaDatasetMaker
    from utils.metric import SacreBleu

    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, \
        GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

    if SOURCE_LANG == "en":
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ja_XX")
    else:
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ja_XX", tgt_lang="en_XX")
      

    modules_to_save = ["final_layer_norm", "self_attn_layer_norm", "layer_norm", "layernorm_embedding", "embed_positions"]
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "shared", "lm_head"]
    config = LoraConfig(
        r=8,
        lora_alpha=8, # anything is fine (simply tune lr)
        lora_dropout=0.1,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="lora_only",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=lora_model)
    
    dataset = EnJaDatasetMaker.load_dataset(f"{SOURCE_LANG}-{TARGET_LANG}-{DATASET_NAME}")
    train_data = dataset["train"].remove_columns(["source", "target"])
    valid_data = dataset["valid"].remove_columns(["source", "target"])
    
    
    compute_metrics = SacreBleu.get_mBART_metric(tokenizer=tokenizer, target_language=TARGET_LANG)
        
    def set_decoder_configuration(gc: GenerationConfig):
        gc.no_repeat_ngram_size = 4
        gc.length_penalty = 2.0
        gc.num_beams = 3
        #gen_config.max_new_tokens = MAX_LENGHT
        gc.max_length = 256
        gc.min_length = 0
        gc.early_stopping = True
        # pad token is set to eos since in GPT2 pad does not exist
        gc.pad_token_id = tokenizer.eos_token_id
        gc.bos_token_id = tokenizer.bos_token_id
        gc.eos_token_id = tokenizer.eos_token_id
        return gc

    gen_config = GenerationConfig()
    gen_config = set_decoder_configuration(gen_config)
    
    train_args = Seq2SeqTrainingArguments(
        report_to="wandb",
        run_name=f"{SOURCE_LANG}-{TARGET_LANG}-{DATASET_NAME}",
        num_train_epochs=3,

        logging_strategy="steps",
        logging_steps=1, # * 4, 2, 1

        evaluation_strategy="steps",
        eval_steps=1250, # * 20_000, 10_000, 5_000
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_config=gen_config,

        output_dir="./.ckp/",
        save_strategy="steps",
        save_steps=1250, # * 20_000, 10_000, 5_000
        save_total_limit=100,
        load_best_model_at_end=True, # defaults to metric: "loss"
        metric_for_best_model="eval_score",
        greater_is_better=True,

        optim="adamw_torch",
        warmup_steps=400, # 3500, 1750, 875
        learning_rate=5e-5, # 3e-5, 5e-5
        bf16=True, # bf16, qint 8 ???
        
        group_by_length=True,
        length_column_name="length",

        # torch_compile=True,
        label_smoothing_factor=0.2, # 0.1, 0.2
        
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, # * 1, 2, 4
        gradient_checkpointing=True,
        # eval_accumulation_steps=4, # ???
    )
    
    trainer = Seq2SeqTrainer(
        lora_model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics
    )

    lora_model.train()
    trainer.train(resume_from_checkpoint=RESUME) # resume_from_checkpoint=True