"""Fine-tune RoBERTa for binary hallucination classification.

CSV files need text and label columns.
"""
import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("train_csv"); parser.add_argument("validation_csv")
    parser.add_argument("--model",default="FacebookAI/roberta-base"); parser.add_argument("--output",default="models/roberta-hallucination")
    args=parser.parse_args()
    dataset=load_dataset("csv",data_files={"train":args.train_csv,"validation":args.validation_csv})
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    tokenized=dataset.map(lambda batch:tokenizer(batch["text"],truncation=True,max_length=512),batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2)
    config=TrainingArguments(output_dir=args.output,learning_rate=2e-5,per_device_train_batch_size=8,
        per_device_eval_batch_size=16,num_train_epochs=3,eval_strategy="epoch",save_strategy="epoch",
        load_best_model_at_end=True,report_to="none")
    Trainer(model=model,args=config,train_dataset=tokenized["train"],eval_dataset=tokenized["validation"],
            processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer)).train()
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
if __name__=="__main__": main()
