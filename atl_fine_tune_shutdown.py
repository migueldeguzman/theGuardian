import os
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_linear_schedule_with_warmup

class GPT2Assistant:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.model = None

    def fine_tune(self, answer_file_path, model_output_dir, epochs=1.0, overwrite=False):
        if self.model is None or overwrite:
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        else:
            self.model = self.model.to(torch.device("cpu"))  # Move model to CPU before loading new weights
            self.model = GPT2LMHeadModel.from_pretrained(model_output_dir)
            self.model = self.model.to(torch.device("cuda"))  # Move model back to GPU if available

        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=answer_file_path,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        total_steps = len(train_dataset) * epochs
        warmup_steps = 0.1 * total_steps

        optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-5, weight_decay=0.013)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            weight_decay=0.013,
            gradient_accumulation_steps=4,
            learning_rate=8e-5,
            lr_scheduler_type='cosine',
            warmup_steps=500
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            optimizers=(optimizer, scheduler)
        )

        trainer.train()
        self.model.save_pretrained(model_output_dir)
        self.tokenizer.save_pretrained(model_output_dir)

    def generate_answer(self, prompt, max_length=2000):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer[len(prompt):]

    def query(self, prompt):
        generated_answer = self.generate_answer(prompt)
        print(generated_answer)
        return generated_answer

def main():
    text_file_path = "/Users/migueldeguzmandev/Desktop/V4_Guardian/guardian.text"
    model_output_dir = "/Users/migueldeguzmandev/Desktop/V4_Guardian/"

    assistant = GPT2Assistant()

    choice = input("Do you want to fine-tune a new model (n), fine-tune an existing model (f), or load an existing one (e)? (n/f/e): ")

    if choice.lower() == "n":
        print("Fine-tuning the model...")
        assistant.fine_tune(text_file_path, model_output_dir)
        print("Model fine-tuning complete.")
    elif choice.lower() == "f":
        print("Fine-tuning the existing model...")
        assistant.fine_tune(text_file_path, model_output_dir, overwrite=True)
        print("Model fine-tuning complete.")
    elif choice.lower() == "e":
        print("Loading the existing model...")
        assistant.model = GPT2LMHeadModel.from_pretrained(model_output_dir)
        print("Existing model loaded.")
    else:
        print("Invalid choice. Exiting the program.")
        sys.exit()

    while True:
        prompt = input("Enter your question (or type 'exit' to stop): ")
        if prompt.lower() == "exit":
            break

        print("Answering in progress...")
        generated_answer = assistant.query(prompt)

        print("\n")

if __name__ == "__main__":
    main()
