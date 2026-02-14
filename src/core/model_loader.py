import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.utils.logging_utils import get_logger

logger = get_logger("core.model_loader")


def load_4bit_engine(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    logger.info("Loading model: %s...", model_id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        for param in model.parameters():
            param.requires_grad = False

        logger.info("Model loaded to: %s", model.device)
        return model, tokenizer

    except Exception as e:
        logger.error("Error: %s", e)
        return None, None


if __name__ == "__main__":
    model, tokenizer = load_4bit_engine()
    if model:
        mem = model.get_memory_footprint() / 1024**3
        logger.info("Used VRAM: %.2f GB", mem)
        logger.debug("Config: %s", model.config)
        prompt = "Test, hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        logger.debug("Input tokens: %s", input_tokens)

        outputs = model.generate(
            input_token,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )
        logger.info("Generation completed (top_k=50)")
        logger.debug("Outputs: %s", outputs[0])

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Generated text: %s", generated_text)
