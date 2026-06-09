import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
from src.utils.logging_utils import get_logger

logger = get_logger("core.model_loader")


def _resolve_dtype(dtype_name):
    """Resolve a config dtype string into a torch dtype."""
    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    normalized = str(dtype_name).strip().lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    return dtype_map[normalized]


def load_4bit_engine(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    *,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    attn_implementation="sdpa",
    device_map="auto",
):
    logger.info("Loading model: %s...", model_id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    compute_dtype = _resolve_dtype(bnb_4bit_compute_dtype)
    bnb_config = None
    if load_in_4bit and BitsAndBytesConfig is not None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    elif load_in_4bit:
        logger.warning(
            "bitsandbytes unavailable; loading model with dtype=%s without quantization.",
            compute_dtype,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs = {
            "device_map": device_map,
            "dtype": compute_dtype,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

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
