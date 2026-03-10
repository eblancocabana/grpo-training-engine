import torch
import logging
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

def apply_flash_attn_varlen_monkey_patch():
    """
    Monkey-patches Qwen2 Attention modules to support 1D sequence packing
    via PyTorch flex_attention (native, no compilation required on PT >= 2.5).
    """
    if not HAS_FLEX:
        logging.getLogger("packing").warning("flex_attention not available. PyTorch 2.5+ required.")
        return

    from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention, Qwen2FlashAttention2, Qwen2Attention

    def get_packed_forward(original_forward):
        def packed_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_value = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: torch.LongTensor = None,
            **kwargs
        ):
            document_ids = kwargs.get("document_ids", None)

            if document_ids is None:
                return original_forward(
                    self, 
                    hidden_states=hidden_states, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    past_key_value=past_key_value, 
                    output_attentions=output_attentions, 
                    use_cache=use_cache, 
                    cache_position=cache_position, 
                    **kwargs
                )

            # 1D Packed execution path
            bsz, q_len, _ = hidden_states.size() # bsz should be 1
            
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            # We define a causal + document mask for flex_attention
            def document_causal_mask(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (document_ids[q_idx] == document_ids[kv_idx])

            block_mask = create_block_mask(document_causal_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, _compile=True)
            
            # Use flex_attention for variable length packing
            attn_output = flex_attention(query_states, key_states, value_states, block_mask=block_mask)

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        return packed_forward

    Qwen2SdpaAttention.forward = get_packed_forward(Qwen2SdpaAttention.forward)
    Qwen2FlashAttention2.forward = get_packed_forward(Qwen2FlashAttention2.forward)
    Qwen2Attention.forward = get_packed_forward(Qwen2Attention.forward)

def pack_sequences(input_ids, attention_mask, labels=None, pad_token_id=0):
    packed_ids = []
    packed_labels = [] if labels is not None else None
    position_ids = []
    document_ids = []
    batch_size = input_ids.shape[0]
    
    for i in range(batch_size):
        mask = attention_mask[i].bool()
        seq_ids = input_ids[i][mask]
        seq_len = seq_ids.shape[0]
        if seq_len == 0:
            continue
            
        packed_ids.append(seq_ids)
        position_ids.append(torch.arange(seq_len, dtype=torch.long, device=input_ids.device))
        document_ids.append(torch.full((seq_len,), i, dtype=torch.int32, device=input_ids.device))
        
        if labels is not None:
            packed_labels.append(labels[i][mask])
            
    packed_ids = torch.cat(packed_ids, dim=0).unsqueeze(0)
    position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)
    document_ids = torch.cat(document_ids, dim=0)
    
    if labels is not None:
        packed_labels = torch.cat(packed_labels, dim=0).unsqueeze(0)
        return packed_ids, position_ids, document_ids, packed_labels
        
    return packed_ids, position_ids, document_ids
