import os
import shutil
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
    ORTQuantizer
)
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to your local folders OR Hugging Face repo IDs
BASE_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LORA_ADAPTER_PATH = "./embedding_model"      # or local path
CROSS_ENCODER_PATH = "./cross_encoder_yoruba_stem"  # or local path

# Output directories
MERGED_BI_DIR = "./temp_merged_bi_encoder"
ONNX_BI_DIR = "./models/bi_encoder_onnx_int8"
ONNX_CE_DIR = "./models/cross_encoder_onnx_int8"

def merge_lora_for_export():
    print("1. Merging LoRA weights into Base Model for Bi-Encoder...")
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModel.from_pretrained(BASE_MODEL_ID)
    
    # Apply LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # Merge weights permanently and unload PEFT overhead
    merged_model = peft_model.merge_and_unload()
    
    # Save temporarily as a standard Hugging Face model
    merged_model.save_pretrained(MERGED_BI_DIR)
    tokenizer.save_pretrained(MERGED_BI_DIR)
    print(f"   Merged model saved to {MERGED_BI_DIR}")

def export_and_quantize_bi_encoder():
    print("\n2. Exporting Bi-Encoder to ONNX...")
    # Load the merged model and trace it to ONNX format
    ort_model = ORTModelForFeatureExtraction.from_pretrained(MERGED_BI_DIR, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_BI_DIR)
    
    # Save the FP32 ONNX model temporarily
    temp_fp32_dir = "./temp_bi_fp32"
    ort_model.save_pretrained(temp_fp32_dir)
    tokenizer.save_pretrained(temp_fp32_dir)
    
    print("3. Quantizing Bi-Encoder to INT8...")
    quantizer = ORTQuantizer.from_pretrained(temp_fp32_dir)
    
    # Dynamic quantization is standard for NLP Transformer encoders
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    
    # Save final quantized model
    quantizer.quantize(save_dir=ONNX_BI_DIR, quantization_config=dqconfig)
    tokenizer.save_pretrained(ONNX_BI_DIR)
    print(f"   INT8 Bi-Encoder saved to {ONNX_BI_DIR}")
    
    # Cleanup temp
    shutil.rmtree(temp_fp32_dir)

def export_and_quantize_cross_encoder():
    print("\n4. Exporting Cross-Encoder to ONNX...")
    # Load Cross-Encoder and trace to ONNX
    ort_model = ORTModelForSequenceClassification.from_pretrained(CROSS_ENCODER_PATH, export=True)
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_PATH)
    
    temp_fp32_dir = "./temp_ce_fp32"
    ort_model.save_pretrained(temp_fp32_dir)
    tokenizer.save_pretrained(temp_fp32_dir)
    
    print("5. Quantizing Cross-Encoder to INT8...")
    quantizer = ORTQuantizer.from_pretrained(temp_fp32_dir)
    dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    
    # Save final quantized model
    quantizer.quantize(save_dir=ONNX_CE_DIR, quantization_config=dqconfig)
    tokenizer.save_pretrained(ONNX_CE_DIR)
    print(f"   INT8 Cross-Encoder saved to {ONNX_CE_DIR}")
    
    # Cleanup temp
    shutil.rmtree(temp_fp32_dir)

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    
    # Execute Pipeline
    merge_lora_for_export()
    export_and_quantize_bi_encoder()
    export_and_quantize_cross_encoder()
    
    # Cleanup initial merge directory
    shutil.rmtree(MERGED_BI_DIR)
    
    print("\n Optimization Complete")