python3.10 -m venv sees
source sees/bin/activate

pip install torch==2.1.2 torchvision==0.16.2 transformers==4.37.1 accelerate==0.26.1 bitsandbytes==0.42.0 \
    huggingface-hub==0.20.3 matplotlib==3.7.4 Pillow==10.2.0 ipykernel==6.29.4 ipython==8.12.3 \
    bertviz==1.4.0 opencv-python==4.10.0.84 requests==2.32.3 numpy==1.26.4 scikit-learn==1.4.2 \
    hdbscan==0.8.33 umap-learn==0.5.6 pypdf==6.1.3

#!/bin/bash

# Find the installed transformers package path
TRANSFORMERS_PATH=$(python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

# Check if the path exists
if [ -d "$TRANSFORMERS_PATH" ]; then
    echo "Transformers package found at: $TRANSFORMERS_PATH"
    
    # Copy the custom files to their respective locations
    cp modeling_llava.py "$TRANSFORMERS_PATH/models/llava/modeling_llava.py"
    cp modeling_llama.py "$TRANSFORMERS_PATH/models/llama/modeling_llama.py"
    
    echo "Files copied successfully."
else
    echo "Error: Transformers package not found."
    exit 1
fi

echo "SETUP DONE"