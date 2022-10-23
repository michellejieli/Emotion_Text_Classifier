"""
After running zip it to save
zip -r summarizeApp.zip summarizeApp 
"""

from transformers import pipeline


#https://huggingface.co/j-hartmann/emotion-english-distilroberta-base?text=Oh+wow.+I+didn%27t+know+that.

model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)

model.save_pretrained("summarizeApp")
