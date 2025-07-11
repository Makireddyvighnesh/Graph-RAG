from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(question, context):
    prompt = f"Answer the following question using only the context:\n\nContext: {context}\n\nQuestion: {question}"
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text'].strip()
