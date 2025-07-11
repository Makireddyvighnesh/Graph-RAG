from transformers import pipeline

generator = pipeline("text2text-generation", model="facebook/bart-large")

def generate_answer(question, context):
    prompt = f"Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]['generated_text']