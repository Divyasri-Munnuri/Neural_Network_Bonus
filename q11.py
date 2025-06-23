from transformers import pipeline

qa_pipeline = pipeline("question-answering", framework="pt")  # "pt" = PyTorch

context = "Hugging Face is a company that creates tools for natural language processing."
question = "What does Hugging Face do?"

result = qa_pipeline(question=question, context=context)
print(result)
