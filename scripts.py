from src import ColBERT

x = ColBERT()
x.load_data()
print("data loaded")
x.load_passages()
print("passages loaded")
x.setup_model()
print("model setup")
x.model_finetune()
print("model finetune")

#x.index()
#print("index setup")