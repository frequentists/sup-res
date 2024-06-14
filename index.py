from src import ColBERT

if __name__ == "__main__":
    x = ColBERT()
    x.load_data()
    x.load_passages()
    x.setup_model(checkpoint=".ragatouille/colbert/none/2024-06/12/03.14.27/checkpoints/colbert")
    print("model setup")
    x.index()
    print("model indexed")
    x.recall()
    print("test")