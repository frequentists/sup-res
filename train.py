from src import ColBERT

if __name__ == "__main__":
    x = ColBERT()
    x.load_data()
    print("data loaded")
    x.load_passages()
    print("passages loaded")
    x.setup_model(train=True)
    print("model setup and trained")