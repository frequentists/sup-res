from src import ColBERT

if __name__ == "__main__":
    x = ColBERT()
    x.load_data()
    print("data loaded")
    x.load_passages()
    print("passages loaded")
    x.setup_model(train=True)
    print("model setup and trained")

    #x.index()
    #print("index setup")

# sbatch --time=08:00 --mem-per-cpu=14G --gpus=1 --gres=gpumem:10G --mail-type=END --mail-user="" --wrap="python train.py"