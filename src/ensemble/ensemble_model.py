from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from colbert.utils import DataState
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_models import (
    SequenceClassificationDataset,
    evaluate_epoch,
    train_model,
)



class WeightedEnsemble(nn.Module):
    def __init__(self, colbert_passages, classifer, state: DataState, output_size = 1):
        super(WeightedEnsemble, self).__init__()
        self.colbert_passages = RAGPretrainedModel.from_index(index_path=state.config["search_checkpoint"])

        train, dev, test, passage2labelid = load_data(
            dataframe="./data/top_" + str_label + ".csv.gz"
        )
        label2passageid = {i: j for j, i in passage2labelid.items()}
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )

        X_dev = dev.destination_context.tolist()
        y_dev = [passage2labelid[row.passage_id] for _, row in dev.iterrows()]

        devset = SequenceClassificationDataset(X_dev, y_dev, tokenizer)
        targets, outputs, probs = evaluate_epoch(model, devset, args)
        print("accuracy", accuracy_score(y_dev, outputs))
        predict_dataset = SequenceClassificationDatasetNoLabels(
            dev.destination_context.tolist(), tokenizer
        )
        self.classifer = 
        
        # Initialize learnable scalar weights
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        
        # Final linear layer to combine model outputs
        self.fc = nn.Linear(2 * output_size, output_size)
        
    def forward(self, x):
        # Get outputs from both models
        results = self.colbert_passages.search(state.test.destination_context.values,k=state.config["search"]["initial"] if state.config["search"]["rerank"] else 10,)
        
        out1 = self.model1(x)
        out2 = self.classifer(x)
        
        # Combine the outputs using learnable weights
        combined_out = torch.cat((self.w1 * out1, self.w2 * out2), dim=1)
        
        # Pass the combined output through the final linear layer
        final_out = self.fc(combined_out)
        
        # Apply softmax
        final_out = F.softmax(final_out, dim=1)
        
        return final_out