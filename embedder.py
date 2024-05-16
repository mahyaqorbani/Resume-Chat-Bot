import os
import torch
import pandas as pd
from pdfminer.high_level import extract_text
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class BERTEmbedder:
    # model_name specifies which pretrained model we'd like to use
    # we can test out a few if we'd like
    def __init__(self, model_name='bert-base-uncased'):
        # pretrained bert embedding
        # if embedding takes too long locally, we can switch to colab and use gpu runtime
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)

    def encode(self, text):
        # generate an embedding (single vector) for input text
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
    
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

def read_pdf_file(file_path):
    # data is in the form of pdf
    # so we need to extract the text from the pdf file
    # PDFminer provides a useful method: pdfminer.high_level.extract_text
    text = extract_text(file_path)
    return text

def main(directory, output_file):
    # generate one CSV file
    # each CSV contains all the embeddings of resumes for one realm of work
    embedder = BERTEmbedder()
    embeddings = []
    filenames = []

    # ex: for each accountant_resume in data/ACCOUNTANT
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            text = read_pdf_file(file_path) # pdf -> text
            embedding = embedder.encode(text)
            embeddings.append(embedding)
            filenames.append(filename)

    # store embeddings in CSV
    # later, we'll import embeddings as pandas df
    df = pd.DataFrame(embeddings, index=filenames)
    df.to_csv(output_file)

if __name__ == "__main__":
    # to generate embeddings, use the following command line arguments
    # python embedder.py [path-to-specific-resumes] [output-csv-name].csv
    # ex: python embedder.py data/ACCOUNTANT accountant-embeddings.csv
    import sys
    directory = sys.argv[1]
    output_file = sys.argv[2]
    main(directory, output_file)
