# Resume Chatbot
In this paper, we evaluate the effectiveness of basic natural language processing techniques in the creation
of a simple pipeline, which takes as input a job description, and aims to output the five most relevant resumes. With the increasing use of automated candidate screening in the recruitment process, our team aimed to experiment with different NLP approaches in order to see if we could implement a successful pipeline which would retrieve ideal candidates for a given job. The most successful approach we attempted included embedding the resumes within our dataset as well as the job description, utilizing a BERT model to classify the job description to its closest matching profession, and utilizing cosine similarity to retrieve the most similar embedded resumes in high-dimensional vector space. However, this approach has its limitations in its effectiveness, which are further examined within our report.

*Evaluating the effectiveness of pretrained BERT models on generating embeddings which deliver contextually relevant results when compared with an embedded user query. <br />  <br />
*Utlizing our own trained model to match the job description to a job profession, executing semantic search amongst only these resumes. <br />  <br />
*To generate embeddings, utilize the following command: <br />
`python embedder.py [path-to-data-folder] [name-of-output-file].csv` <br />
For example: `python embedder.py data/ACCOUNTANT accountant-embeddings.csv` <br /> <br />

*Link to Kaggle Resume dataset used: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset?resource=download <br /> <br />

Project structure represented as file tree: <br />

project_root/ <br />
│
├── data/ <br />
│   ├── ACCOUNTANT/ <br />
│   ├── ADVOCATE/ <br />
│   │   ... (other directories for different professions) <br />
│   └── TEACHER/ <br />
│ <br />
├── embeddings/ <br />
│   ├── bert-base-uncased-embeddings/ <br />
│   │   ├── accountant-embeddings.csv <br />
│   │   ├── advocate-embeddings.csv <br />
│   │   │   ... (other CSV files for other professions) <br />
│   │   └── teacher-embeddings.csv <br />
│ <br />
└── embedder.py <br />

