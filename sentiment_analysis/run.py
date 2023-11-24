from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
from datetime import datetime

def split_columns_to_lists(input_file):
    # Read the CSV file into a DataFrame with explicit encoding
    df = pd.read_csv(input_file, encoding='utf-8')  # Change 'utf-8' to the appropriate encoding

    # Extract 'review' and 'label' columns as lists
    test_reviews = df['review'].tolist()
    test_labels = df['label'].replace({'pos': "positive", 'neg': "negative"}).tolist()

    return test_reviews, test_labels

# Replace 'test.csv' with your actual file name
input_file = 'test.csv'

print("Splitting into lists...")
og_test_reviews, og_test_labels = split_columns_to_lists(input_file)

print("Getting predictions...")

for num_tests in [250, 500, 1000]:
    for MODEL in "JiaqiLee/imdb-finetuned-bert-base-uncased", "distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment-latest":
        
        if (num_tests == 250) and ("distilbert" in MODEL):
            continue
        if (num_tests != 250) and ("cardiff" not in MODEL):
            continue

        # MODEL = "JiaqiLee/imdb-finetuned-bert-base-uncased"
        # MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
        # MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        print(MODEL, num_tests)
        start = datetime.now()

        # sentiment_task_roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        sentiment_task_roberta = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True, max_length=510)
        # print(output)

        # config = AutoConfig.from_pretrained(MODEL)
        # max_sequence_length = config.max_position_embeddings
        # print("Max sequence length:", max_sequence_length)

        test_reviews = og_test_reviews[:num_tests//2] + og_test_reviews[-num_tests//2:]
        test_labels = og_test_labels[:num_tests//2] + og_test_labels[-num_tests//2:]

        # print(test_labels)

        predictions = []
        for out in tqdm(sentiment_task_roberta(test_reviews, batch_size=4, truncation=True, max_length=510)):
            # print(out)
            predictions.append(out)

        print("Formatting predictions...")
        label_preds = [pred["label"] for pred in predictions]
        print(set(label_preds))
        label_preds = [0 if pred.upper() == "NEGATIVE" else 1 for pred in label_preds]
        test_labels = [0 if pred.upper() == "NEGATIVE" else 1 for pred in test_labels]

        print("Calculating accuracy score...")

        print("Score:")
        score = accuracy_score(test_labels, label_preds)
        print(score)
        matrix = confusion_matrix(test_labels, label_preds)
        print(matrix)

        end = datetime.now()
        print(end - start)
        print()
        print("---")
        print()


