import pandas as pd
import spacy
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Pré-processamento de texto
nlp = spacy.load("pt_core_news_sm")  # Modelo

def preprocess_text(text):
    doc = nlp(text.lower())  # minúsculas
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in string.punctuation
        and not token.is_stop
        and token.is_alpha
    ]
    return " ".join(tokens)

#Leitura do arrquivo
def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", names=["mensagem", "label"])
    return df

# Treinamento do modelo

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# Avaliação

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

def main():
    # Carregar dados
    df = load_data("db.txt")
    df["mensagem_processada"] = df["mensagem"].apply(preprocess_text)

    # Separar entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df["mensagem_processada"], df["label"], test_size=0.2, random_state=42
    )

    # Treinar
    model, vectorizer = train_model(X_train, y_train)

    # Avaliar
    evaluate_model(model, vectorizer, X_test, y_test)

    # Classifica mensagens originais
    print("\nClassificando mensagens originais...")
    X_all_vec = vectorizer.transform(df["mensagem_processada"])
    df["predicao"] = model.predict(X_all_vec)

    #Salvar os resultados em um arquivo csv
    df.to_csv("resultado_classificacao.csv", index=False)
    print("Resultados salvos em resultado_classificacao.csv")

if __name__ == "__main__":
    main()