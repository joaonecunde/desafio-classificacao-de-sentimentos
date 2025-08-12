# Desafio - Classificação de Sentimentos em Mensagens de Texto

## Descrição

Este projeto consiste no desenvolvimento de uma aplicação simples de Inteligência Artificial para classificar mensagens de texto como **Positivas** ou **Negativas**, utilizando técnicas básicas de Processamento de Linguagem Natural (NLP) e aprendizado de máquina.

---

## Funcionalidades

* Leitura de arquivo `.txt` contendo mensagens e seus respectivos rótulos (Positiva/Negativa).
* Pré-processamento do texto: conversão para minúsculas, remoção de stopwords, lematização usando spaCy.
* Vetorização das mensagens com TF-IDF para transformação em dados numéricos.
* Treinamento de modelo de classificação Naive Bayes para identificar sentimento das mensagens.
* Avaliação da acurácia do modelo em conjunto de teste.
* Classificação automática de todas as mensagens e geração de arquivo CSV com resultados.

---

## Requisitos

* Python 3.7 ou superior
* Bibliotecas Python:

  * pandas
  * scikit-learn
  * spacy
* Modelo spaCy para português (`pt_core_news_sm`) — será baixado automaticamente se não estiver presente.

---

## Pré-requisitos e instalação do modelo de linguagem

Para rodar o projeto corretamente, é necessário instalar as bibliotecas e o modelo de linguagem do spaCy para português.

### Passos recomendados:

1. Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

2. Instale o spaCy e o modelo de português:

```bash
pip install spacy
python -m spacy download pt_core_news_sm
```

3. (Opcional) Instale o NLTK caso queira usar como alternativa:

```bash
pip install nltk
```

---

**Nota:**
O script assume que o modelo `pt_core_news_sm` do spaCy está instalado. Caso não esteja, será necessário instalá-lo conforme instruções acima para evitar erros de execução.

---

## Instalação

1. Clone o repositório ou faça download do código.
2. Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .env
source .env/bin/activate    # Linux/macOS
.env\Scripts\activate       # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. (Opcional) Caso o modelo do spaCy não esteja instalado, o script tentará baixá-lo automaticamente. Se preferir, pode instalar manualmente:

```bash
python -m spacy download pt_core_news_sm
```

---

## Uso

1. Prepare um arquivo `.txt` contendo mensagens com formato:

```
mensagem<TAB>Positiva|Negativa
```

Exemplo:

```
Estou muito feliz com o atendimento	Positiva
Produto chegou com defeito	Negativa
```

2. Execute o script principal:

```bash
python main.py
```

3. O script fará:

* Leitura e pré-processamento do arquivo
* Treinamento e avaliação do modelo
* Classificação das mensagens
* Geração do arquivo `resultado_classificacao.csv` com as mensagens originais, seus rótulos e as predições do modelo.

---

## Estrutura do Projeto

```
.
├── db.txt               # Arquivo de dados com mensagens e rótulos
├── main.py              # Script principal para rodar o desafio
├── resultado_classificacao.csv  # Saída do modelo (gerada após rodar o script)
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

---

## Considerações Técnicas

* O pré-processamento usa spaCy para português, mas utiliza NLTK como fallback caso spaCy não esteja disponível, garantindo maior robustez.
* O modelo de classificação é um Multinomial Naive Bayes, simples e eficiente para texto.
* A vetorização é feita via TF-IDF para dar peso às palavras mais relevantes.
* O código está organizado para ser claro, modular e fácil de entender, seguindo boas práticas.

---

## Contato

[João Necunde](https://www.linkedin.com/in/joao-necunde/)

---
