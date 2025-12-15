{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMlT8/g6kdeyCRjouz6HnwK",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/deepasrivaradharajan/RBI-DSIM-Analytical-Portfolio-./blob/main/Complaint_Topic_Modeler_NLP.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nk9Lf11mFRrD",
        "outputId": "193f1b74-91a5-4e19-fa03-2a06ecb61a19"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1. Synthesizing 1000 Customer Complaint Documents...\n",
            "2. Preprocessing Text (Stopword Removal, TF-IDF Vectorization)...\n",
            "3. Training Latent Dirichlet Allocation (LDA) Model to Find Topics...\n",
            "\n",
            "4. Final Topics and Top Keywords (The Recruiter's View):\n",
            "----------------------------------------------------------\n",
            "Topic 1: received / otp / issue / also / problem\n",
            "Topic 2: atm / broken / gateway / error / cash\n",
            "Topic 3: high / wrong / emi / amount / interest\n",
            "Topic 4: net / banking / frozen / excessive / charges\n",
            "Topic 5: application / loan / status / long / time\n",
            "----------------------------------------------------------\n",
            "\n",
            "Goal: This model demonstrates the ability to auto-categorize large volumes of unstructured data, a key need for bank operations.\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.decomposition import LatentDirichletAllocation\n",
        "import re\n",
        "from nltk.corpus import stopwords\n",
        "import nltk\n",
        "\n",
        "# Download stopwords list if not already present\n",
        "try:\n",
        "    stopwords.words('english')\n",
        "except LookupError:\n",
        "    # This line ensures NLTK data is available in environments like Colab/VS Code\n",
        "    nltk.download('stopwords')\n",
        "\n",
        "# --- 1. Data Synthesis (Simulating Raw Customer Complaint Text) ---\n",
        "print(\"1. Synthesizing 1000 Customer Complaint Documents...\")\n",
        "SEED = 123\n",
        "np.random.seed(SEED) # Corrected: np is now defined\n",
        "\n",
        "# Define common banking complaint keywords\n",
        "topics = {\n",
        "    'ATM_Issue': ['ATM is broken', 'machine ate my card', 'cash not dispensed', 'pin issue', 'out of service'],\n",
        "    'Fee_Dispute': ['unauthorized charge', 'service fee is too high', 'excessive charges', 'late payment fine', 'transaction charge'],\n",
        "    'Loan_Query': ['loan application status', 'high interest rate', 'document submission', 'EMI amount is wrong', 'foreclosure request'],\n",
        "    'Digital_App': ['mobile app crashes', 'login failure', 'net banking frozen', 'OTP not received', 'payment gateway error'],\n",
        "    'Customer_Service': ['no one answers the phone', 'poor branch service', 'long waiting time', 'rude employee', 'manager unavailable'],\n",
        "}\n",
        "\n",
        "documents = []\n",
        "keywords_list = list(topics.values())\n",
        "for i in range(1000):\n",
        "    # Select 2 random keywords and mix them into a sentence\n",
        "    # Corrected: np is now defined\n",
        "    keyword1 = np.random.choice(keywords_list[np.random.randint(0, len(keywords_list))])\n",
        "    keyword2 = np.random.choice(keywords_list[np.random.randint(0, len(keywords_list))])\n",
        "\n",
        "    text = f\"Dear bank, I had a problem with my account this morning. The {keyword1} and also the issue with the {keyword2} was frustrating. Please look into this immediately.\"\n",
        "    documents.append(text)\n",
        "\n",
        "# --- 2. Text Preprocessing and Vectorization ---\n",
        "print(\"2. Preprocessing Text (Stopword Removal, TF-IDF Vectorization)...\")\n",
        "\n",
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "def preprocess_text(text):\n",
        "    text = text.lower()\n",
        "    text = re.sub(r'[^a-z\\s]', '', text) # Remove punctuation/numbers\n",
        "    tokens = text.split()\n",
        "    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]\n",
        "    return \" \".join(tokens)\n",
        "\n",
        "preprocessed_documents = [preprocess_text(doc) for doc in documents]\n",
        "\n",
        "# Use TF-IDF to convert text to numerical features\n",
        "tfidf_vectorizer = TfidfVectorizer(max_features=500) # Limit features for efficiency\n",
        "tfidf = tfidf_vectorizer.fit_transform(preprocessed_documents)\n",
        "\n",
        "# --- 3. Unsupervised Topic Modeling (LDA) ---\n",
        "print(\"3. Training Latent Dirichlet Allocation (LDA) Model to Find Topics...\")\n",
        "# We will look for 5 distinct topics based on our source data\n",
        "n_components = 5\n",
        "lda = LatentDirichletAllocation(\n",
        "    n_components=n_components,\n",
        "    max_iter=5,\n",
        "    learning_method='online',\n",
        "    random_state=SEED\n",
        ")\n",
        "lda.fit(tfidf)\n",
        "\n",
        "# --- 4. Topic Interpretation (Core Deliverable) ---\n",
        "print(\"\\n4. Final Topics and Top Keywords (The Recruiter's View):\")\n",
        "print(\"----------------------------------------------------------\")\n",
        "feature_names = tfidf_vectorizer.get_feature_names_out()\n",
        "\n",
        "def print_top_words(model, feature_names, n_top_words):\n",
        "    for topic_idx, topic in enumerate(model.components_):\n",
        "        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]\n",
        "        top_words = [feature_names[i] for i in top_words_idx]\n",
        "        print(f\"Topic {topic_idx + 1}: {' / '.join(top_words)}\")\n",
        "\n",
        "print_top_words(lda, feature_names, 5)\n",
        "print(\"----------------------------------------------------------\")\n",
        "print(\"\\nGoal: This model demonstrates the ability to auto-categorize large volumes of unstructured data, a key need for bank operations.\")"
      ]
    }
  ]
}