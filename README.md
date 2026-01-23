# LSTM Text Generation

Implémentation de modèles de langage basés sur les réseaux LSTM (Long Short-Term Memory) avec PyTorch pour la génération de texte en français et en anglais.

## Description

Ce projet explore la génération de texte automatique à travers l'apprentissage des structures linguistiques par des réseaux LSTM. Il propose deux implémentations distinctes :

- **Modèle Français** : Entraîné sur 100 000 phrases du corpus Tatoeba
- **Modèle Anglais** : Entraîné sur le dataset WikiText avec optimisation de la perplexité

### Fonctionnalités

- Architecture LSTM personnalisée avec régularisation avancée
- Multiples stratégies de génération (greedy, sampling, top-k)
- Contrôle de la température et pénalité de répétition
- Évaluation complète avec métriques de diversité et perplexité
- Interface interactive (chatbot) pour tester les générations

## Architecture

### Modèle Français (`LSTMLanguageModel`)
- **Embedding** : 256 dimensions
- **Hidden layers** : 512 unités, 2 couches LSTM
- **Vocabulaire** : ~16 400 mots (fréquence min = 2)
- **Dropout** : 0.5 pour régularisation
- **Techniques avancées** :
  - Logic patching pour corrections grammaticales
  - Pénalité de répétition adaptative
  - Génération best-of-N avec scoring heuristique

### Modèle Anglais (`LSTMModel`)
- **Embedding** : 400 dimensions
- **Hidden layers** : 800 unités, 3 couches LSTM
- **Vocabulaire** : ~20 400 mots (fréquence min = 5)
- **Dropout** : 0.5
- **Perplexité atteinte** : 93.94

## Résultats

### Modèle Français
```
Exemples de génération :
- "je suis" → "je suis désolé de ne pas te voir"
- "je vais" → "je vais vous voir demain"
- "il est" → "il est très gentil"
```

### Modèle Anglais - Métriques d'évaluation

| Dataset   | Diversité Ref | Diversité Pred | Longueur Moy |
|-----------|---------------|----------------|--------------|
| WikiNews  | 92.1%         | 90.5%          | 53.2 tokens  |
| WikiText  | 91.8%         | 90.6%          | 52.8 tokens  |
| Book      | 93.2%         | 90.5%          | 53.1 tokens  |

**Points forts** :
- Excellente diversité (~90.5%) proche des textes de référence
- Génération stable avec longueur cohérente
- Robustesse entre différents domaines

## Installation

### Prérequis
```bash
Python 3.8+
PyTorch 2.0+
```

### Dépendances
```bash
pip install torch numpy matplotlib pandas
```

## Utilisation

### 1. Modèle Français

Ouvrez `LSTMLanguageModel.ipynb` et exécutez les cellules dans l'ordre :

```python
# Charger le modèle pré-entraîné
checkpoint = torch.load("mon_modele_lstm_francais.pth",
                        map_location=torch.device('cpu'),
                        weights_only=False)

# Générer du texte
model.generate_text("je suis", max_length=15, temperature=0.7)
```

**Options de génération** :
- `max_length` : Nombre maximum de mots à générer
- `temperature` : Contrôle de la créativité (0.5-1.0)
- `top_k` : Restriction aux k mots les plus probables
- `repetition_penalty` : Pénalité pour éviter les répétitions (1.0-2.0)

### 2. Modèle Anglais

Ouvrez `TextGen_LSTM.ipynb` :

```python
# Charger le modèle
checkpoint = torch.load('lstm_wikitext_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Générer avec sampling
generate_words(model, vocab, "The government announced",
               max_length=100, mode='sampling', temp=0.7, top_k=50)
```

### 3. Évaluation des résultats

Consultez `LSTM_Evaluation_Results.ipynb` pour voir :
- Graphiques de comparaison de diversité
- Analyses statistiques de longueur de génération
- Métriques détaillées par dataset

## Structure du Projet

```
lstm-text-generation/
├── LSTMLanguageModel.ipynb         # Modèle français avec interface interactive
├── TextGen_LSTM.ipynb              # Modèle anglais + évaluation
├── LSTM_Evaluation_Results.ipynb   # Visualisation des résultats
├── mon_modele_lstm_francais.pth    # Checkpoint modèle français
├── lstm_wikitext_model.pth         # Checkpoint modèle anglais
└── README.md
```

## Méthodologie

### Pipeline d'entraînement
1. **Nettoyage du corpus** : Filtrage, tokenisation, normalisation
2. **Construction du vocabulaire** : Tokens spéciaux (<BOS>, <EOS>, <PAD>, <UNK>)
3. **Préparation des données** : Création de paires (input, target) avec padding
4. **Entraînement** :
   - Optimiseur AdamW avec weight decay
   - Learning rate scheduler (ReduceLROnPlateau)
   - Gradient clipping (max_norm=1.0)
   - Early stopping (patience=7)
5. **Génération** : Sampling avec température et top-k filtering

### Techniques de régularisation
- Dropout (0.5) entre les couches LSTM
- Weight decay (0.001)
- Gradient clipping
- Vocabulaire filtré par fréquence

## Exemples d'utilisation avancée

### Génération best-of-N (modèle français)
```python
# Génère 10 candidats et sélectionne le meilleur selon des critères heuristiques
model.generate_best("je suis", n_candidates=10, max_length=15,
                    temperature=0.7, top_k=15)
```

### Interface de chat interactif
```python
# Chatbot interactif avec vérification du vocabulaire
chat_with_model_secure(loaded_model)
```

## Améliorations Futures

- [ ] Augmenter la taille du corpus d'entraînement
- [ ] Implémenter l'attention mechanism
- [ ] Tester des architectures Transformer
- [ ] Ajouter le beam search pour la génération
- [ ] Fine-tuning sur des domaines spécifiques
- [ ] Évaluation BLEU et ROUGE pour la qualité

## Datasets Utilisés

- **Tatoeba** : Phrases françaises (100 000 phrases)
- **WikiText** : Articles Wikipédia anglais (20 741 phrases)
- **Évaluation** : WikiNews, WikiText, BookCorpus

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## Licence

Ce projet est sous licence MIT.

## Auteur

Guillaume - [Guillaumevr79](https://github.com/Guillaumevr79)

---

Si ce projet vous est utile, n'hésitez pas à lui donner une étoile !
