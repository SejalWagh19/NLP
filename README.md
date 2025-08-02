# Natural Language Processing
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#nlp">NLP</a></li>
    <li><a href="#core-concepts-in-nlp">Core Concepts in NLP</a></li>
    <li><a href="#text-preprocessing">Text Preprocessing</a></li>
    <ol>
      <li><a href="#tokenization">Tokenization with Example</a></li>
      <li><a href="#stemming">Stemming</a></li>
      <li><a href="#lemmatization">Lemmatization</a></li>
      <li><a href="#stopwords">Stopwords</a></li>
      <li><a href="#parts-of-speech-tag">Parts of Speech Tag</a></li>
    </ol>
   <li><a href=""text-to-vector>Text to Vector</a></li>
  </ol>
</details>
This repository represents the implementation of NLP concepts using NLTK library.

---

## NLP
Natural Language Processing (NLP) is all about enabling machines to understand and process human language. One of the very first and most fundamental steps is text processing. Whether you're building a chatbot, a sentiment analysis model, or a search engine, you can't skip this step.

Let's first understand a few foundational terms often used in text processing:
* Corpus - Documents/Paragraphs
* Documents - Sentences
* Words - All the words that are present in the corpus
* Vocabulary - Unique Words

## Core Concepts in NLP
<ol>
  <li><b>Corpus:</b> A corpus is a large collection of textual data. It can be considered as a collection of paragraphs or documents.</li>
  <li><b>Document:</b> A document is typically a single paragraph or a unit of text within a corpus.</li>
  <li><b>Words:</b> These are all individual lexical units (tokens) that appear in the corpus, regardless of repetition.</li>
  <li><b>Vocabulary:</b> The vocabulary refers to the set of unique words present in the corpus. It forms the foundation for constructing vector representations of textual data.</li>
</ol>

---

# Text Preprocessing
## 1) Tokenization
Tokenization is a text preprocessing technique that involves splitting raw text into smaller units called tokens. These tokens could be:
* Sentence-level tokens (splitting a paragraph into individual sentences)
* Word-level tokens (splitting a sentence into individual words)<br>

`"Better tokens lead to better models."`

Each token can then be processed and transformed into numerical vectors, an essential step since the models can't understand raw text directly.

## Example 
**Example 1: Paragraph Tokenization**

*Input Paragraph:*
```
"On a rainy afternoon, Maria decided to bake cookies. She gathered flour, sugar, and chocolate chips from the pantry."
```
**Step 1:** Sentence Tokenization <br>
Using sentence boundary markers such as full stops (.) and exclamation marks (!), the paragraph is tokenized into sentences:

```
[
      "On a rainy afternoon, Maria decided to bake cookies",
      "She gathered flour, sugar, and chocolate chips from the pantry"
]
```

**Step 2:** Word Tokenization <br>
Each sentence is further tokenized into words:

```
Sentence 1: ["On", "a", "rainy", "afternoon", "Maria", "decided", "to", "bake", "cookies"]
Sentence 2: ["She", "gathered", "flour", "sugar", "and", "chocolate", "chips", "from", "the", "pantry"]
```

## 

**Example 2: Vocabulary Extraction**

*Input Text:*
```
"I like to drink Apple Juice. My friend likes Mango Juice."
```
**Tokenized Sentences:**
```
[
  "I like to drink Apple Juice",
  "My friend likes Mango Juice"
]
```
**Word Tokens:**
```
["I", "like", "to", "drink", "Apple", "Juice", "My", "friend", "likes", "Mango", "Juice"]
```
`Total Words: 11`

**Unique Words (Vocabulary):**
```
["I", "like", "to", "drink", "Apple", "Juice", "My", "friend", "likes", "Mango"]
```
`Vocabulary Size: 10`

**Note:** "Juice" appears twice, hence the count of total words are 11 but unique words are 10.
### ⤵️ Tokenization Functions from NLTK
Download the pre-trained tokenizer models needed for sentence and word tokenization.

```
import nltk
nltk.download('punkt')
```

| Function                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `sent_tokenize(corpus)`           | Breaks a paragraph into individual sentences.                               |
| `word_tokenize(corpus)`           | Breaks a sentence into words and punctuation.                               |
| `wordpunct_tokenize(corpus)`      | Splits text into words and also separates all punctuation marks.            |
| `TreebankWordTokenizer().tokenize(corpus)` | Breaks text into words using rules from the Penn Treebank (eg: "don't" → "do" + "n't"). |

## 2) Stemming
Stemming is the process of **reducing a word to its word stem** that affixes to suffixes and prefixes or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP).

**Types of Stemmer**
i) **Porter Stemmer**
- The Porter Stemmer is one of the oldest and most widely used stemming algorithms in NLP. 
- It works by applying a series of rule-based steps to remove common word endings such as -ing, -ed, and -ly. 
- For example, it converts the word `"running" to "run"` and `"caresses" to "caress"`
- It's simple and efficient, making it useful for basic text analysis in English. 
- However, a major disadvantage is that it can sometimes be too aggressive, removing parts of words inappropriately. 
- For instance, "university" might be stemmed to "univers", which is not a real word.

ii) **Snowball Stemmer**
- The Snowball Stemmer is essentially an improved version of the Porter Stemmer, designed to be more consistent and support multiple languages, including English, French, German, and more.
- It uses a more refined set of rules and provides better accuracy for modern NLP tasks.
- Like Porter, it turns `"running" into "run"`, but tends to avoid some of the over-stemming problems.
- The main disadvantage is that while it's better than Porter, it still doesn't always produce real root words, just shorter versions, and is still based on fixed rules rather than word meaning.

iii) **Regexp Stemmer**
- The Regexp Stemmer is a customizable stemmer where you define your own rules using regular expressions (regex).
- This gives you full control over how stemming is done, which is helpful in specialized tasks or domain-specific text.
- For example, you can write a rule to remove the ending "ing" from any word, so "jumping" becomes "jump".
- The advantage is flexibility, but the disadvantage is that it requires manual setup and good knowledge of regex, and can easily miss edge cases or introduce errors if not carefully designed.

### ⤵️ Stemming Functions from NLTK
`
import nltk
`

**Porter Stemmer**
```
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
stemming.stem(word)
```

**Snowball Stemmer**
```
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
snowball_stemmer.stem(word)
```

**Regexp Stemmer**
```
from nltk.stem import RegexpStemmer
reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
reg_stemmer.stem(word)
```

## 3) Lemmatization
Lemmatization technique is like stemming. The output we will get after lemmatization is called **'lemma'**, which is a **root word** rather than root stem, the output of stemming. After lemmatization, we will be getting a valid word that means the same thing.

**Wordnet Lemmatizer:**
NLTK provides WordNetLemmatizer class which is a thin wrapper around the wordnet corpus. This class uses morphy() function to the WordNet CorpusReader class to find a lemma.

| POS       | Tag     |
|---------- |---------|
| Noun      | `n`     |
| Verb      | `v`     |
| Adjective | `a`     |
| Adverb    | `r`     |

### ⤵️ Lemmatization Functions from NLTK
`
import nltk
`

**WordNetLemmatizer**

```
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize(word,pos='v')
```

## 4) Stopwords
- Stopwords refer to high-frequency lexical items i.e., common words (e.g., "the", "is", "in", "and") that often do not contribute significant semantic value to text-based tasks. 
- Removing stopwords is a common preprocessing step to reduce dimensionality, eliminate noise, and improve computational efficiency, especially in tasks like information retrieval, text classification, and topic modeling. 

### ⤵️ Stopwords Functions from NLTK
```
import nltk
nltk.download('stopwords')
```
The nltk library provides a built-in stopword corpus through nltk.corpus.stopwords, which can be accessed and filtered using functions like stopwords.words('english'). 
```
from nltk.corpus import stopwords
stopwords.words('english')
```

## 5) Parts of Speech Tag
### POS Tag Reference

| **Tag**| **Description**                                  | **Example(s)**                |
|--------|--------------------------------------------------|-------------------------------|
| CC     | Coordinating conjunction                         | and, but, or                  |
| CD     | Cardinal digit                                   | one, two, 100                 |
| DT     | Determiner                                       | the, a, an                    |
| EX     | Existential there                                | there is, there exists        |
| FW     | Foreign word                                     | voila, bonjour                |
| IN     | Preposition/Subordinating conjunction            | in, on, because               |
| JJ     | Adjective                                        | big                           |
| JJR    | Adjective, comparative                           | bigger                        |
| JJS    | Adjective, superlative                           | biggest                       |
| LS     | List marker                                      | 1), a), i)                    |
| MD     | Modal                                            | could, will, should           |
| NN     | Noun, singular                                   | desk                          |
| NNS    | Noun, plural                                     | desks                         |
| NNP    | Proper noun, singular                            | Harrison                      |
| NNPS   | Proper noun, plural                              | Americans                     |
| PDT    | Predeterminer                                    | "all the kids"                |
| POS    | Possessive ending                                | parent's                      |
| PRP    | Personal pronoun                                 | I, he, she                    |
| PRP$   | Possessive pronoun                               | my, his, hers                 |
| RB     | Adverb                                           | very, silently                |
| RBR    | Adverb, comparative                              | better                        |
| RBS    | Adverb, superlative                              | best                          |
| RP     | Particle                                         | give up                       |
| TO     | "To"                                             | to go 'to' the store          |
| UH     | Interjection                                     | ugh, hmm, errrrrrrrm          |
| VB     | Verb, base form                                  | take                          |
| VBD    | Verb, past tense                                 | took                          |
| VBG    | Verb, gerund/present participle                  | taking                        |
| VBN    | Verb, past participle                            | taken                         |
| VBP    | Verb, singular present, non-3d                   | take                          |
| VBZ    | Verb, 3rd person singular present                | takes                         |
| WDT    | Wh-determiner                                    | which                         |
| WP     | Wh-pronoun                                       | who, what                     |
| WP$    | Possessive wh-pronoun                            | whose                         |
| WRB    | Wh-adverb                                        | where, when                   |

## ⤵️ POS Tag NLTK
```
import nltk
nltk.download('averaged_perceptron_tagger')
```

If you want to directly pass a sentence and check the POS for each word, then do:
```
nltk.pos_tag(sentence.split())
```
---

# Text to Vector
## 1) One-Hot Encoding

Consider the following example: <br>
|     | Text                 | Output|
|-----|----------------------|-------|
|D1   | The food is good     |1      |
|D2   | The food is bad      |0      |
|D3   | Pizza is Amazing     |1      |

### Vocabulary (Unique Words)  
**Vocabulary size = 7**

`["The", "food", "is", "good", "bad", "Pizza", "Amazing"]`

| Word      | Vector                |
|-----------|-----------------------|
| The       | [1, 0, 0, 0, 0, 0, 0] |
| food      | [0, 1, 0, 0, 0, 0, 0] |
| is        | [0, 0, 1, 0, 0, 0, 0] |
| good      | [0, 0, 0, 1, 0, 0, 0] |
| bad       | [0, 0, 0, 0, 1, 0, 0] |
| Pizza     | [0, 0, 0, 0, 0, 1, 0] |
| Amazing   | [0, 0, 0, 0, 0, 0, 1] |

### Encoded Representation

**D1: The food is good**
```
[
  [1, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0]
]  # 4x7
```

**D2: The food is bad**
```
[
  [1, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0]
]  # 4x7
```

**D3: Pizza is Amazing**
```
[
  [0, 0, 0, 0, 0, 1, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1]
]  # 3x7
```

### Test Text: "Burger is bad"
```
[
  [0, 0, 0, 0, 0, 0, 0], # "Burger" is out of vocabulary
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0]
]
```

### Advantages
Easy to implement in Python, Using libraries like:  
`python:
sklearn.OneHotEncoder
pandas.get_dummies
`

### Disadvantages
<ol>
    <li>Sparse matrix → Overfitting</li>
    <li>ML algorithms → Require fixed input size</li>
    <li>No semantic meaning is captured</li>
    <li>Out of Vocabulary (OOV) issues</li>
</ol>


## 2) Bag of Words
The Bag of Words (BoW) model is a text representation technique used in NLP, where a text corpus is represented as a "bag" of words, disregarding grammar and word order but keeping track of the word frequencies. The BoW model transforms text data into a structured format, such as a matrix, where rows represent individual text samples (e.g., documents or sentences) and columns represent words in the vocabulary. <br><br>
**Steps in Preprocessing for BoW:**
<ol>
  <li><b>Text Lowercasing:</b> The text is converted to lowercase to maintain consistency and prevent differentiating between words like "Apple" and "apple."</li>
  <li><b>Stopwords Removal:</b> Common words (stopwords) like "and," "the," "is," etc., which do not carry significant meaning in most NLP tasks, are removed to reduce noise in the dataset.</li>
  <li><b>Tokenization:</b> The remaining text is broken down into tokens (usually words).</li>
  <li><b>Vocabulary Construction:</b> The distinct words (tokens) across the entire corpus are used to create a vocabulary, which forms the basis for the BoW model.</li>
  <li><b>Frequency Count or Binary Encoding:</b> After the vocabulary is built, two primary approaches can be used to encode the presence and frequency of words</li>
</ol>

**Types of Bag of Words (BoW):**
<ol>
  <li>
    <b>Binary Bag of Words (Binary BoW):</b> <br>
    - <b>Representation:</b> In Binary BoW, each word is represented as either present or absent in a given text sample. <br>
    - <b>Encoding:</b> The feature values are binary (0 or 1). If a word is present in a document, it is represented as 1; otherwise, it is 0. <br>
    - <b>Example:</b> <br>
     Vocabulary: ["dog", "cat", "fish"] <br>
     Document 1: "dog and cat" → [1, 1, 0] <br>
     Document 2: "fish and cat" → [0, 1, 1] <br>
  </li>
  <li>
    <b>Frequency-Based Bag of Words (Frequency BoW):</b> <br>
    - <b>Representation:</b> Each word is represented by its frequency of occurrence in the document. <br>
    - <b>Encoding:</b> The feature values represent the count of times each word appears in a given document. <br>
    - <b>Example:</b> <br>
     Vocabulary: ["dog", "cat", "fish"] <br>
     Document 1: "dog and cat and dog" → [2 1, 0] <br>
     Document 2: "fish and cat" → [0, 1, 1] <br>
  </li>
</ol>

Example: <br>
| Text                      | Output|
|---------------------------|-------|
| He is a good boy          |1      |
| She is a good girl        |0      |
| Boy and girl are good     |1      |

Lowercase all the words in the text and remove the stopwords.
`
S1 → good boy;
S2 → good girl;
S3 → Boy girl good
`

| Vocabulary  | Frequency|
|-------------|----------|
| good        |3         |
| boy         |2         |
| girl        |2         |

```
      [good  boy  girl]    output
S1    [ 1    1     0  ]      1
S2    [ 1    0     1  ]      1
S3    [ 1    1     1  ]      1
```

### Advantages  
1) Simple and Intuitive
2) Fixed Sized Input is given to → ML Algorithms

### Disadvantages
<ol>
  <li>Sparse matrix/array → Overfitting</li>
  <li>Ordering of the word is getting changed</li>
  <li>Out of Vocabulary (OOV)</li>
  <li>Semantic meaning is still not captured</li>
</ol>












