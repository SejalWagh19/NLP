# Text Preprocessing in NLP
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#nlp">NLP</a></li>
    <li><a href="#core-concepts-in-nlp">Core Concepts in NLP</a></li>
    <li><a href="#tokenization">Tokenization</a></li>
    <li><a href="#example">Example</a></li>
    <li><a href="#tokenization-functions-from-nltk">Tokenization Functions from NLTK</a></li>
  </ol>
</details>
This repository represents the implementation of NLP concepts using NLTK library.

------

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

## Tokenization
Tokenization is a text preprocessing technique that involves splitting raw text into smaller units called tokens. These tokens could be:
* Sentence-level tokens (splitting a paragraph into individual sentences)
* Word-level tokens (splitting a sentence into individual words)<br>

`"Better tokens lead to better models."`

Each token can then be processed and transformed into numerical vectors, an essential step since the models can't understand raw text directly.

## Example 
**Example 1: Paragraph Tokenization**
**Input Paragraph:**
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
**Input Text:**
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
## Tokenization Functions from NLTK
Download the pre-trained tokenizer models needed for sentence and word tokenization.

```
import nltk
nltk.download('punkt')
```

| Function                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `sent_tokenize(corpus)`           | Breaks a paragraph into individual sentences.                              |
| `word_tokenize(corpus)`           | Breaks a sentence into words and punctuation.                              |
| `wordpunct_tokenize(corpus)`      | Splits text into words and also separates all punctuation marks.           |
| `TreebankWordTokenizer().tokenize(corpus)` | Breaks text into words using rules from the Penn Treebank (eg: "don't" → "do" + "n't"). |






