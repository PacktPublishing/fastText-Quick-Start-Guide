# fastText Quick Start Guide

<a href="https://www.packtpub.com/big-data-and-business-intelligence/fasttext-quick-start-guide?utm_source=github&utm_medium=repository&utm_campaign=9781789130997"><img src="https://d255esdrn735hr.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/B10726.png" alt="fastText Quick Start Guide" height="256px" align="right"></a>

This is the code repository for [fastText Quick Start Guide](https://www.packtpub.com/big-data-and-business-intelligence/fasttext-quick-start-guide?utm_source=github&utm_medium=repository&utm_campaign=9781789130997), published by Packt.

**Get started with Facebook's library for text representation and classification**

## What is this book about?

Facebook's fastText library handles text representation and classification, used for Natural Language Processing (NLP). Most organizations have to deal with enormous amounts of text data on a daily basis, and gaining efficient data insights requires powerful NLP tools such as fastText. 
This book is your ideal introduction to fastText. You will learn how to create fastText models from the command line, without the need for complicated code. You will explore the algorithms that fastText is built on and how to use them for word representation and text classification. 
Next, you will use fastText in conjunction with other popular libraries and frameworks such as Keras, TensorFlow, and PyTorch. 
Finally, you will deploy fastText models to mobile devices. By the end of this book, you will have all the required knowledge to use fastText in your own applications at work or in projects.

This book covers the following exciting features:
* Create models using the default command line options in fastText
* Understand the algorithms used in fastText to create word vectors
* Combine command line text transformation capabilities and the fastText library to implement a training, validation, and prediction pipeline
* Explore word representation and sentence classification using fastText
* Use Gensim and spaCy to load the vectors, transform, lemmatize, and perform other NLP tasks efficiently


If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789130999) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, chapter2.

The code will look like the following:
```
import csv
import sys
w = csv.writer(sys.stdout)
for row in csv.DictReader(sys.stdin):
 w.writerow([row['stars'], row['text'].replace('\n', '')])
```

**Following is what you need for this book:**
This book is for data analysts, data scientists, and machine learning developers who want to perform efficient word representation and sentence classification using Facebook's fastText library. Basic knowledge of Python programming is required

With the following software and hardware list you can run all code files present in the book (Chapter 1-7).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 1 | fastText 0.1.0, Python 3 or above, Docker Community Edition 18.03 | Windows, Mac OS X or Linux  |
| 2-6 | fastText 0.1.0, Python 3 or above, Perl 5.o or above, Jupyter Notebook 5.5.0, Keras 2.2.0, Gensim 3.5.0, NumPy 1.14, SciPy 1.1.0, Pandas 0.23.2, TensorFlow 1.9, PyTorch 0.4.0, torchtext 0.2.3, Scikit-learn 0.19.1, spaCy 2.0.11 | Mac OS X, and Linux (Any) |
| 7 | Python 3 or above, Flask 0.12.4, Android Studio | Windows, Mac OS X or  Linux (Any) |



### Related products
* Natural Language Processing with Python Cookbook  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/natural-language-processing-python-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781787289321) [[Amazon]](https://www.amazon.com/dp/178728932X)

* Natural Language Processing with TensorFlow [[Packt]](https://www.packtpub.com/application-development/natural-language-processing-tensorflow?utm_source=github&utm_medium=repository&utm_campaign=9781788478311) [[Amazon]](https://www.amazon.com/dp/1788478312)



## Get to Know the Author
**Joydeep Bhattacharjee**
 is a Principal Engineer who works for Nineleaps Technology Solutions. After graduating from National Institute of Technology at Silchar, he started working in the software industry, where he stumbled upon Python. Through Python, he stumbled upon machine learning. Now he primarily develops intelligent systems that can parse and process data to solve challenging problems at work. He believes in sharing knowledge and loves mentoring in machine learning. He also maintains a machine learning blog on Medium.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.



### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781789130997">https://packt.link/free-ebook/9781789130997 </a> </p>