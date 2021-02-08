# GFG Data Scientist Task
This repository contains the project submission for Data Scientist role in the Data Science and Analytics Department at Global Fashion Group (GFG).

## Contents
  1. Introduction
  2. Folder content
  3. Exploratory Data Analysis (EDA)
  4. Proposed Methodology
  4. Models exploration
  5. Evaluation
  6. Further work
  7. Last words
  8. References


## 1. Introduction
The one-size-fits-all marketing approach is getting old. Customers are asking for more tailored products and services. The lead companies of tomorrow ought to understand their clients better. <br/><br/>
Identifying and organizing customer groups using specific variables and characteristics they have in common is commonly referred as **Customer Segmentation**. These variables could be personality traits, demographics, geography, or even their income. Segmentation provides in-depth consumer data that helps marketers tailor their products and services to customers’ needs. This personalization gives companies a competitive advantage and a better chance at customer conversion and brand loyalty. <br/><br/>
Demographic, Behavioral, and Geographical and 3 different types of customer segmentation. Today, we will be interested in the first one, and more specifically customers' ***gender***.

## 2. Folder content
In an effort to make this repository as easy to read as possible, I created 3 different ipynb files with a specific purpose each. <br/>
1. Exploratory data analysis (EDA)
    - Going through the data, taking notes, and visualizing patterns
2. Machine Learning models exploration
    - Testing different models, visualizing the output
3. Pipeline
    - Providing only the code relevant to the output

I suggest to read them in order.

## 3. Exploratory Data Analysis (EDA)
This section is dedicated to EDA. We will explore the data, ask ourselves questions, visualize features, and more. <br/>
Looking at the type of our 33 features, one can note the presence of 1 categorical variable (is_newsletter_subscriber), 3 float variables, and the rest as integers.
Models do not like categorical feature, they prefer binary values. We will change that later on. The variable 'coupon_discount_applied' is stored as float, which is surprising to me as I usually only see 15%, 20% or any "nice" integers used for coupon discount. <br/>
Our dataset has 191,287 records, which correspond to the number of customer_id. There is no duplicate nor missing value in our data. <br/>
Taking a quick look at each individual feature, there are few comments we can make.
  - In average, customers' first order was more than 6 years ago.
  - The last order was made more than 2 years ago in average. Less than 5% of customers have ordered an item less than a week ago (from the dataset today's date reference).
  - The data has numerous significant outliers. Some of them are suspicious.
      - Is it really possible to have used 1,122 different shipping addresses? Even for 6 years, that represents more than 15 different shipping addresses per month...
      - 2,022 returns out of 2,209 items purchased for 1 particular customer, really? That's a truly unhappy client...
  - 'revenue' feature has 1 negative value and several 0 values. I think it is reasonable to think there are some errors here.

