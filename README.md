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
The one-size-fits-all marketing approach is getting old. Customers are asking for more tailored products and services. The lead companies of tomorrow ought to understand their clients better. 

Identifying and organizing customer groups using specific variables and characteristics they have in common is commonly referred as **Customer Segmentation**. These variables could be personality traits, demographics, geography, or even their income. Segmentation provides in-depth consumer data that helps marketers tailor their products and services to customers’ needs. This personalization gives companies a competitive advantage and a better chance at customer conversion and brand loyalty.

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
This section is dedicated to EDA. We will explore the data, ask ourselves questions, visualize features, and more. 

#### a. High-level notes 
Looking at the type of our 33 features, one can note the presence of 1 categorical variable (is_newsletter_subscriber), 3 float variables, and the rest as integers.
Models do not like categorical feature, they prefer binary values. We will change that later on. <br/>
Our dataset has 191,287 records, which correspond to the number of customer_id. There is no duplicate nor missing value in our data. 

Taking a quick look at each individual feature, there are few comments we can make.
  - In average, customers' first order was more than 6 years ago.
  - The last order was made more than 2 years ago in average. Less than 5% of customers have ordered an item less than a week ago (from the dataset today's date reference).
  - The data has numerous significant outliers. Some of them are suspicious.
      - Is it really possible to have used 1,122 different shipping addresses? Even for 6 years, that represents more than 15 different shipping addresses per month...
      - 2,022 returns out of 2,209 items purchased for 1 particular customer, really? That's a truly unhappy client...
  - 'revenue' feature has 1 negative value and several 0 values. I think it is reasonable to think there are some errors here.
  - 'returns' is described by the formulation problem as "Number of returned **orders**". However, 9.1% of the customers have a number of returns higher than orders. Let's assume that the feature 'returns' is correct, so we will interpret it as "Number of returned **items**".

When we plot the density distribution of each feature, we can see that they are almost all positively skewed (see below). 

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/subplots%20GFG%20fig1.png) 

#### b. Gender related items

Let's now take a look at the distribution of female and male items:
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/distribution%20male_female%20GFG%20fig2.png) 
<br/> One can note that female_items is more distributed towards the right side of the graph (high values). Indeed, customers have in average 5.3 times more female_items then male_items. We need to make sure our training sample is balanced when fitting our models later on. 
<br/> Another interesting point, items is the sum of female_items, male_items, and unisex_items. However, for 42% of the records male_items or female_items is not the sum of app, acc and ftw. We will therefore need to see the correlation between those features before deciding to remove them.

#### c. Coupon discount

As mentioned earlier, the feature 'coupon_discount_applied' is looking quite unnatural. In fact, 0.7% of the customers have an average discount rate of more than 100% (dsitribution plot below) and the upper limit is 57,980%. Funny. <br/>
We will not try to correct those values however, since I believe it is not characteristic of a particular gender, hence not helping with the prediction.<br/>
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/distribution%20coupon%20over%20100%20GFG%20fig3.png) 

#### d. Revenue

The 'revenue' variable is the overall amount of Dollar spent per person. This should not be negative, and 0 values are most likely errors as well (I doubt a customer can get 13 items for free). We will therefore replace the 0 values by the median of each items' value (e.g. the median of 'revenue' for 1 item is 63.61, which will replace the 0 for items being equal to 1).

## 4. Proposed methodology

When tackling a classification problem, it is usually recommended to have a good intuition about the feature importances. One might already sense, without prior exposure to data, that gender was a crucial deciding factor to determine survival rate in the case of the Titanic tragedy. Similarly, if we had to guess the gender of our customers without the help of computing power, how would we do it?<br/>
Let's first take a random sample and try to have a guess.
Feature | Value
------------ | -------------
days_since_first_order | 1122
days_since_last_order | 44
is_newsletter_subscriber | Y
orders | 5
items | 6
returns | 1
different_addresses | 0
shipping_addresses | 2
devices | 2
vouchers | 2
cc_payments | 0
paypal_payments | 5
afterpay_payments | 1
female_items | 2
male_items | 2
unisex_items | 2
wapp_items | 0
wftw_items | 3
mapp_items | 1
wacc_items | 0
macc_items | 1
mftw_items | 1
sprt_items | 0
msite_orders | 2
desktop_orders | 3
android_orders | 0
ios_orders | 0
work_orders | 0 
home_orders | 5
parcelpoint_orders | 0
coupon_discount_applied | 0.0944
revenue | 504.94
customer_id | 3.242546e+09

Hard to say... If you ask me, I would probably label this customer as female because of wftw_items being 3 compared to the other gender related items. But I wouldn't be very confident about my choice. <br/>
Now, what if instead this customer had the following values:
Feature | Value
------------ | -------------
female_items | 15
male_items | 1
unisex_items | 1
wapp_items | 13
wftw_items | 3
mapp_items | 0
wacc_items | 0
macc_items | 1
mftw_items | 0
 
Considering those values, I would certainly make my guess as female customer. Of course I might be wrong, it is possible that this customer is a man and has for habit to purchase items for his daughters and wife. But I believe chances should be on our side.
