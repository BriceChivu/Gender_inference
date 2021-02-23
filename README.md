# GFG Data Scientist Task
This repository contains the project submission for Data Scientist role in the Data Science and Analytics Department at Global Fashion Group (GFG).

## Contents
  1. Introduction
  2. Folder content
  3. Exploratory Data Analysis (EDA)
  4. Proposed Methodology
  5. Models exploration
  6. Evaluation
  7. Further work
  8. Last words
  9. References


## 1. Introduction
The one-size-fits-all marketing approach is getting old. Customers are asking for more tailored products and services. The lead companies of tomorrow ought to understand their clients better. 

Identifying and organizing customer groups using specific variables and characteristics they have in common is commonly referred as **Customer Segmentation**. These variables could be personality traits, demographics, geography, or even their income. Segmentation provides in-depth consumer data that helps marketers tailor their products and services to customers’ needs. This personalization gives companies a competitive advantage and a better chance at customer conversion and brand loyalty.

Demographic, Behavioral, and Geographical are 3 different types of customer segmentation. Today, we will be interested in the first one, and more specifically customers' ***gender***.

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
  - 'revenue' feature has 1 negative value and several 0 values. It is reasonable to think there are some errors here.
  - 'returns' is described by the formulation problem as "Number of returned **orders**". However, 9.1% of the customers have a number of returns higher than orders. Let's assume that the feature 'returns' is correct, so we will interpret it as "Number of returned **items**".

When we plot the density distribution of each feature, we can see that they are almost all positively skewed (see below). 

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/subplots%20GFG%20fig1.png) 

#### b. Gender related items

Let's now take a look at the distribution of female and male items:
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/distribution%20male_female%20GFG%20fig2.png) 
<br/> One can note that female_items is more distributed towards the right side of the graph (high values). Indeed, customers have in average 5.3 times more female_items then male_items. We need to make sure our training sample is balanced when fitting our models later on. 
<br/> Another interesting point, items is the sum of female_items, male_items, and unisex_items. However, for 42% of the records male_items or female_items is not the sum of app, acc and ftw. The reason being some of those items are classified as unisex_items. We will therefore need to see the correlation between those features before deciding to remove them or not.

#### c. Coupon discount

As mentioned earlier, the feature 'coupon_discount_applied' is looking quite unnatural. In fact, 0.7% of the customers have an average discount rate of more than 100% (dsitribution plot below) and the upper limit is 57,980%. Funny. <br/>
We will not try to correct those values however, since I believe it is not characteristic of a particular gender, hence not helping with the prediction.<br/>
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/distribution%20coupon%20over%20100%20GFG%20fig3.png) 

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

The **proposed methodology** to tackle our problem is based on the above example. We will label some customers having high ratio female_items/male_items or male_items/female_items and use those created labels as a starting point to train our models.
We will therefore adopt a supervised learning approach.

Before moving on to machine learning models, let's see if there is anything else to understand from our existing features and some created ones. We will take a look at our 19,580 (this number can change when we tweak some parameters, see EDA ipynb file) customers labeled (9,790 females 9,790 males) and see if we can spot other interesting trends in our data. <br/>
Let's first plot the mean and median of each feature.

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/mean_median_labeled%20GFG%20fig4.png) 
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/mean_median_labeled_2%20GFG%20fig5.png) 

Few remarks:
   - Female customers have in average a lot more female_items than male_items. This was expected as it was exactly our baseline for label creation.
   - Similarly, male customers have in average a lot more male_items than female_items. Again, that is normal.
   - Males have higher 'mapp_items', 'macc_items', and 'mftw_items' and females have higher 'wapp_items', 'wacc_items', and 'wftw_items'. This is to be expected since there is a significant correlation between those items and female_items/male_items.
   - Females have more orders compared to males.
   - One can note the presence of a new feature called 'avg_revenue_per_item'. Its purpose is to capture the information of 'revenue' and 'items' into 1 variable. We can see that females have in average a higher 'avg_revenue_per_item' meaning that they tend to purchase more expensive items.
   - The feature 'returns_per_item' shows the average number of returns per item. We can see that females tend to return more than males.
   - Females have in average higher 'items', 'vouchers', 'cc_payments', 'paypal_payments', 'msite_orders', 'desktop_orders' and 'home_orders'. This is explained by the fact that those features are highly correlated to the number of orders. By looking at 'online_orders', 'shipped_items', and 'electronic_payment', we can see that there is actually no difference between male and female with regards to those particular variables.
   - The feature 'days_since_last_order' suggests that females' purchase frequency is higher than males'.


### **Conclusion** <br/>
Based on our labeled customers, it seems that:
1. females make more orders as opposed to males,
2. females do shopping more often,
3. females tend to return orders more often.

Based on the remarks above, we will select the following features to train our models:
  - 'days_since_last_order'
  - 'orders'
  - 'avg_revenue_per_item'
  - 'f_items' (sum of female related items)
  - 'm_items' (sum of male related items)
  - 'returns_per_item'

## 5. Models Exploration

Now that we have selected features that we believe make sense, let's try to fit models and predict labels. Below is a table summarizing what models we will use and why.

Classification model | Advantages
------------ | -------------
Logistic Regression | makes no assumptions about distributions of classes in feature space, gives coefficient size and direction of association (positive or negative), very fast to train and predict, good accuracy
K-means | scales to large data sets, guarantees convergence, easy to implement
K-Nearest Neighbors | no training Period, easy to implement
Random Forest | will not overfit, high accuracy
Support Vector Classification | works well when clear margin of separation between classes, memory efficient

For details regarding each models, you can refer to the Machine Learning models exploration ipynb file.

The models performing the best to my opinion are **Logistic Regression** and **Support Vector Classification**. <br/>
Below are graphs for Logistic Regression showing the distribution of labels with regards to different variables.

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/LogisticRegression%206%20plots%20GFG%20fig6.png) 
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/LogisticRegression%204%20plots%20GFG%20fig7.png) 

As we can notice, the model is doing a great job at segregating female and male based on the variables given. The feature importances graph below shows that 'f_items', 'orders', and 'returns_per_items' are significant predictors for female label. On the other hand, 'days_since_last_order' and 'm_items' are strong predictors for male label. This tallies with what was discussed earlier.

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/feature_importances%20Logr%20GFG%20fig8.png) 

Next, let's take a look at the Support Vector Classification model.

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/SVC%206%20plots%20GFG%20fig9.png) 
![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/SVC%204%20plots%20GFG%20fig10.png) 

The boundary between f_items and m_items is even more strict using SVC model. We can confirm this by taking a look at the feature importances below. Only m_items and f_items are driving the predictions.

![Settings Window](https://github.com/BriceChivu/GFG_Data_Scientist_Task/blob/main/Pictures/feature_importances%20SVC%20GFG%20fig11.png) 

The major difference between the Logistic Regression and SVC predictions is the number of females versus males. In fact, the first model predict around **1.5** more females than males as opposed to **3.7** for SVC. The difference is mainly located around customers with low values of orders and with balanced gender items. SVC tends to classify them as females whereas Logistic Regression chooses males.

## 6. Evaluation

While it is pretty straighforward to judge of the accuracy of each model based on our created labels, it is much more complex to know in advance whether those models would perform well with real labels. 

Obviously, the predictions depend highly on the input features, that is why we spent some time defining the correct variables to consider for our models. I believe what we have done so far is coherent in terms of business sense. However, I do believe the predictions will be somewhat off when it comes to customers with low number of orders and with balanced gender items. Nevertheless, we can be confident on our predictions for customers with high female_items/male_items ratio or high male_items/female_items ratio.

## 7. Further work

To improve the accuracy of our predictions, the new features below might come handy:
1. Product type (Is it make-up? Soccer clothes? Video games?)
2. Customer name (Jessica or Bryan?)
3. Email address (jessicabeldore@gmail.com or bryanstanley@gmail.com?)
4. Reviews (NLP gender classification using comments or reviews)
5. Time spent choosing products before buying (e.g. average time spent browsing before adding to cart)

As mentioned before, the initial labels are critical to predictions' accuracy. As the famous saying goes: "garbage in, garbage out". Having real labels attached to our input data could immensely improve our forecast. <br/>
A solution to this could be to collect a group of labels attached to our data, then use a semi-supervised learning approach to propagate labels (think of LabelPropagation in sklearn library) and finally implement a supervised learning model to conclude the predictions.

After predicting gender, it would be also interesting to infer other customer information such as the age, the residential address, purchase habits, and more. Those information may reveal themselves crucial when tailoring products or services to clients. <br/>
If those data points are not directly accessible, one might look at other features in order to perform perdictions (e.g. the product category might be a strong predictor to target customer age, or the frequency of orders per day of the week might give a good sense of customer purchase habits).

## 8. Last words

I would like to thank Global Fashion Group for giving me the opportunity to work on this interesting problem. It was an intensive yet very educational exercise.

## 9. References
##### Exploring alternatives: Reading on clustering and semi-supervised learning
1. Supervised K-means clustering https://bit.ly/3aKHBD6
2. Stackoverflow research http://bit.ly/3buaum0
3. Transductive learning and Inductive learning http://bit.ly/3skdus1
4. Airbnb case http://bit.ly/2MkxuLQ

##### Interesting articles on customers' gender topic
4. Products more expensive for women than for men http://bit.ly/3bwFYIb
5. Different types of customer segmentation http://bit.ly/3dEv4mr
