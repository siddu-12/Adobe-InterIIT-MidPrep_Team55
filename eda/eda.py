import matplotlib.pyplot as plt
import nltk
import pandas as pd
import pmdarima as pm

df = pd.read_excel('content_simulation_train.xlsx')
# Sort the DataFrame by the 'date' column
df.sort_values(by='date', inplace=True)
print("Size of data", df.shape)
df.head()

# We must preprocess the original dataset to get it into an optimal format.
# This is only performed once and then used by all future experiments.
# In this step, we loaded the provided TSV files, imputed NaNs, factorized categorical variables, \
# optimized variable data types, processed lists into columns, decoded BERT tokens into text, extracted text features (e.g., word counts), and computed a dozen of count features related to engaging and engaged users.
# As this was a one-time preprocessing step, we did not include it in our comparison in Figure 1


print("Count of unique values in username", len(df['username'].value_counts()))
print("Count of unique values in infered company", len(df['inferred company'].value_counts()))

company = df['inferred company'][0]
print("All the usernames with infered company as", company)
df[df['inferred company'] == company].head()

username_list = df[df['inferred company'] == company]['username'].value_counts()
print(username_list)
username_list = username_list.index.to_list()

# Filter the DataFrame based on the usernames list
filtered_df = df[df['username'].isin(username_list)]

# Calculate mean, median, and range for 'likes' column for each 'username'
result = filtered_df.groupby('username')['likes'].agg(
    ['mean', 'median', lambda x: x.max(), lambda x: x.min()]).reset_index()
result.columns = ['username', 'mean_likes', 'median_likes', 'like_max', 'like_min']

print(result.head())

# Plot the count of likes for each username
plt.figure(figsize=(10, 6))
for username in username_list:
    user_data = filtered_df[filtered_df['username'] == username]
    # plt.plot(user_data['date'], user_data['likes']/user_data['likes'].mean(), label=username)
    plt.plot(user_data['date'], (user_data['likes'] - user_data['likes'].mean()) / user_data['likes'].std(),
             label=username)

plt.title('Likes Count for Usernames')
plt.xlabel('Index')
plt.ylabel('Likes Count')
plt.legend()
plt.show()

all_usernames = df['username'].value_counts()
all_usernames = all_usernames.index.to_list()


def plotPeaks(index):
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]

    # Sort the DataFrame by the 'date' column
    username_filtered_df.sort_values(by='date', inplace=True)

    # Find peaks in the likes column
    peaks, _ = find_peaks(username_filtered_df['likes'], prominence=0.1)
    prominences = peak_prominences(username_filtered_df['likes'], peaks)[0]

    # Get the indices of the top 10 peaks based on prominence
    top_10_indices = peaks[prominences.argsort()[-10:][::-1]]

    # Plot the signal and highlight the top 10 peaks
    print(username_list[index])
    plt.figure(figsize=(12, 6))
    toPlot = pd.DataFrame({'date': username_filtered_df['date'].iloc[top_10_indices],
                           'likes': username_filtered_df['likes'].iloc[top_10_indices]})
    # Sort the DataFrame based on the values in the first column
    toPlot = toPlot.sort_values(by='date')
    plt.plot(toPlot['date'], toPlot['likes'], 'o', label='Top 10 Peaks', markersize=10)
    plt.xticks(fontsize=6)
    plt.legend()
    plt.show()


for i in range(len(username_list)):
    plotPeaks(i)


def getPopularWords(index):
    # Sort filtered_df according to the 'likes' column
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]
    username_filtered_df = username_filtered_df.sort_values(by='likes', ascending=False)
    username_filtered_df = username_filtered_df.iloc[:100]

    # Combine 'content' column into a single string
    all_text = ' '.join(username_filtered_df['content'])

    # Tokenize the text
    tokens = word_tokenize(all_text)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word occurrences
    word_counts = Counter(tokens)

    # Get the top N common words
    top_words = word_counts.most_common(10)  # Change 10 to the desired number of top words

    # Print the top words
    print("Top Common Words:")
    print("Username", username_list[index])
    for word, count in top_words:
        print(f"{word}: {count}")
    # find company name hiddent in the tweet or mentions


nltk.download('stopwords')
nltk.download('punkt')
for i in range(len(username_list)):
    getPopularWords(i)


def find_theme(tweet):
    # Tokenize the tweet
    tokens = word_tokenize(tweet)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word occurrences
    word_counts = Counter(tokens)

    # Get the most common word as the theme
    if word_counts:
        theme = word_counts.most_common(1)[0][0]
        return theme
    else:
        return None


def getPopularThemes(content):
    # Combine 'content' column into a single string
    all_text = ' '.join(content)

    # Tokenize the text
    tokens = word_tokenize(all_text)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word occurrences
    word_counts = Counter(tokens)

    # Get the top N common words
    top_words = word_counts.most_common(10)  # Change 10 to the desired number of top words

    # Print the top words
    print("Top Common Words:")

    for word, count in top_words:
        print(f"{word}: {count}")
    # find company name hiddent in the tweet or mentions


def getThemeOfUsername(index, order):
    # Sort filtered_df according to the 'likes' column
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]
    username_filtered_df = username_filtered_df.sort_values(by='likes', ascending=False)
    if order == 'high':
        username_filtered_df = username_filtered_df.iloc[:100]
    elif order == 'low':
        username_filtered_df = username_filtered_df.iloc[-100:]
    print(username_list[index])
    content = []
    for tweet in username_filtered_df['content']:
        theme = find_theme(tweet)
        content.append(theme)

        # print(f"Tweet: Theme - {theme}")
    getPopularThemes(content)


# For most liked tweets

nltk.download('punkt')
# high words
for i in range(len(username_list)):
    getThemeOfUsername(i, "high")

# low themes
for i in range(len(username_list)):
    getThemeOfUsername(i, "low")


# check if the tweets with more hyperlinks and mentions have more likes?
# tweet length, has hashtags, if images are same , more likes on videos
def getCountOfWord(index, word_to_count):
    # Sort filtered_df according to the 'likes' column
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]
    username_filtered_df = username_filtered_df.sort_values(by='date', ascending=True)
    username_filtered_df = username_filtered_df.sort_values(by='likes', ascending=False)

    word_count = []
    likes_mean = []

    rows_per_group = 30

    row_groups = [username_filtered_df.iloc[i:i + rows_per_group, :] for i in
                  range(0, len(username_filtered_df), rows_per_group)]

    # Print information about each group
    cnt = 0
    for i, group in enumerate(row_groups, start=1):
        # print(f"Group {i}: Rows {group.index}")
        # word_to_count = 'mention'

        # Count occurrences of the word in the 'tweets' column
        count_of_word = group['content'].str.contains(word_to_count, case=False).sum()
        word_count.append(count_of_word)
        likes_mean.append(group['likes'].median())
        # Print the count
        # print(f"The word '{word_to_count}' appears {count_of_word} times.")
    # Plot the histogram

    print("Username", username_list[index])
    print(word_count)
    x = likes_mean
    # print(row_groups)
    plt.scatter(x, word_count, color='blue', marker='o')
    plt.title('Histogram of word_count')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


# usage of mention
for i in range(len(username_list)):
    getCountOfWord(i, 'mention')

# usage of hyperlink
for i in range(len(username_list)):
    getCountOfWord(i, 'hyperlink')

# usage of hashtags
for i in range(len(username_list)):
    getCountOfWord(i, '#')


# how per tweet likes change on mention


def WordCountSort(index, word_to_count):
    # Sort filtered_df according to the 'likes' column
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]
    # Create a 'count_of_mentions' column
    username_filtered_df['count_of_mentions'] = username_filtered_df['content'].str.count(word_to_count)
    username_filtered_df = username_filtered_df.sort_values(by='count_of_mentions', ascending=True)
    # Group by 'count_of_mentions' and calculate the mean of 'likes'
    mean_likes_by_mentions = username_filtered_df.groupby('count_of_mentions')['likes'].mean().reset_index()

    # Print the result
    print(mean_likes_by_mentions)


# usage of mention
# Plot the count of likes for each username

for i in range(len(username_list)):
    print("Username", username_list[i])
    WordCountSort(i, 'mention')

for i in range(len(username_list)):
    print("Username", username_list[i])
    WordCountSort(i, '#')


def getPopularHashtags(index):
    # Sort filtered_df according to the 'likes' column
    username_filtered_df = filtered_df[filtered_df['username'] == username_list[index]]

    # Extract hashtags using a regular expression and create a new DataFrame
    hashtags_df = username_filtered_df['content'].str.extractall(r'(#\w+)').reset_index(level=1, drop=True).rename(
        columns={0: 'hashtags'})

    # Merge the original DataFrame with the hashtags DataFrame
    result_df = pd.merge(username_filtered_df, hashtags_df, left_index=True, right_index=True)

    # Calculate average likes and count of occurrences for each hashtag
    agg_df = result_df.groupby('hashtags').agg({'likes': 'mean', 'content': 'count'}).reset_index()
    agg_df = agg_df.rename(columns={'likes': 'average_likes', 'content': 'occurrences'})

    # Merge the aggregated data back to the hashtags DataFrame
    hashtags_df = pd.merge(hashtags_df, agg_df, on='hashtags')
    # Remove duplicate rows based on the 'hashtags' column
    hashtags_df = hashtags_df.drop_duplicates(subset='hashtags')
    hashtags_df = hashtags_df.sort_values(by='average_likes', ascending=False)
    # Print the result DataFrame with hashtags, average likes, and occurrences
    print(hashtags_df)


# all the one posted on important days like valentine day and father day
for i in range(len(username_list)):
    print("Username", username_list[i])
    getPopularHashtags(i)

# Applying Time Series


username_filtered_df = filtered_df[filtered_df['username'] == username_list[3]]
time_series = username_filtered_df[['date', 'likes']]

# Assuming 'date' is your date column
time_series['date'] = pd.to_datetime(time_series['date'])
time_series = time_series.set_index('date')
# Resample to daily frequency and fill missing values
df_resampled = time_series.resample('D').mean().fillna(method='ffill')
# Find day of the week and day of the month
df_resampled['day_of_week'] = df_resampled.index.dayofweek
df_resampled['day_of_month'] = df_resampled.index.day
print(df_resampled.head())

# Seasonal - fit stepwise auto-ARIMA
SARIMAX_model = pm.auto_arima(df_resampled["likes"][:950], start_p=1, start_q=1,
                              test='adf',
                              max_p=3, max_q=3,
                              m=12,  # 12 is the frequncy of the cycle
                              start_P=0,
                              seasonal=True,  # set to seasonal
                              d=None,
                              D=1,  # order of the seasonal differencing
                              trace=False, exogenous=df_resampled[:950][['day_of_week', 'day_of_month']],
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)


def forecast(ARIMA_model, periods=140):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(time_series.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='D')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(time_series["likes"], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    # plt.fill_between(lower_series.index,
    # lower_series,
    # upper_series,
    # color='k', alpha=.15)

    plt.title("SARIMA - Forecast of Number of orderes")
    plt.show()


forecast(SARIMAX_model)

forecast(SARIMAX_model)

forecast(SARIMAX_model)

# 1. Check URLs
# 2. Check time series for seasnality and trend
# 3. Text


# URLs and Media


df_media = df

df['media'][3]

# Create a new DataFrame df_media based on the 'media' column
df_media = df

df_media['type_of_content'] = df_media['media'].apply(lambda x: x[1])
# For 'Video' type of media, extract additional information
if 'V' in df['type_of_media'].values:
    df['thumbnail_url'] = df_media[df['type_of_media'] == 'V'].apply(
        lambda x: x[0]['thumbnailUrl'] if x is not None and x[0] and 'thumbnailUrl' in x[0] else None)
    df['content_type'] = 'Video'
    df['url'] = df_media[df['type_of_media'] == 'Video'].apply(
        lambda x: x[0]['url'] if x is not None and x[0] and 'url' in x[0] else None)

# For 'Photo' type of media
if 'Photo' in df['type_of_media'].values:
    df['thumbnail_url'] = df_media[df['type_of_media'] == 'Photo'].apply(
        lambda x: x[0]['thumbnailUrl'] if x is not None and x[0] and 'thumbnailUrl' in x[0] else None)
    df['content_type'] = 'Photo'
    df['url'] = df_media[df['type_of_media'] == 'Photo'].apply(
        lambda x: x[0]['url'] if x is not None and x[0] and 'url' in x[0] else None)

# For 'GIF' type of media
if 'GIF' in df['type_of_media'].values:
    df['thumbnail_url'] = df_media[df['type_of_media'] == 'GIF'].apply(
        lambda x: x[0]['thumbnailUrl'] if x is not None and x[0] and 'thumbnailUrl' in x[0] else None)
    df['content_type'] = 'GIF'
    df['url'] = df_media[df['type_of_media'] == 'GIF'].apply(
        lambda x: x[0]['url'] if x is not None and x[0] and 'url' in x[0] else None)

df_media['media'][5]


def changeType(x):
    if x == 'P':
        return 'Photo'
    elif x == 'V':
        return 'Video'
    if x == 'G':
        return 'GIF'


df_media['type_of_content'] = df_media['type_of_content'].apply(lambda x: changeType(x))
df_media.head()

# Calculate the mean likes for each type of content
mean_likes_by_content_type = df_media_no_outliers.groupby('type_of_content')['likes'].mean()

# Plot the graph
mean_likes_by_content_type.plot(kind='bar', color='skyblue')
plt.title('Mean Likes by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Mean Likes')
plt.show()

# removing the outliers
from scipy.stats import zscore

# Calculate Z-scores for the 'likes' column
z_scores = zscore(df_media['likes'])

# Define a threshold for outliers (e.g., Z-score greater than 3 or less than -3)
threshold = 3

# Create a boolean mask to identify outliers
outliers_mask = (abs(z_scores) > threshold)

# Remove outliers from the DataFrame
df_media_no_outliers = df_media[~outliers_mask]

# Display the original DataFrame shape and the shape after removing outliers
print(f"Original DataFrame shape: {df_media.shape}")
print(f"DataFrame shape after removing outliers: {df_media_no_outliers.shape}")


def getUrls(media, urlType):
    # Define the start and end strings
    start_string = urlType + "='"
    end_string = "'"

    # Use regular expression to find the substring between start and end strings
    result = re.search(f'{re.escape(start_string)}(.*?){re.escape(end_string)}', media)

    # Check if the pattern was found
    if result:
        extracted_text = result.group(1)
        return extracted_text
    else:
        return None


def getAttr(media, attr, end_string):
    # Define the start and end strings
    start_string = attr + "="

    # Use regular expression to find the substring between start and end strings
    result = re.search(f'{re.escape(start_string)}(.*?){re.escape(end_string)}', media)

    # Check if the pattern was found
    if result:
        extracted_text = result.group(1)
        return extracted_text
    else:
        return None


df_media['preview_url'] = df_media['media'].apply(lambda x: getUrls(x, "previewUrl"))
df_media['full_url'] = df_media['media'].apply(lambda x: getUrls(x, "fullUrl"))
df_media['thumbnail_url'] = df_media['media'].apply(lambda x: getUrls(x, "thumbnailUrl"))
df_media['video_url'] = df_media['media'].apply(lambda x: getUrls(x, "url"))
df_media['video_duration'] = df_media['media'].apply(lambda x: getAttr(x, "duration", ","))
df_media['video_view'] = df_media['media'].apply(lambda x: getAttr(x, "views", ")"))

df_media.head()

df_media_no_outliers['preview_url'] = df_media_no_outliers['media'].apply(lambda x: getUrls(x, "previewUrl"))
df_media_no_outliers['full_url'] = df_media_no_outliers['media'].apply(lambda x: getUrls(x, "fullUrl"))
df_media_no_outliers['thumbnail_url'] = df_media_no_outliers['media'].apply(lambda x: getUrls(x, "thumbnailUrl"))
df_media_no_outliers['video_url'] = df_media_no_outliers['media'].apply(lambda x: getUrls(x, "url"))
df_media_no_outliers['video_duration'] = df_media_no_outliers['media'].apply(lambda x: getAttr(x, "duration", ","))
df_media_no_outliers['video_view'] = df_media_no_outliers['media'].apply(lambda x: getAttr(x, "views", ")"))

df_media_no_outliers.head()

# Assuming 'likes' and 'video views' are columns in the DataFrame

likes = df_media_no_outliers[df_media_no_outliers['type_of_content'] == 'Video']['likes']
video_views = df_media_no_outliers[df_media_no_outliers['type_of_content'] == 'Video']['video_view']

# Plot the graph
plt.scatter(likes, video_views, color='blue', alpha=0.5)
plt.title('Likes vs Video Views')
plt.xlabel('Likes')
plt.ylabel('Video Views')
plt.show()

video_df = df_media_no_outliers[df_media_no_outliers['type_of_content'] == 'Video']

# Convert the 'video_view' column to a pandas Series
likes = pd.Series(video_df['likes'])
views = pd.Series(video_df['video_view'])
duration = pd.Series(video_df['video_duration'])

# # Calculate the correlation between the two columns
# # Replace all occurrences of None with NaN in a specific column (replace 'your_column' with the actual column name)
views.replace('None', pd.NaT, inplace=True)
# # Replace NaN with 0 in the specific column
views.fillna(0, inplace=True)

correlation = likes.corr(views.astype(int))

# Print the correlation coefficient
print(f'Correlation between {likes} and {video_views}: {correlation}')

# # Calculate the correlation between the two columns
# # Replace all occurrences of None with NaN in a specific column (replace 'your_column' with the actual column name)
duration.replace('None', pd.NaT, inplace=True)
# # Replace NaN with 0 in the specific column
duration.fillna(0, inplace=True)

correlation = likes.corr(duration.astype(float))

# Print the correlation coefficient
print(f'Correlation between {likes} and {duration}: {correlation}')

# # Calculate the correlation between the two columns
# # Replace all occurrences of None with NaN in a specific column (replace 'your_column' with the actual column name)
duration.replace('None', pd.NaT, inplace=True)
# # Replace NaN with 0 in the specific column
duration.fillna(0, inplace=True)

correlation = views.astype(int).corr(duration.astype(float))

# Print the correlation coefficient
print(f'Correlation between {views} and {duration}: {correlation}')

# check how many urls exists
print("Total Photos in dataset", df_media[df_media['type_of_content'] == 'Photo'].shape[0])
print("Total Unique preview url in dataset", len(df_media['preview_url'].unique()))
print("Total Unique full url in dataset", len(df_media['full_url'].unique()))
print("Total videos in dataset", df_media[df_media['type_of_content'] == 'Video'].shape[0])
print("Total Unique thumbnail url in dataset", len(df_media['thumbnail_url'].unique()))

print("Total Unique preview url in dataset", df_media['preview_url'].value_counts())


def tweetLength(x):
    return len(x)


df_media_no_outliers['tweet_length'] = df_media_no_outliers['content'].apply(lambda x: tweetLength(x))
df_media['tweet_length'] = df_media['content'].apply(lambda x: tweetLength(x))
correlation = df_media_no_outliers['likes'].corr(df_media_no_outliers['tweet_length'])

# Print the correlation coefficient
print(f'Correlation between {df_media_no_outliers["likes"]} and {df_media_no_outliers["tweet_length"]}: {correlation}')



# Load the Drive helper and mount
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re

df = pd.read_excel('/content/drive/MyDrive/Inter IIT Tech Meet 2023/content_simulation_train.xlsx')
# Sort the DataFrame by the 'date' column
# df.sort_values(by='date', inplace=True)
print("Size of data", df.shape)
df.head()


def getUrls(media, urlType):
    # Define the start and end strings
    start_string = urlType + "='"
    end_string = "'"

    # Use regular expression to find the substring between start and end strings
    result = re.search(f'{re.escape(start_string)}(.*?){re.escape(end_string)}', media)

    # Check if the pattern was found
    if result:
        extracted_text = result.group(1)
        return extracted_text
    else:
        return None


def getAttr(media, attr, end_string):
    # Define the start and end strings
    start_string = attr + "="

    # Use regular expression to find the substring between start and end strings
    result = re.search(f'{re.escape(start_string)}(.*?){re.escape(end_string)}', media)

    # Check if the pattern was found
    if result:
        extracted_text = result.group(1)
        return extracted_text
    else:
        return None


import nltk

nltk.download('punkt')
nltk.download('stopwords')


def find_theme(tweet):
    # Tokenize the tweet
    tokens = word_tokenize(tweet)

    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Count word occurrences
    word_counts = Counter(tokens)

    # Get the most common word as the theme
    if word_counts:
        theme = word_counts.most_common(1)[0][0]
        return theme
    else:
        return None


df['preview_url'] = df['media'].apply(lambda x: getUrls(x, "previewUrl"))
df['full_url'] = df['media'].apply(lambda x: getUrls(x, "fullUrl"))
df['thumbnail_url'] = df['media'].apply(lambda x: getUrls(x, "thumbnailUrl"))
df['video_url'] = df['media'].apply(lambda x: getUrls(x, "url"))
df['video_duration'] = df['media'].apply(lambda x: getAttr(x, "duration", ","))
df['video_view'] = df['media'].apply(lambda x: getAttr(x, "views", ")"))

df.head()

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['theme'] = df['content'].apply(lambda x: find_theme(x))
df.head()

df['hashtags'] = df['content'].apply(lambda x: re.findall(r'#\w+', x))
df.head()

df['type_of_content'] = df['media'].apply(lambda x: x[1])


def changeType(x):
    if x == 'P':
        return 'Photo'
    elif x == 'V':
        return 'Video'
    if x == 'G':
        return 'GIF'


df['type_of_content'] = df['type_of_content'].apply(lambda x: changeType(x))
df.head()

df['date'] = pd.to_datetime(df['date'])
# Aggregate likes for each company
likes_sum_per_company = df.groupby(['inferred company', 'date'])['likes'].sum().reset_index()

# Display the result
print(likes_sum_per_company)
# Sort and select top 10 companies
top_10_companies = likes_sum_per_company.groupby('inferred company')['likes'].sum().nlargest(10).index

# Plot time series for the top 10 companies on one graph
plt.figure(figsize=(12, 8))

for company in top_10_companies:
    company_data = likes_sum_per_company[
        (likes_sum_per_company['inferred company'] == company) & (likes_sum_per_company['date'].dt.year == 2020)]
    plt.plot(company_data['date'], (company_data['likes'] - company_data['likes'].mean()) / company_data['likes'].std(),
             label=company)

plt.title('Time Series for Top 10 Companies')
plt.xlabel('Timestamp')
plt.ylabel('Likes')
plt.legend()
plt.show()

top_10_companies[0]

from statsmodels.tsa.seasonal import seasonal_decompose

result1 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[0])]['likes'],
    model='additive', period=12)
result2 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[1])]['likes'],
    model='additive', period=12)

# Plot the decomposed components
result1.plot()
plt.suptitle('Seasonal Decomposition - Series 1')
plt.show()

result2.plot()
plt.suptitle('Seasonal Decomposition - Series 2')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result1 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[2])]['likes'],
    model='additive', period=12)
result2 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[3])]['likes'],
    model='additive', period=12)

# Plot the decomposed components
result1.plot()
plt.suptitle('Seasonal Decomposition - Series 1')
plt.show()

result2.plot()
plt.suptitle('Seasonal Decomposition - Series 2')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result1 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[3])]['likes'],
    model='additive', period=12)
result2 = seasonal_decompose(
    likes_sum_per_company[(likes_sum_per_company['inferred company'] == top_10_companies[4])]['likes'],
    model='additive', period=12)

# Plot the decomposed components
result1.plot()
plt.suptitle('Seasonal Decomposition - Series 1')
plt.show()

result2.plot()
plt.suptitle('Seasonal Decomposition - Series 2')
plt.show()

from statsmodels.tsa.stattools import cross_correlation

lags, crosscorr = cross_correlation(df1['value'], df2['value'])
plt.stem(lags, crosscorr)
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.title('Cross-Correlation Function')
plt.show()
