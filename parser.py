import pandas as pd
import requests
from bs4 import BeautifulSoup

# Step 1: Loading existing dataset
existing_dataset = pd.read_csv('D:/Python/modules/Crypton/nn2/pricedata.csv')

# Step 2: Checking the last date in the dataset
last_date = existing_dataset['Date'].iloc[-1]

# Step 3: Parsing historical data from the website
url = ''
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Step 4: Reading data from the next day after the last observation to the current day
new_data = []
rows = soup.find_all('tr')  # Setting selector for the data table on the website
for row in rows:
    date = row.find('td', class_='date-column').text.strip()
    if date > last_date:  # Checking the date
        # Reading and process other columns (price, volume, etc.)
        price = row.find('td', class_='price-column').text.strip()
        volume = row.find('td', class_='volume-column').text.strip()
        # Adding new data row
        new_data.append({'Date': date, 'Price': price, 'Volume': volume})

# Step 5: Updating dataset and save
new_dataset = existing_dataset.append(new_data, ignore_index=True)
new_dataset.to_csv('updated_historical_data.csv', index=False)
