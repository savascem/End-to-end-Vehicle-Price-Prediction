import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pprint import pprint


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def get_car_details(car_url):
    car_details = {}
    response = requests.get(car_url)
    soup = BeautifulSoup(response.text, "html.parser")

    car_name = soup.select_one("h1.listing-title")
    car_details["Car Name"] = car_name.text.strip() if car_name else "N/A"

    details = soup.select("dl.fancy-description-list")
    for detail in details:
        items = detail.find_all(["dt", "dd"])
        for i in range(0, len(items), 2):
            if items[i].name == "dt" and (i+1 < len(items) and items[i+1].name == "dd"):
                key = items[i].text.strip()
                value = items[i+1].text.strip()
                car_details[key] = value

    # get car price
    price_section = soup.select_one("div.price-section")
    if price_section:
        primary_price = price_section.select_one("span.primary-price")
        secondary_price = price_section.select_one("span.secondary-price")
        car_details["Primary Price"] = primary_price.text.strip() if primary_price else "N/A"
        car_details["Secondary Price"] = secondary_price.text.strip() if secondary_price else "N/A"


        # create car ID
    car_details["Car ID"] = car_details["Car Name"] + "_" + str(car_details["Primary Price"])

    return car_details


car_ids = set()
car_data = []


def get_car_links(soup):
    car_links = []
    cars = soup.select("a.vehicle-card-link")
    for car in cars:
        car_links.append('https://www.cars.com' + car['href'])
    return car_links


def get_next_page(soup):
    next_page = soup.select_one('a#next_paginate')
    if next_page:
        return 'https://www.cars.com' + next_page['href']
    else:
        return None


car_data = []
max_page = 1
current_page = 1
car_makes = ["ford"]


for car_make in car_makes:
    url = f'https://www.cars.com/shopping/results/?dealer_id=&keyword=&list_price_max=&list_price_min=&makes[]={car_make}&maximum_distance=all&mileage_max=&page_size=20&sort=best_match_desc&stock_type=used&year_max=&year_min=&zip='
    car_data = []
    current_page = 1

    while url and (not max_page or current_page <= max_page):
        print("###################################################")
        print(f"Okunan sayfa: {current_page}")
        print("###################################################")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        car_links = get_car_links(soup)

        for link in car_links:
            try:
                car_details = get_car_details(link)
                # Eğer bu araç daha önce görülmediyse, verilere ekleyin
                if car_details["Car ID"] not in car_ids:
                    car_data.append(car_details)
                    car_ids.add(car_details["Car ID"])
                    print(car_details.get('Car Name', 'Unknown Car'))
                else:
                    print("Aynı araç atlandı: " + car_details["Car Name"])
            except Exception as e:
                print(f"Hata: {e} - {link}")
                continue
            time.sleep(1)

            # getting url for next page
        url = get_next_page(soup)
        current_page += 1

    df = pd.DataFrame(car_data)
    df.to_csv(f'Cars_{car_make}.csv', index=False)
    print(df.columns)
    print(df.shape)
    print(df.head())

df.to_csv(".\data\ford.csv")


