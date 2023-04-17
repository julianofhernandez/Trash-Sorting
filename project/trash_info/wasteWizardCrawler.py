# API request
import requests
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


def main():
    driver = load_selenium()
    names = get_names(driver)
    categories = get_categories(names)
    scores = get_scores(names)
    df = pd.DataFrame(
        list(zip(names, categories, scores)), columns=["item", "category", "score"]
    )
    df.to_csv("project/trash_info/waste_wizard.csv", index=False)
    driver.close()


def load_selenium():
    # requires Chrome be installed
    options = webdriver.ChromeOptions()
    options.headless = True
    options.page_load_strategy = "none"
    chrome_path = ChromeDriverManager().install()
    chrome_service = Service(chrome_path)
    driver = Chrome(options=options, service=chrome_service)
    driver.get(
        "http://www.cityofsacramento.org/public-works/rsw/waste-wizard#!rc-cpage=wizard_material_list"
    )
    time.sleep(5)
    return driver


def get_names(driver):
    names = []
    # 23 sections in material list
    for i in range(1, 24):
        sublist = driver.find_element(By.ID, "page-section-rows-" + str(i))
        items = sublist.find_elements(By.TAG_NAME, "li")

        for item in items:
            a_tag = item.find_element(By.TAG_NAME, "a")
            url = a_tag.get_attribute("href")
            name = a_tag.get_attribute("textContent")
            name = name.strip()
            names.append(name)
    return names


def get_categories(names):
    categories = []
    for name in names:
        category = searchForBin(name)
        print(name, category)
        categories.append(category)
    return categories


def get_scores(names):
    scores = []
    for name in names:
        score = searchForScore(name)
        scores.append(score)
    return scores


# Get suggestion list
# https://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest=paper&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289

# Use ID from suggestion list to search here
# https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/473087.json


def get_id(searchTerm):
    url = (
        "http://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest="
        + searchTerm
        + "&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289"
    )
    response = recollect_api(url)
    if len(response.json()) == 0:
        print("no id found for " + searchTerm)
        return None
    else:
        waste = response.json()[0]
        # print('Searching for: '+searchTerm+' but found '+waste['title']+'\n')
        return waste["id"]


def searchForBin(searchTerm):
    id = get_id(searchTerm)
    if id is not None:
        url = (
            "https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/"
            + id
            + ".json"
        )
        response = recollect_api(url)
        for section in response.json()["sections"]:
            try:
                if section["title"].lower() == "best option":
                    return section["rows"][0]["value"]
            except:
                pass
        return None


def searchForScore(searchTerm):
    url = (
        "http://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest="
        + searchTerm
        + "&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289"
    )
    response = recollect_api(url)
    if len(response.json()) == 0:
        return None
    else:
        return response.json()[0]["score"]


def searchForSpecialInstructions(searchTerm):
    id = get_id(searchTerm)
    if id is not None:
        url = (
            "https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/"
            + id
            + ".json"
        )
        response = recollect_api(url)
        for section in response.json()["sections"]:
            try:
                if section["title"].lower() == "special instructions":
                    return section["rows"][0]["value"]
            except:
                pass
        print("special instructions not found for:" + searchTerm)


def recollect_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response
    else:
        print("Failed response from API")


if __name__ == "__main__":
    main()
